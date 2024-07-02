import os
import random

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np

np.set_printoptions(precision=3)
import csv
import pickle
import json
import logging
import glob
import math
import torch
import quaternion
import matplotlib.pyplot as plt
import matplotlib.image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis, quat_from_two_vectors, quat_to_angle_axis
from src.habitat import (
    make_semantic_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
    get_quaternion,
)
from src.geom import get_cam_intr, get_scene_bnds, get_collision_distance
from src.tsdf_rollout import TSDFPlanner, Frontier, Object
from src.eval_utils import prepare_step_dict, get_item, encode, load_scene_features, rgba2rgb, load_checkpoint, collate_wrapper, construct_selection_prompt
from inference.models import YOLOWorld

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from easydict import EasyDict

def infer_prefilter(model, tokenizer, sample):
    # return prefiltered object list
    filter_input_ids = sample.filter_input_ids.to("cuda")
    if len(torch.where(sample.filter_input_ids==22550)[1]) == 0:
        logging.info(f"invalid: no token 'answer'!")
        return None
    answer_ind = torch.where(sample.filter_input_ids==22550)[1][0].item()
    filter_input_ids = filter_input_ids[:, :answer_ind+2]
    with torch.no_grad():
        with torch.inference_mode() and torch.autocast(device_type="cuda"):
            filter_output_ids = model.generate(
                filter_input_ids,
                feature_dict=None,
                do_sample=False,
                max_new_tokens=100,
            )
    # parse the prefilter output
        filter_outputs = tokenizer.decode(filter_output_ids[0, filter_input_ids.shape[1]:]).replace("</s>", "").strip()
    # print("the output of prefiltering", filter_outputs)
    if filter_outputs == "No object available":
        return []
    else:
        filter_outputs = filter_outputs.split("\n")
        # print("parsed output of prefiltering", filter_outputs)
        return filter_outputs

def infer_selection(model, tokenizer, sample):
    feature_dict = EasyDict(
        scene_feature = sample.scene_feature.to("cuda"),
        scene_insert_loc = sample.scene_insert_loc,
        scene_length = sample.scene_length,
    )
    input_ids = sample.input_ids.to("cuda")
    if len(torch.where(sample.input_ids==22550)[1]) == 0:
        logging.info(f"invalid: no token 'answer'!")
        return None
    answer_ind = torch.where(sample.input_ids==22550)[1][0].item()
    input_ids = input_ids[:, :answer_ind+2]
    with torch.no_grad():
        with torch.inference_mode() and torch.autocast(device_type="cuda"):
            output_ids = model.generate(
                input_ids,
                feature_dict=feature_dict,
                do_sample=False,
                max_new_tokens=10,
            )
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace("</s>", "").strip()
    return outputs

def inference(model, tokenizer, step_dict, cfg):
    step_dict["use_prefiltering"] = cfg.prefiltering
    #step_dict["use_egocentric_views"] = cfg.egocentric_views
    #step_dict["use_action_memory"] = cfg.action_memory
    step_dict["top_k_categories"] = cfg.top_k_categories
    try:
        sample = get_item(
            tokenizer, step_dict
        )
    except:
        logging.info(f"Get item failed! (most likely no frontiers and no objects)")
        return None
    if cfg.prefiltering:
        filter_outputs = infer_prefilter(model,tokenizer,sample)
        if filter_outputs is None:
            return None
        selection_dict = sample.selection_dict[0]
        selection_input = construct_selection_prompt(
            tokenizer, 
            selection_dict.text_before_object,
            selection_dict.feature_before_object,
            selection_dict.frontier_text,
            selection_dict.frontier_features,
            selection_dict.object_info_dict,
            1024,
            True,
            filter_outputs,
            cfg.top_k_categories
        )
        sample = collate_wrapper([selection_input])
    outputs = infer_selection(model,tokenizer,sample)
    return outputs

def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # load object detection model
    detection_model = YOLOWorld(model_id=cfg.detection_model_name)

    # Load dataset
    questions_list = json.load(open(cfg.questions_list_path, "r"))
    total_questions = len(questions_list)
    # sort the data according to the question id
    questions_list = sorted(questions_list, key=lambda x: x['question_id'])

    print("load model")
    # Initialize LLaVA model
    # model_path = "liuhaotian/llava-v1.5-7b"
    model_path = "/work/pi_chuangg_umass_edu/yuncong/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device_map=None, add_multisensory_token=True)  
    # model = model.to("cuda")
    load_checkpoint(model, cfg.model_path)
    model = model.to("cuda")
    # model = None
    model.eval()
    print("finish loading model")

    # load success list and path length list
    if os.path.exists(os.path.join(str(cfg.output_dir), "success_list.pkl")):
        with open(os.path.join(str(cfg.output_dir), "success_list.pkl"), "rb") as f:
            success_list = pickle.load(f)
    else:
        success_list = []
    if os.path.exists(os.path.join(str(cfg.output_dir), "path_length_list.pkl")):
        with open(os.path.join(str(cfg.output_dir), "path_length_list.pkl"), "rb") as f:
            path_length_list = pickle.load(f)
    else:
        path_length_list = {}

    success_count = 0
    max_target_observation = cfg.max_target_observation

    # Run all questions
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data['question_id']
        question = question_data['question']
        answer = question_data['answer']

        # Extract question
        scene_id = question_data["episode_history"]
        # TODO: leave this scene out for now (no extracted features)
        if scene_id in ["00802-wcojb4TFT35"]:
            continue
        init_pts = question_data["position"]
        init_quat = quaternion.quaternion(*question_data["rotation"])
        logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id}")

        # load scene
        split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
        scene_mesh_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.glb")
        navmesh_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.navmesh")
        semantic_texture_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".semantic.glb")
        scene_semantic_annotation_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".semantic.txt")
        assert os.path.exists(scene_mesh_path) and os.path.exists(navmesh_path), f'{scene_mesh_path}, {navmesh_path}'
        assert os.path.exists(semantic_texture_path) and os.path.exists(scene_semantic_annotation_path), f'{semantic_texture_path}, {scene_semantic_annotation_path}'

        try:
            del tsdf_planner
        except:
            logging.info("No planner to delete")
            pass

        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            logging.info("No simulator to close")
            pass

        sim_settings = {
            "scene": scene_mesh_path,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": cfg.hfov,
            "scene_dataset_config_file": cfg.scene_dataset_config_path,
        }
        sim_cfg = make_semantic_cfg(sim_settings)
        simulator = habitat_sim.Simulator(sim_cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        pathfinder.load_nav_mesh(navmesh_path)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        logging.info(f"Load scene {scene_id} successfully")

        bbox_path = os.path.join(cfg.semantic_bbox_data_path, scene_id + ".json")
        if not os.path.exists(bbox_path):
            logging.info(f"Question id {scene_id} invalid: no bbox data!")
            continue
        bounding_box_data = json.load(open(bbox_path, "r"))
        object_id_to_bbox = {int(item['id']): {'bbox': item['bbox'], 'class': item['class_name']} for item in bounding_box_data}
        object_id_to_name = {int(item['id']): item['class_name'] for item in bounding_box_data}

        scene_feature_map = load_scene_features(cfg.scene_features_path, scene_id)

        target_obj_id = 0
        episode_data_dir = os.path.join(str(cfg.output_dir), str(question_id))
        episode_observations_dir = os.path.join(episode_data_dir, 'observations')
        episode_object_observe_dir = os.path.join(episode_data_dir, 'object_observations')
        episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
        os.makedirs(episode_data_dir, exist_ok=True)
        os.makedirs(episode_observations_dir, exist_ok=True)
        os.makedirs(episode_object_observe_dir, exist_ok=True)
        os.makedirs(episode_frontier_dir, exist_ok=True)

        if len(os.listdir(episode_observations_dir)) >= 50:
            logging.info(f"Question id {question_id} already has enough target observations!")
            success_count += 1
            continue

        pts = init_pts
        angle, axis = quat_to_angle_axis(init_quat)
        angle = angle * axis[1] / np.abs(axis[1])
        rotation = get_quaternion(angle, camera_tilt)

        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
        num_step = max(num_step, 50)
        logging.info(
            f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
        )
        try:
            del tsdf_planner
        except:
            logging.info("No planner to delete")
            pass
        # initialize the TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pos_habitat_to_normal(pts),
            init_clearance=cfg.init_clearance * 2,
        )

        # record the history of the agent's path
        pts_pixs = np.empty((0, 2))
        pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(pts)[:2]))

        logging.info(f'\n\nQuestion id {question_id} initialization successful!')

        # run steps
        path_length = 0
        prev_pts = pts.copy()
        target_found = False
        cnt_step = -1
        target_observation_count = 0
        first_object_choice = None
        while cnt_step < num_step - 1:
            cnt_step += 1
            logging.info(f"\n== step: {cnt_step}")
            step_dict = {}
            angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
            total_views = 1 + cfg.extra_view_phase_1
            all_angles = [angle + angle_increment * (i - total_views // 2) for i in range(total_views)]
            # let the main viewing angle be the last one to avoid potential overwriting problems
            main_angle = all_angles.pop(total_views // 2)
            all_angles.append(main_angle)

            unoccupied_map, _ = tsdf_planner.get_island_around_pts(
                pts_normal, height=1.2
            )
            occupied_map = np.logical_not(unoccupied_map)

            # observe and update the TSDF
            keep_forward_observation = False
            observation_kept_count = 0
            for view_idx, ang in enumerate(all_angles):
                if cnt_step == 0:
                    keep_forward_observation = True  # at the first exploration step, always keep the forward observation
                if view_idx == total_views - 1 and observation_kept_count == 0:
                    keep_forward_observation = True  # if all previous observation is invalid, then we have to keep the forward one
                if pts_pixs.shape[0] >= 3:
                    if np.linalg.norm(pts_pixs[-1] - pts_pixs[-2]) < 1e-3 and np.linalg.norm(pts_pixs[-2] - pts_pixs[-3]) < 1e-3:
                        keep_forward_observation = True  # the agent is stuck somehow

                # check whether current view is valid
                collision_dist = tsdf_planner._voxel_size * get_collision_distance(
                    occupied_map,
                    pos=tsdf_planner.habitat2voxel(pts),
                    direction=tsdf_planner.rad2vector(ang)
                )
                if collision_dist < cfg.collision_dist:
                    if not (view_idx == total_views - 1 and keep_forward_observation):
                        # logging.info(f"Collision detected at step {cnt_step} view {view_idx}")
                        continue

                agent_state.position = pts
                agent_state.rotation = get_quaternion(ang, camera_tilt)
                agent.set_state(agent_state)
                pts_normal = pos_habitat_to_normal(pts)

                # Update camera info
                sensor = agent.get_state().sensor_states["depth_sensor"]
                quaternion_0 = sensor.rotation
                translation_0 = sensor.position
                cam_pose = np.eye(4)
                cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
                cam_pose[:3, 3] = translation_0
                cam_pose_normal = pose_habitat_to_normal(cam_pose)
                cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

                # Get observation at current pose - skip black image, meaning robot is outside the floor
                obs = simulator.get_sensor_observations()
                rgb = obs["color_sensor"]
                depth = obs["depth_sensor"]
                semantic_obs = obs["semantic_sensor"]
                if cfg.save_obs:
                    plt.imsave(
                        os.path.join(episode_observations_dir, "{}.png".format(cnt_step)), rgb
                    )

                # check whether the observation is valid
                keep_observation = True
                black_pix_ratio = np.sum(semantic_obs == 0) / (img_height * img_width)
                if black_pix_ratio > cfg.black_pixel_ratio:
                    keep_observation = False
                positive_depth = depth[depth > 0]
                if positive_depth.size == 0 or np.percentile(positive_depth, 30) < cfg.min_30_percentile_depth:
                    keep_observation = False
                if not keep_observation:
                    if not (view_idx == total_views - 1 and keep_forward_observation):
                        # logging.info(f"Invalid observation: black pixel ratio {black_pix_ratio}, 30 percentile depth {np.percentile(depth[depth > 0], 30)}")
                        continue

                # construct an frequency count map of each semantic id to a unique id
                with torch.no_grad():
                    target_in_view, annotated_rgb = tsdf_planner.update_scene_graph(
                        detection_model=detection_model,
                        rgb=rgb[..., :3],
                        semantic_obs=semantic_obs,
                        obj_id_to_name=object_id_to_name,
                        obj_id_to_bbox=object_id_to_bbox,
                        cfg=cfg.scene_graph,
                        target_obj_id=target_obj_id,
                        return_annotated=True
                    )

                # TSDF fusion
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=cam_pose_tsdf,
                    obs_weight=1.0,
                    margin_h=int(cfg.margin_h_ratio * img_height),
                    margin_w=int(cfg.margin_w_ratio * img_width),
                )

                observation_kept_count += 1

                # if target_found:
                #     break

            if target_found:
                break

            logging.info(f'length original scene graph {len(tsdf_planner.simple_scene_graph.keys())}')
            # remove keys in simple_scene_graph if key not found in scene
            for key in list(tsdf_planner.simple_scene_graph.keys()):
                scene_str_keys = [str(x) for x in scene_feature_map.keys()]
                obj_str_keys = [str(x) for x in object_id_to_name.keys()]
                if str(key) not in scene_str_keys or str(key) not in obj_str_keys:
                    del tsdf_planner.simple_scene_graph[key]
            logging.info(f'length updated scene graph {len(tsdf_planner.simple_scene_graph.keys())}')

            # record current scene graph
            step_dict["scene_graph"] = list(tsdf_planner.simple_scene_graph.keys())
            step_dict["scene_graph"] = [int(x) for x in step_dict["scene_graph"]]
            step_dict["obj_map"] = object_id_to_name

            update_success = tsdf_planner.update_frontier_map(pts=pts_normal, cfg=cfg.planner)
            if not update_success:
                logging.info("Warning! Update frontier map failed!")

            # Turn to face each frontier point and get rgb image
            for i, frontier in enumerate(tsdf_planner.frontiers):
                pos_voxel = frontier.position
                pos_world = pos_voxel * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
                pos_world = pos_normal_to_habitat(np.append(pos_world, floor_height))
                assert (frontier.image is None and frontier.feature is None) or (frontier.image is not None and frontier.feature is not None), f"{frontier.image}, {frontier.feature is None}"
                # Turn to face the frontier point
                if frontier.image is None:
                    view_frontier_direction = np.asarray([pos_world[0] - pts[0], 0., pos_world[2] - pts[2]])
                    default_view_direction = np.asarray([0., 0., -1.])
                    if np.linalg.norm(view_frontier_direction) < 1e-3:
                        view_frontier_direction = default_view_direction
                    if np.dot(view_frontier_direction, default_view_direction) / np.linalg.norm(view_frontier_direction) < -1 + 1e-3:
                        # if the rotation is to rotate 180 degree, then the quaternion is not unique
                        # we need to specify rotating along y-axis
                        agent_state.rotation = quat_to_coeffs(
                            quaternion.quaternion(0, 0, 1, 0)
                            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
                        ).tolist()
                    else:
                        agent_state.rotation = quat_to_coeffs(
                            quat_from_two_vectors(default_view_direction, view_frontier_direction)
                            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
                        ).tolist()
                    agent.set_state(agent_state)
                    # Get observation at current pose - skip black image, meaning robot is outside the floor
                    obs = simulator.get_sensor_observations()
                    rgb = obs["color_sensor"]
                    plt.imsave(
                        os.path.join(episode_frontier_dir, f"{cnt_step}_{i}.png"),
                        rgb,
                    )
                    processed_rgb = rgba2rgb(rgb)
                    with torch.no_grad():
                        img_feature = encode(model, image_processor, processed_rgb).mean(1)
                    assert img_feature is not None
                    frontier.image = f"{cnt_step}_{i}.png"
                    frontier.feature = img_feature

            if tsdf_planner.max_point is None and tsdf_planner.target_point is None:
                if first_object_choice is None:
                    # choose a frontier, and set it as the explore target
                    step_dict["frontiers"] = []
                    # since we skip the stuck frontier for input of the vlm, we need to map the
                    # vlm output frontier id to the tsdf planner frontier id
                    ft_id_to_vlm_id = {}
                    vlm_id_count = 0
                    for i, frontier in enumerate(tsdf_planner.frontiers):
                        if frontier.is_stuck:
                            continue
                        frontier_dict = {}
                        pos_voxel = frontier.position
                        pos_world = pos_voxel * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
                        pos_world = pos_normal_to_habitat(np.append(pos_world, floor_height))
                        frontier_dict["coordinate"] = pos_world.tolist()
                        assert frontier.image is not None and frontier.feature is not None
                        frontier_dict["rgb_feature"] = frontier.feature
                        frontier_dict["rgb_id"] = frontier.image

                        step_dict["frontiers"].append(frontier_dict)

                        ft_id_to_vlm_id[i] = vlm_id_count
                        vlm_id_count += 1
                    vlm_id_to_ft_id = {v: k for k, v in ft_id_to_vlm_id.items()}

                    # add model prediction here
                    if len(step_dict["frontiers"]) > 0:
                        step_dict["frontier_features"] = torch.cat(
                            [
                                frontier["rgb_feature"] for frontier in step_dict["frontiers"]
                            ],
                            dim=0
                        ).to("cpu")
                    else:
                        step_dict["frontier_features"] = None
                    step_dict["question"] = question
                    step_dict["scene"] = scene_id
                    step_dict["scene_feature_map"] = scene_feature_map

                    # jiachen TODO: encapsulate the following code into a function
                    '''
                    try:
                        sample = get_item(
                            tokenizer, step_dict
                        )
                    except:
                        logging.info(f"Get item failed! (most likely no frontiers and no objects)")
                        break
                    feature_dict = EasyDict(
                        scene_feature = sample.scene_feature.to("cuda"),
                        scene_insert_loc = sample.scene_insert_loc,
                        scene_length = sample.scene_length,
                    )
                    input_ids = sample.input_ids.to("cuda")
                    if len(torch.where(sample.input_ids==22550)[1]) == 0:
                        logging.info(f"Question id {question_id} invalid: no token 22550!")
                        break
                    answer_ind = torch.where(sample.input_ids==22550)[1][0].item()
                    input_ids = input_ids[:, :answer_ind+2]
                    with torch.no_grad():
                        with torch.inference_mode() and torch.autocast(device_type="cuda"):
                            output_ids = model.generate(
                                input_ids,
                                feature_dict=feature_dict,
                                do_sample=False,
                                max_new_tokens=10,
                            )
                        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace("</s>", "").strip()
                    '''
                    outputs = inference(model, tokenizer, step_dict, cfg)
                    if outputs is None:
                        # encounter generation error
                        logging.info(f"Question id {question_id} invalid: model generation error!")
                        break
                    ############################
                    try:
                        target_type, target_index = outputs.split(" ")[0], outputs.split(" ")[1]
                        print(f"Prediction: {target_type}, {target_index}")
                    except:
                        logging.info(f"Wrong output format, failed!")
                        break

                    if target_type not in ["object", "frontier"]:
                        logging.info(f"Invalid prediction type: {target_type}, failed!")
                        break

                    if target_type == "object":
                        if int(target_index) < 0 or int(target_index) >= len(tsdf_planner.simple_scene_graph):
                            logging.info(f"Prediction out of range: {target_index}, {len(tsdf_planner.simple_scene_graph)}, failed!")
                            break
                        pred_target_obj_id = list(tsdf_planner.simple_scene_graph.keys())[int(target_index)]
                        target_point = tsdf_planner.habitat2voxel(tsdf_planner.simple_scene_graph[pred_target_obj_id])[:2]
                        logging.info(f"Next choice: Object at {target_point}")
                        tsdf_planner.frontiers_weight = np.zeros((len(tsdf_planner.frontiers)))
                        max_point_choice = Object(target_point.astype(int), pred_target_obj_id)
                    else:
                        target_index = int(target_index)
                        if target_index not in vlm_id_to_ft_id.keys():
                            logging.info(f"Predicted frontier index invalid: {target_index}, failed!")
                            break
                        target_index = vlm_id_to_ft_id[target_index]
                        target_point = tsdf_planner.frontiers[target_index].position
                        logging.info(f"Next choice: Frontier at {target_point}")
                        tsdf_planner.frontiers_weight = np.zeros((len(tsdf_planner.frontiers)))
                        max_point_choice = tsdf_planner.frontiers[target_index]

                    if max_point_choice is None:
                        logging.info(f"Question id {question_id} invalid: no valid choice!")
                        break

                    if type(max_point_choice) == Object:
                        first_object_choice = max_point_choice
                else:
                    logging.info(f"Keep choosing object {first_object_choice}")
                    max_point_choice = first_object_choice

                update_success = tsdf_planner.set_next_navigation_point(
                    choice=max_point_choice,
                    pts=pts_normal,
                    cfg=cfg.planner,
                    pathfinder=pathfinder
                )
                if not update_success:
                    logging.info(f"Question id {question_id} invalid: find next navigation point failed!")
                    break

            if cfg.save_frontier_video:
                frontier_video_path = os.path.join(episode_data_dir, "frontier_video")
                os.makedirs(frontier_video_path, exist_ok=True)
                num_images = len(tsdf_planner.frontiers)
                side_length = int(np.sqrt(num_images)) + 1
                side_length = max(2, side_length)
                fig, axs = plt.subplots(side_length, side_length, figsize=(20, 20))
                for h_idx in range(side_length):
                    for w_idx in range(side_length):
                        axs[h_idx, w_idx].axis('off')
                        i = h_idx * side_length + w_idx
                        if i < num_images:
                            img_path = os.path.join(episode_frontier_dir, tsdf_planner.frontiers[i].image)
                            img = matplotlib.image.imread(img_path)
                            axs[h_idx, w_idx].imshow(img)
                            if type(max_point_choice) == Frontier and max_point_choice.image == tsdf_planner.frontiers[i].image:
                                axs[h_idx, w_idx].set_title('Chosen')
                global_caption = f"{question}\n{answer}"
                if type(max_point_choice) == Object:
                    global_caption += '\nToward target object'
                fig.suptitle(global_caption, fontsize=16)
                plt.tight_layout(rect=(0., 0., 1., 0.95))
                plt.savefig(os.path.join(frontier_video_path, f'{cnt_step}.png'))
                plt.close()

            return_values = tsdf_planner.agent_step(
                pts=pts_normal,
                angle=angle,
                pathfinder=pathfinder,
                cfg=cfg.planner,
                save_visualization=cfg.save_visualization,
            )
            if return_values[0] is None:
                logging.info(f"Question id {question_id} invalid: agent_step failed!")
                break
            pts_normal, angle, pts_pix, fig, target_arrived = return_values

            # update the agent's position record
            pts_pixs = np.vstack((pts_pixs, pts_pix))
            if cfg.save_visualization:
                # Add path to ax5, with colormap to indicate order
                visualization_path = os.path.join(episode_data_dir, "visualization")
                os.makedirs(visualization_path, exist_ok=True)
                ax5 = fig.axes[4]
                ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)

                fig.tight_layout()
                plt.savefig(os.path.join(visualization_path, "{}_map.png".format(cnt_step)))
                plt.close()

            # update position and rotation
            pts_normal = np.append(pts_normal, floor_height)
            pts = pos_normal_to_habitat(pts_normal)
            rotation = get_quaternion(angle, camera_tilt)

            logging.info(f"Current position: {pts}")
            path_length += float(np.linalg.norm(pts - prev_pts))
            prev_pts = pts.copy()

            if len(pts_pixs) >= 3 and np.linalg.norm(pts_pixs[-1] - pts_pixs[-2]) <= 1 and np.linalg.norm(pts_pixs[-2] - pts_pixs[-3]) <= 1:
                if type(max_point_choice) == Frontier:
                    logging.info(f"Question id {question_id} stuck at frontier {max_point_choice.position}!!!")
                    max_point_choice.is_stuck = True

            if target_type == "object" and target_arrived:
                # the model found the target object and arrived at a proper observation point
                # get an observation and save it
                # the returned position and orientation should directly point to the target object
                agent_state_obs = habitat_sim.AgentState()
                agent_state_obs.position = pts
                agent_state_obs.rotation = rotation
                agent.set_state(agent_state_obs)
                obs = simulator.get_sensor_observations()
                rgb = obs["color_sensor"]
                plt.imsave(
                    os.path.join(episode_object_observe_dir, f"target_{target_observation_count}.png"), rgb
                )
                target_observation_count += 1
                if target_observation_count >= max_target_observation:
                    target_found = True
                    break

        if target_found:
            success_count += 1
            # We only consider samples that model predicts object (use baseline results other samples for now)
            # TODO: you can save path_length in the same format as you did for the baseline
            if question_id not in success_list:
                success_list.append(question_id)
            path_length_list[question_id] = path_length
            logging.info(f"Question id {question_id} finish with {cnt_step} steps, {path_length} length")
        else:
            logging.info(f"Question id {question_id} failed, {path_length} length")
        logging.info(f"{question_idx + 1}/{total_questions}: Success rate: {success_count}/{question_idx + 1}")
        logging.info(f"Mean path length for success exploration: {np.mean(list(path_length_list.values()))}")
        # logging.info(f'Scene {scene_id} finish')

        # ensure that the observation dir has at most 50 images
        all_img_paths = glob.glob(os.path.join(episode_observations_dir, "*.png"))
        if len(all_img_paths) > 50:
            selected_img_paths = random.sample(all_img_paths, 50)
            for path in all_img_paths:
                if path not in selected_img_paths:
                    os.remove(path)

        with open(os.path.join(str(cfg.output_dir), "success_list.pkl"), "wb") as f:
            pickle.dump(success_list, f)
        with open(os.path.join(str(cfg.output_dir), "path_length_list.pkl"), "wb") as f:
            pickle.dump(path_length_list, f)

    with open(os.path.join(str(cfg.output_dir), "success_list.pkl"), "wb") as f:
        pickle.dump(success_list, f)
    with open(os.path.join(str(cfg.output_dir), "path_length_list.pkl"), "wb") as f:
        pickle.dump(path_length_list, f)

    logging.info(f'All scenes finish')
    try:
        simulator.close()
    except:
        pass


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(str(cfg.output_dir), "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg)
