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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis, quat_from_two_vectors, quat_to_angle_axis
from src.habitat import (
    make_simple_cfg,
    make_semantic_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
    get_quaternion,
    get_navigable_point_to
)
from src.geom import get_cam_intr, get_scene_bnds, get_collision_distance
from src.tsdf_rollout import TSDFPlanner, Frontier, Object
from src.eval_utils import prepare_step_dict, get_item, encode, load_scene_features, rgba2rgb, load_checkpoint
from habitat_sim.utils.common import d3_40_colors_rgb
from inference.models import YOLOWorld

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from easydict import EasyDict


def get_info(pathfinder, pos):
    is_navigable = pathfinder.is_navigable(pos)
    hit_record = pathfinder.closest_obstacle_surface_point(pos, 0.5)
    dist = hit_record.hit_dist
    return is_navigable, dist


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
    all_questions_list = os.listdir(cfg.path_data_dir)
    all_questions_list = [question_id for question_id in all_questions_list if 650 < int(question_id.split('-')[0]) < 750]
    total_questions = len(all_questions_list)
    all_scene_list = sorted(
        list(set(
            [question_id.split('_')[0] for question_id in all_questions_list]
        ))
    )
    logging.info(f"Loaded {len(all_questions_list)} questions in {len(all_scene_list)} scenes.")

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

    # for each scene, answer each question
    question_ind = 0
    success_count = 0

    success_list = []
    path_length_list = []

    for scene_id in all_scene_list:
        all_question_id_in_scene = [q for q in all_questions_list if scene_id in q]

        ##########################################################
        # rand_q = np.random.randint(0, len(all_questions_in_scene) - 1)
        # all_questions_in_scene = all_questions_in_scene[rand_q:rand_q+1]
        # all_questions_in_scene = [q for q in all_questions_in_scene if '00324' in q['question_id']]
        # if len(all_questions_in_scene) == 0:
        #     continue
        # random.shuffle(all_questions_in_scene)
        # all_questions_in_scene = all_questions_in_scene[:2]
        # all_questions_in_scene = [q for q in all_questions_in_scene if "00109" in q['question_id']]
        ##########################################################

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
            pass
        try:
            simulator.close()
        except:
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

        # load semantic object bbox data
        bounding_box_data = json.load(open(os.path.join(cfg.semantic_bbox_data_path, scene_id + ".json"), "r"))
        object_id_to_bbox = {int(item['id']): {'bbox': item['bbox'], 'class': item['class_name']} for item in bounding_box_data}
        object_id_to_name = {int(item['id']): item['class_name'] for item in bounding_box_data}

        scene_feature_map = load_scene_features(cfg.scene_features_path, scene_id)

        # Evaluate each question
        for question_id in all_question_id_in_scene:
            question_ind += 1
            metadata = json.load(open(os.path.join(cfg.path_data_dir, question_id, "metadata.json"), "r"))

            # load question data
            question = metadata["question"]
            init_pts = metadata["init_pts"]
            init_angle = metadata["init_angle"]
            target_obj_id = metadata['target_obj_id']
            target_obj_class = metadata['target_obj_class']
            # get target object global location
            obj_bbox = [item['bbox'] for item in bounding_box_data if int(item['id']) == target_obj_id][0]
            obj_bbox = np.asarray(obj_bbox)  # (2, 3)
            obj_bbox_center = np.mean(obj_bbox, axis=0)
            obj_bbox_center = obj_bbox_center[[0, 2, 1]]

            episode_data_dir = os.path.join(str(cfg.output_dir), question_id)
            episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
            os.makedirs(episode_data_dir, exist_ok=True)
            os.makedirs(episode_frontier_dir, exist_ok=True)

            pts = init_pts
            angle = init_angle
            rotation = get_quaternion(angle, camera_tilt)

            # initialize the TSDF
            pts_normal = pos_habitat_to_normal(pts)
            floor_height = pts_normal[-1]
            tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            logging.info(
                f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
            )
            try:
                del tsdf_planner
            except:
                pass
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height_offset=0,
                pts_init=pts_normal,
                init_clearance=cfg.init_clearance * 2,
            )

            target_center_voxel = tsdf_planner.world2vox(pos_habitat_to_normal(obj_bbox_center))
            # record the history of the agent's path
            pts_pixs = np.empty((0, 2))
            pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(pts)[:2]))

            logging.info(f'\n\nQuestion id {scene_id} initialization successful!')

            # run steps
            path_length = 0
            prev_pts = pts.copy()
            target_found = False
            cnt_step = -1
            while cnt_step < num_step - 1:
                cnt_step += 1
                logging.info(f"\n== step: {cnt_step}")
                step_dict = {}

                # for each position, get the views from different angles
                if target_obj_id in tsdf_planner.simple_scene_graph.keys():
                    angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                    total_views = 1 + cfg.extra_view_phase_2
                else:
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
                for view_idx, ang in enumerate(all_angles):
                    # logging.info(f"Step {cnt_step}, view {view_idx + 1}/{total_views}")

                    # check whether current view is valid
                    collision_dist = tsdf_planner._voxel_size * get_collision_distance(
                        occupied_map,
                        pos=tsdf_planner.habitat2voxel(pts),
                        direction=tsdf_planner.rad2vector(ang)
                    )
                    if collision_dist < cfg.collision_dist and view_idx != total_views - 1:  # the last view is the main view, and is not dropped
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

                    # check whether the observation is valid
                    keep_observation = True
                    black_pix_ratio = np.sum(semantic_obs == 0) / (img_height * img_width)
                    if black_pix_ratio > cfg.black_pixel_ratio:
                        keep_observation = False
                    positive_depth = depth[depth > 0]
                    if positive_depth.size == 0 or np.percentile(positive_depth, 30) < cfg.min_30_percentile_depth:
                        keep_observation = False
                    if not keep_observation and view_idx != total_views - 1:
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

                    # check stop condition
                    if target_in_view:
                        if target_obj_id in tsdf_planner.simple_scene_graph.keys():
                            target_obj_pix_ratio = np.sum(semantic_obs == target_obj_id) / (img_height * img_width)
                            if target_obj_pix_ratio > 0:
                                obj_pix_center = np.mean(np.argwhere(semantic_obs == target_obj_id), axis=0)
                                bias_from_center = (obj_pix_center - np.asarray([img_height // 2, img_width // 2])) / np.asarray([img_height, img_width])
                                # currently just consider that the object should be in around the horizontal center, not the vertical center
                                # due to the viewing angle difference
                                if target_obj_pix_ratio > cfg.stop_min_pix_ratio and np.abs(bias_from_center)[1] < cfg.stop_max_bias_from_center:
                                    logging.info(f"Stop condition met at step {cnt_step} view {view_idx}")
                                    target_found = True

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

                    if cfg.save_obs:
                        observation_save_dir = os.path.join(episode_data_dir, 'observations')
                        os.makedirs(observation_save_dir, exist_ok=True)
                        if target_found:
                            plt.imsave(os.path.join(observation_save_dir, f"{cnt_step}-view_{view_idx}-target.png"), annotated_rgb)
                        else:
                            plt.imsave(os.path.join(observation_save_dir, f"{cnt_step}-view_{view_idx}.png"), annotated_rgb)

                    if target_found:
                        break

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
                    logging.info(f"Question id {scene_id} invalid: update frontier map failed!")
                    break

                step_dict["frontiers"] = []
                # Seems buggy here
                for i, frontier in enumerate(tsdf_planner.frontiers):
                    frontier_dict = {}
                    pos_voxel = frontier.position
                    pos_world = pos_voxel * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
                    pos_world = pos_normal_to_habitat(np.append(pos_world, floor_height))
                    frontier_dict["coordinate"] = pos_world.tolist()
                    # Turn to face the frontier point
                    if frontier.image is not None:
                        frontier_dict["rgb_feature"] = frontier.feature
                        frontier_dict["rgb_id"] = frontier.image
                    else:
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
                        frontier_dict["rgb_feature"] = img_feature
                        frontier_dict["rgb_id"] = f"{cnt_step}_{i}.png"
                    step_dict["frontiers"].append(frontier_dict)
                    assert frontier_dict["rgb_feature"] is not None

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
                    target_obj_id = list(tsdf_planner.simple_scene_graph.keys())[int(target_index)]
                    target_point = tsdf_planner.habitat2voxel(tsdf_planner.simple_scene_graph[target_obj_id])[:2]
                    logging.info(f"Next choice: Object at {target_point}")
                    tsdf_planner.frontiers_weight = np.zeros((len(tsdf_planner.frontiers)))
                    max_point_choice = Object(target_point.astype(int), target_obj_id)
                else:
                    if int(target_index) < 0 or int(target_index) >= len(tsdf_planner.frontiers):
                        logging.info(f"Prediction out of range: {target_index}, {len(tsdf_planner.frontiers)}, failed!")
                        break
                    target_point = tsdf_planner.frontiers[int(target_index)].position
                    logging.info(f"Next choice: Frontier at {target_point}")
                    tsdf_planner.frontiers_weight = np.zeros((len(tsdf_planner.frontiers)))
                    max_point_choice = tsdf_planner.frontiers[int(target_index)]

                if max_point_choice is None:
                    logging.info(f"Question id {question_id} invalid: no valid choice!")
                    break

                return_values = tsdf_planner.get_next_navigation_point(
                    choice=max_point_choice,
                    pts=pts_normal,
                    angle=angle,
                    pathfinder=pathfinder,
                    cfg=cfg.planner,
                    save_visualization=cfg.save_visualization,
                )
                if return_values[0] is None:
                    logging.info(f"Question id {question_id} invalid: find next navigation point failed!")
                    break
                pts_normal, angle, pts_pix, fig = return_values

                # update the agent's position record
                pts_pixs = np.vstack((pts_pixs, pts_pix))
                if cfg.save_visualization:
                    # Add path to ax5, with colormap to indicate order
                    visualization_path = os.path.join(episode_data_dir, "visualization")
                    os.makedirs(visualization_path, exist_ok=True)
                    ax5 = fig.axes[4]
                    ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                    ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)

                    # add target object bbox
                    color = 'green' if target_obj_id in tsdf_planner.simple_scene_graph.keys() else 'red'
                    ax5.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s=120)
                    ax1, ax2, ax4 = fig.axes[0], fig.axes[1], fig.axes[3]
                    ax4.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s=120)
                    ax1.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s=120)
                    ax2.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s=120)

                    fig.tight_layout()
                    plt.savefig(os.path.join(visualization_path, "{}_map.png".format(cnt_step)))
                    plt.close()

                if cfg.save_frontier_video:
                    frontier_video_path = os.path.join(episode_data_dir, "frontier_video")
                    os.makedirs(frontier_video_path, exist_ok=True)
                    if type(max_point_choice) == Frontier:
                        img_path = os.path.join(episode_frontier_dir, max_point_choice.image)
                        os.system(f"cp {img_path} {os.path.join(frontier_video_path, f'{cnt_step:04d}-frontier.png')}")
                    else:  # navigating to the objects
                        if cfg.save_obs:
                            img_path = os.path.join(observation_save_dir, f"{cnt_step}-view_{total_views - 1}.png")
                            if os.path.exists(img_path):
                                os.system(f"cp {img_path} {os.path.join(frontier_video_path, f'{cnt_step:04d}-object.png')}")

                # update position and rotation
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)
                rotation = get_quaternion(angle, camera_tilt)

                logging.info(f"Current position: {pts}")
                path_length += float(np.linalg.norm(pts - prev_pts))
                prev_pts = pts.copy()

            if target_found:
                success_count += 1
                success_list.append(1)
                logging.info(f"Question id {question_id} finish with {cnt_step} steps, {path_length} length")
            else:
                success_list.append(0)
                logging.info(f"Question id {question_id} failed.")
            path_length_list.append(path_length)

            logging.info(f"{question_ind}/{total_questions}: Success rate: {success_count}/{question_ind}")

        logging.info(f'Scene {scene_id} finish')

    with open(os.path.join(cfg.output_dir, "success_list.pkl"), "wb") as f:
        pickle.dump(success_list, f)
    with open(os.path.join(cfg.output_dir, "path_length_list.pkl"), "wb") as f:
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
