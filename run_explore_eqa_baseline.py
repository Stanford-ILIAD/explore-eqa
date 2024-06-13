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
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from inference.models import YOLOWorld
import supervision as sv
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
from src.geom import get_cam_intr, get_scene_bnds, IoU
from src.vlm import VLM
from src.tsdf import TSDFPlanner


def update_scene_graph(detection_model, scene_objects, rgb, semantic_obs, obj_id_to_name, cfg, target_obj_id):
    target_found = False

    unique_obj_ids = np.unique(semantic_obs)
    class_to_obj_id = {}
    for obj_id in unique_obj_ids:
        if obj_id == 0 or obj_id not in obj_id_to_name.keys() or obj_id_to_name[obj_id] in ['wall', 'floor', 'ceiling']:
            continue
        if obj_id_to_name[obj_id] not in class_to_obj_id.keys():
            class_to_obj_id[obj_id_to_name[obj_id]] = [obj_id]
        else:
            class_to_obj_id[obj_id_to_name[obj_id]].append(obj_id)
    all_classes = list(class_to_obj_id.keys())

    if len(all_classes) == 0:
        return target_found, rgb

    detection_model.set_classes(all_classes)

    results = detection_model.infer(rgb, confidence=cfg.confidence)
    detections = sv.Detections.from_inference(results).with_nms(threshold=cfg.nms_threshold)

    adopted_indices = []
    for i in range(len(detections)):
        class_name = all_classes[detections.class_id[i]]
        x_start, y_start, x_end, y_end = detections.xyxy[i].astype(int)
        bbox_mask = np.zeros(semantic_obs.shape, dtype=bool)
        bbox_mask[y_start:y_end, x_start:x_end] = True
        for obj_id in class_to_obj_id[class_name]:
            obj_x_start, obj_y_start = np.argwhere(semantic_obs == obj_id).min(axis=0)
            obj_x_end, obj_y_end = np.argwhere(semantic_obs == obj_id).max(axis=0)
            obj_mask = np.zeros(semantic_obs.shape, dtype=bool)
            obj_mask[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = True
            if IoU(bbox_mask, obj_mask) > cfg.iou_threshold:
                if obj_id not in scene_objects:
                    scene_objects.append(obj_id)
                adopted_indices.append(i)
                if obj_id == target_obj_id:
                    target_found = True
                break

    if len(adopted_indices) == 0:
        return target_found, rgb
    else:
        # filter out the detections that are not adopted
        detections = detections[adopted_indices]

        annotated_image = rgb.copy()
        BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
        LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
        annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
        annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
        return target_found, annotated_image




def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # load object detection model
    detection_model = YOLOWorld(model_id=cfg.detection_model_name)

    # Load VLM
    vlm = VLM(cfg.vlm)

    # Load dataset
    all_paths_list = os.listdir(cfg.path_data_dir)

    total_questions = 0
    success_count = 0

    # Run all questions
    for question_idx in tqdm(range(len(all_paths_list))):
        total_questions += 1
        question_id = all_paths_list[question_idx]
        metadata = json.load(os.path.join(cfg.path_data_dir, question_id, "metadata.json"))

        # Extract question
        scene_id = metadata["scene"]
        question = metadata["question"]
        init_pts = metadata["init_pts"]
        init_angle = metadata["init_angle"]
        target_obj_id = metadata["target_obj_id"]
        target_obj_class = metadata["target_obj_class"]
        logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id} Target: {target_obj_class}")

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

        # Set up scene in Habitat
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
        object_id_to_name = {int(item['id']): item['class_name'] for item in bounding_box_data}

        episode_data_dir = os.path.join(str(cfg.output_dir), str(question_id))
        episode_frontier_dir = os.path.join(str(cfg.frontier_dir), str(question_id))
        os.makedirs(episode_data_dir, exist_ok=True)
        os.makedirs(episode_frontier_dir, exist_ok=True)

        pts = init_pts
        angle = init_angle

        # Floor - use pts height as floor height
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()
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
        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pos_habitat_to_normal(pts),
            init_clearance=cfg.init_clearance * 2,
        )

        # Run steps
        scene_objects = []
        target_found = False
        pts_pixs = np.empty((0, 2))  # for plotting path on the image
        for cnt_step in range(num_step):
            logging.info(f"\n== step: {cnt_step}")

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            logging.info(f"Current pts: {pts}")
            agent_state.position = pts
            agent_state.rotation = rotation
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
                    os.path.join(episode_data_dir, "{}.png".format(cnt_step)), rgb
                )
            num_black_pixels = np.sum(
                np.sum(rgb, axis=-1) == 0
            )  # sum over channel first
            if num_black_pixels < cfg.black_pixel_ratio * img_width * img_height:
                # check whether the target object is observed
                target_in_view, annotated_rgb = update_scene_graph(
                    detection_model,
                    scene_objects,
                    rgb[..., :3],
                    semantic_obs,
                    object_id_to_name,
                    cfg.scene_graph,
                    target_obj_id
                )
                annotated_rgb = Image.fromarray(annotated_rgb)
                annotated_rgb.save(
                    os.path.join(episode_data_dir, f"{cnt_step}_annotated.png")
                )

                # check stop condition
                if target_in_view:
                    if target_obj_id in scene_objects:
                        target_obj_pix_ratio = np.sum(semantic_obs == target_obj_id) / (img_height * img_width)
                        if target_obj_pix_ratio > 0:
                            obj_pix_center = np.mean(np.argwhere(semantic_obs == target_obj_id), axis=0)
                            bias_from_center = (obj_pix_center - np.asarray(
                                [img_height // 2, img_width // 2])) / np.asarray([img_height, img_width])
                            # currently just consider that the object should be in around the horizontal center, not the vertical center
                            # due to the viewing angle difference
                            if target_obj_pix_ratio > cfg.stop_min_pix_ratio and np.abs(bias_from_center)[
                                1] < cfg.stop_max_bias_from_center:
                                logging.info(f"Stop condition met at step {cnt_step}")
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
                rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")

                # Get VLM relevancy
                prompt_rel = f"\nConsider the question: '{question}'. Are you confident about answering the question with the current view?"
                # logging.info(f"Prompt Rel: {prompt_text}")
                smx_vlm_rel = vlm.get_loss(rgb_im, prompt_rel, ["Yes", "No"])
                logging.info(f"Rel - Prob: {smx_vlm_rel}")

                # Get frontier candidates
                prompt_points_pix = []
                if cfg.use_active:
                    prompt_points_pix, fig = (
                        tsdf_planner.find_prompt_points_within_view(
                            pts_normal,
                            img_width,
                            img_height,
                            cam_intr,
                            cam_pose_tsdf,
                            **cfg.visual_prompt,
                        )
                    )
                    fig.tight_layout()
                    plt.savefig(
                        os.path.join(
                            episode_data_dir, "{}_prompt_points.png".format(cnt_step)
                        )
                    )
                    plt.close()

                # Visual prompting
                draw_letters = ["A", "B", "C", "D"]  # always four
                fnt = ImageFont.truetype(
                    "data/Open_Sans/static/OpenSans-Regular.ttf",
                    30,
                )
                actual_num_prompt_points = len(prompt_points_pix)
                if actual_num_prompt_points >= cfg.visual_prompt.min_num_prompt_points:
                    rgb_im_draw = rgb_im.copy()
                    draw = ImageDraw.Draw(rgb_im_draw)
                    for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
                        draw.ellipse(
                            (
                                point_pix[0] - cfg.visual_prompt.circle_radius,
                                point_pix[1] - cfg.visual_prompt.circle_radius,
                                point_pix[0] + cfg.visual_prompt.circle_radius,
                                point_pix[1] + cfg.visual_prompt.circle_radius,
                            ),
                            fill=(200, 200, 200, 255),
                            outline=(0, 0, 0, 255),
                            width=3,
                        )
                        draw.text(
                            tuple(point_pix.astype(int).tolist()),
                            draw_letters[prompt_point_ind],
                            font=fnt,
                            fill=(0, 0, 0, 255),
                            anchor="mm",
                            font_size=12,
                        )
                    rgb_im_draw.save(
                        os.path.join(episode_data_dir, f"{cnt_step}_draw.png")
                    )

                    # get VLM reasoning for exploring
                    if cfg.use_lsv:
                        prompt_lsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Answer with a single letter."
                        # logging.info(f"Prompt Exp: {prompt_text}")
                        lsv = vlm.get_loss(
                            rgb_im_draw,
                            prompt_lsv,
                            draw_letters[:actual_num_prompt_points],
                        )
                        lsv *= actual_num_prompt_points / 3
                    else:
                        lsv = (
                            np.ones(actual_num_prompt_points) / actual_num_prompt_points
                        )

                    # base - use image without label
                    if cfg.use_gsv:
                        prompt_gsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring? Answer with Yes or No."
                        # logging.info(f"Prompt Exp base: {prompt_gsv}")
                        gsv = vlm.get_loss(rgb_im, prompt_gsv, ["Yes", "No"])[0]
                        gsv = (
                            np.exp(gsv / cfg.gsv_T) / cfg.gsv_F
                        )  # scale before combined with lsv
                    else:
                        gsv = 1
                    sv = lsv * gsv
                    logging.info(f"Exp - LSV: {lsv} GSV: {gsv} SV: {sv}")

                    # Integrate semantics only if there is any prompted point
                    tsdf_planner.integrate_sem(
                        sem_pix=sv,
                        radius=1.0,
                        obs_weight=1.0,
                    )  # voxel locations already saved in tsdf class

                # Save data
                # result[step_name]["smx_vlm_pred"] = smx_vlm_pred
                # result[step_name]["smx_vlm_rel"] = smx_vlm_rel
            else:
                logging.info("Skipping black image!")
                # result[step_name]["smx_vlm_pred"] = np.ones((4)) / 4
                # result[step_name]["smx_vlm_rel"] = np.array([0.01, 0.99])

            if target_found:
                break

            # Determine next point
            if cnt_step < num_step:
                pts_normal, angle, pts_pix, fig = tsdf_planner.find_next_pose(
                    pts=pts_normal,
                    angle=angle,
                    flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                    **cfg.planner,
                )
                pts_pixs = np.vstack((pts_pixs, pts_pix))
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)

                # Add path to ax5, with colormap to indicate order
                ax5 = fig.axes[4]
                ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
                fig.tight_layout()
                plt.savefig(
                    os.path.join(episode_data_dir, "{}_map.png".format(cnt_step + 1))
                )
                plt.close()
            rotation = quat_to_coeffs(
                quat_from_angle_axis(angle, np.array([0, 1, 0]))
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()

        if target_found:
            success_count += 1

        logging.info(f"Success rate: {success_count}/{total_questions} = {success_count / total_questions}")



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
    cfg.frontier_dir = os.path.join(cfg.output_dir, "frontier")
    if not os.path.exists(cfg.frontier_dir):
        os.makedirs(cfg.frontier_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(cfg.output_dir, "log.log")
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
