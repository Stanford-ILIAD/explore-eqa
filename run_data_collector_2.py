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
    get_navigable_point_to_new
)
from src.geom import get_cam_intr, get_scene_bnds, get_collision_distance, points_in_circle
from src.tsdf_2 import TSDFPlanner, Frontier, Object
from inference.models import YOLOWorld

'''
This script extend the run_data_collector.py to allow generating paths from
partially explored scenes. The script will first run a random question in the scene, then starts to
record the path of another question.
'''


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
    with open(os.path.join(cfg.question_data_path, "generated_questions.json")) as f:
        questions_data = json.load(f)
    all_scene_list = list(set([q["episode_history"] for q in questions_data]))
    all_scene_list.sort(key=lambda x: int(x.split("-")[0]), reverse=True)
    logging.info(f"Loaded {len(questions_data)} questions.")

    # for each scene, answer each question
    question_ind = 0
    success_count = 0
    total_questions = len(questions_data)
    for scene_id in all_scene_list:
        all_questions_in_scene = [q for q in questions_data if q["episode_history"] == scene_id]

        ##########################################################
        # rand_q = np.random.randint(0, len(all_questions_in_scene) - 1)
        # all_questions_in_scene = all_questions_in_scene[rand_q:rand_q+1]
        # all_questions_in_scene = [q for q in all_questions_in_scene if '00009-vLpv2VX547B_68_printer' in q['question_id']]
        # if len(all_questions_in_scene) == 0:
        #     continue
        # random.shuffle(all_questions_in_scene)
        # all_questions_in_scene = all_questions_in_scene
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

        path_idx = cfg.path_id_offset

        for question_data in all_questions_in_scene:
            question_ind += 1
            floor_height_curr = question_data['position'][1]

            # randomly select another question in the scene to initialize
            try_count = 0
            while True:
                # the question data should not be the same as the current one
                # and should be on the same floor
                try_count += 1
                prev_question_data = random.choice(all_questions_in_scene)
                if (prev_question_data['question_id'] != question_data['question_id'] and
                    np.abs(prev_question_data['position'][1] - floor_height_curr) < 0.1):
                    break
                if try_count > 1000:
                    break

            target_obj_id = prev_question_data['object_id']
            target_position = prev_question_data['position']
            target_rotation = prev_question_data['rotation']
            episode_data_dir = os.path.join(str(cfg.dataset_output_dir), f"{question_data['question_id']}_path_{path_idx}")
            episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
            egocentric_save_dir = os.path.join(episode_data_dir, 'egocentric')

            # if the data has already generated, skip
            if os.path.exists(episode_data_dir) and os.path.exists(os.path.join(episode_data_dir, "metadata.json")):
                logging.info(f"Question id {question_data['question_id']}-path {path_idx} already exists")
                success_count += 1
                continue

            if os.path.exists(episode_data_dir) and not os.path.exists(os.path.join(episode_data_dir, "metadata.json")):
                os.system(f"rm -r {episode_data_dir}")

            os.makedirs(episode_data_dir, exist_ok=True)
            os.makedirs(episode_frontier_dir, exist_ok=True)
            os.makedirs(egocentric_save_dir, exist_ok=True)

            # get a navigation start point
            floor_height = target_position[1]
            tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
            scene_length = (tsdf_bnds[:2, 1] - tsdf_bnds[:1, 0]).mean()
            min_dist = max(cfg.min_travel_dist, scene_length * cfg.min_travel_dist_ratio)
            pathfinder.seed(random.randint(0, 1000000))
            start_position, path_points, travel_dist = get_navigable_point_to_new(
                target_position, pathfinder, max_search=1000, min_dist=min_dist
            )
            if start_position is None or path_points is None:
                logging.info(f"Cannot find a navigable path to the target object in question {question_data['question_id']}-path {path_idx}!")
                continue

            # set the initial orientation of the agent as the initial path direction
            init_orientation = path_points[1] - path_points[0]
            init_orientation[1] = 0
            angle, axis = quat_to_angle_axis(
                quat_from_two_vectors(np.asarray([0, 0, -1]), init_orientation)
            )
            angle = angle * axis[1] / np.abs(axis[1])
            pts = start_position.copy()
            rotation = get_quaternion(angle, camera_tilt)

            # initialize the TSDF
            pts_normal = pos_habitat_to_normal(pts)
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
                occupancy_height=cfg.occupancy_height,
                vision_height=cfg.vision_height
            )

            # convert path points to normal and drop y-axis for tsdf planner
            path_points = [pos_habitat_to_normal(p) for p in path_points]
            path_points = [p[:2] for p in path_points]

            # record the history of the agent's path
            pts_pixs = np.empty((0, 2))
            pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(start_position)[:2]))

            logging.info(f'\n\nQuestion id {question_data["question_id"]}-path {path_idx} initialization successful!')

            # run steps
            target_found = False
            max_explore_dist = travel_dist * cfg.max_step_dist_ratio_prev
            explore_dist = 0.0
            cnt_step = -1
            while explore_dist < max_explore_dist and cnt_step < 40:  # 40 is just a not normal step number to avoid agent stuck in a place
                cnt_step += 1

                logging.info(f"\n== step: {cnt_step}")

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
                    pts_normal, height=cfg.vision_height
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

                    if cfg.save_obs_prev:
                        observation_save_dir = os.path.join(episode_data_dir, 'observations')
                        os.makedirs(observation_save_dir, exist_ok=True)
                        if target_found:
                            plt.imsave(os.path.join(observation_save_dir, f"prev_{cnt_step}-view_{view_idx}-target.png"), annotated_rgb)
                        else:
                            plt.imsave(os.path.join(observation_save_dir, f"prev_{cnt_step}-view_{view_idx}.png"), annotated_rgb)

                    if target_found:
                        break

                if cfg.save_visualization_prev:
                    tsdf_values = tsdf_planner._tsdf_vol_cpu[:, :, 4]  # (h, w), float
                    # save the value as image
                    visualization_path = os.path.join(episode_data_dir, "visualization")
                    os.makedirs(visualization_path, exist_ok=True)
                    im = plt.imshow(tsdf_values)
                    plt.colorbar(im, orientation='vertical')
                    plt.savefig(
                        os.path.join(visualization_path, f"prev_{cnt_step}_tsdf_value.png")
                    )
                    plt.close()


                update_success = tsdf_planner.update_frontier_map(pts=pts_normal, cfg=cfg.planner)
                if not update_success:
                    logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: update frontier map failed!")
                    break

                max_point_choice = tsdf_planner.get_next_choice(
                    pts=pts_normal,
                    angle=angle,
                    path_points=path_points,
                    pathfinder=pathfinder,
                    target_obj_id=target_obj_id,
                    cfg=cfg.planner,
                )
                if max_point_choice is None:
                    logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: no valid choice!")
                    break

                return_values = tsdf_planner.get_next_navigation_point(
                    choice=max_point_choice,
                    pts=pts_normal,
                    angle=angle,
                    path_points=path_points,
                    pathfinder=pathfinder,
                    cfg=cfg.planner,
                    save_visualization=cfg.save_visualization_prev,
                )
                if return_values[0] is None:
                    logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: find next navigation point failed!")
                    break
                pts_normal, angle, pts_pix, fig, path_points = return_values

                # Turn to face each frontier point and get rgb image
                # print(f"Start to save {len(tsdf_planner.frontiers)} frontier observations")
                for i, frontier in enumerate(tsdf_planner.frontiers):
                    pos_voxel = frontier.position
                    pos_world = pos_voxel * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
                    pos_world = pos_normal_to_habitat(np.append(pos_world, floor_height))
                    # Turn to face the frontier point
                    if frontier.image is not None:
                        assert os.path.exists(os.path.join(episode_frontier_dir, frontier.image))
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
                            os.path.join(episode_frontier_dir, f"prev_{cnt_step}_{i}.png"),
                            rgb,
                        )
                        frontier.image = f"prev_{cnt_step}_{i}.png"

                # update the agent's position record
                pts_pixs = np.vstack((pts_pixs, pts_pix))
                if cfg.save_visualization_prev:
                    # Add path to ax5, with colormap to indicate order
                    visualization_path = os.path.join(episode_data_dir, "visualization")
                    os.makedirs(visualization_path, exist_ok=True)
                    ax5 = fig.axes[4]
                    ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                    ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
                    fig.tight_layout()
                    plt.savefig(os.path.join(visualization_path, "prev_{}_map.png".format(cnt_step)))
                    plt.close()

                if cfg.save_frontier_video and len(tsdf_planner.frontiers) != 0:
                    frontier_video_path = os.path.join(episode_data_dir, "frontier_video")
                    os.makedirs(frontier_video_path, exist_ok=True)
                    num_images = len(tsdf_planner.frontiers)
                    side_length = int(np.sqrt(num_images)) + 1
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
                    global_caption = f"{prev_question_data['question']}\n{prev_question_data['answer']}"
                    if type(max_point_choice) == Object:
                        global_caption += '\nToward target object'
                    fig.suptitle(global_caption, fontsize=16)
                    plt.tight_layout(rect=(0., 0., 1., 0.95))
                    plt.savefig(os.path.join(frontier_video_path, f'prev_{cnt_step}.png'))
                    plt.close()

                # update position and rotation
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)
                rotation = get_quaternion(angle, camera_tilt)
                explore_dist += np.linalg.norm(pts_pixs[-1] - pts_pixs[-2]) * tsdf_planner._voxel_size

                if len(pts_pixs) >= 3 and np.linalg.norm(pts_pixs[-1] - pts_pixs[-2]) < 1e-3 and np.linalg.norm(pts_pixs[-2] - pts_pixs[-3]) < 1e-3:
                    logging.info(f"Agent is stuck at step {cnt_step}")
                    break

                logging.info(f"Current position: {pts}, {explore_dist:.3f}/{max_explore_dist:.3f}")

                if target_found:
                    break

            logging.info(f"The previous exploration of question {question_data['question_id']} has finished!")

            ##########################################
            ##########################################
            ##########################################
            ##########################################

            # start to record the path of the current question
            target_obj_id = question_data['object_id']
            target_position = question_data['position']
            target_rotation = question_data['rotation']

            # get the starting points of other generated paths for this object, if there exists any
            # get all the folder in the form os.path.join(str(cfg.dataset_output_dir), f"{question_data['question_id']}_path_*")
            glob_path = os.path.join(str(cfg.dataset_output_dir), f"{question_data['question_id']}_path_*")
            all_generated_path_dirs = glob.glob(glob_path)
            prev_start_positions = []
            for path_dir in all_generated_path_dirs:
                metadata_path = os.path.join(path_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        prev_start_positions.append(json.load(f)["init_pts"])
            if len(prev_start_positions) == 0:
                prev_start_positions = None
            else:
                prev_start_positions = np.asarray(prev_start_positions)

            # get a navigation start point
            pathfinder.seed(random.randint(0, 1000000))
            start_position, path_points, travel_dist = get_navigable_point_to_new(
                target_position, pathfinder, max_search=1000, min_dist=min_dist,
                prev_start_positions=prev_start_positions
            )
            if start_position is None or path_points is None:
                logging.info(f"Cannot find a navigable path to the target object in question {question_data['question_id']}-path {path_idx}!")
                continue

            # set the initial orientation of the agent as the initial path direction
            init_orientation = path_points[1] - path_points[0]
            init_orientation[1] = 0
            angle, axis = quat_to_angle_axis(
                quat_from_two_vectors(np.asarray([0, 0, -1]), init_orientation)
            )
            angle = angle * axis[1] / np.abs(axis[1])
            pts = start_position.copy()
            rotation = get_quaternion(angle, camera_tilt)

            # add another clear area in the new init position
            start_pos_voxel = tsdf_planner.habitat2voxel(start_position)
            tsdf_planner.init_points += points_in_circle(
                start_pos_voxel[0],
                start_pos_voxel[1],
                int(cfg.init_clearance * 2 / tsdf_planner._voxel_size),
                tsdf_planner._vol_dim[:2]
            )
            # clear up the "stuck" mark in existing frontiers
            for frontier in tsdf_planner.frontiers:
                frontier.is_stuck = False

            # convert path points to normal and drop y-axis for tsdf planner
            path_points = [pos_habitat_to_normal(p) for p in path_points]
            path_points = [p[:2] for p in path_points]

            # record the history of the agent's path
            pts_pixs = np.empty((0, 2))
            pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(start_position)[:2]))

            logging.info(f'\n\nQuestion id {question_data["question_id"]}-path {path_idx} initialization successful!')

            metadata = {}
            metadata["question"] = question_data["question"]
            metadata["answer"] = question_data["answer"]
            metadata["scene"] = question_data["episode_history"]
            metadata["init_pts"] = pts.tolist()
            metadata["init_angle"] = angle
            metadata["target_obj_id"] = target_obj_id
            metadata["target_obj_class"] = question_data["class"]

            # run steps
            target_found = False
            previous_choice_path = None
            max_explore_dist = travel_dist * cfg.max_step_dist_ratio
            max_step = int(travel_dist * cfg.max_step_ratio)
            zero_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
            explore_dist = 0.0
            cnt_step = -1
            while explore_dist < max_explore_dist and cnt_step < max_step:
                cnt_step += 1

                step_dict = {}
                step_dict["agent_state"] = {}
                step_dict["agent_state"]["init_pts"] = pts.tolist()
                step_dict["agent_state"]["init_angle"] = rotation

                logging.info(f"\n== step: {cnt_step}")

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
                    pts_normal, height=cfg.vision_height
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
                            if cfg.save_egocentric_view:
                                plt.imsave(os.path.join(egocentric_save_dir, f"{cnt_step}_view_{view_idx}.png"), zero_image)
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
                    if not keep_observation:
                        if not (view_idx == total_views - 1 and keep_forward_observation):
                            # logging.info(f"Invalid observation: black pixel ratio {black_pix_ratio}, 30 percentile depth {np.percentile(depth[depth > 0], 30)}")
                            if cfg.save_egocentric_view:
                                plt.imsave(os.path.join(egocentric_save_dir, f"{cnt_step}_view_{view_idx}.png"), zero_image)
                            continue

                    # construct an frequency count map of each semantic id to a unique id
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

                    if cfg.save_egocentric_view:
                        plt.imsave(os.path.join(egocentric_save_dir, f"{cnt_step}_view_{view_idx}.png"), rgb)

                    observation_kept_count += 1

                    if target_found:
                        break

                if target_found:
                    break

                if cfg.save_visualization:
                    tsdf_values = tsdf_planner._tsdf_vol_cpu[:, :, 4]  # (h, w), float
                    # save the value as image
                    im = plt.imshow(tsdf_values)
                    plt.colorbar(im, orientation='vertical')
                    plt.savefig(
                        os.path.join(visualization_path, f"{cnt_step}_tsdf_value.png")
                    )
                    plt.close()

                # record current scene graph
                step_dict["scene_graph"] = list(tsdf_planner.simple_scene_graph.keys())
                step_dict["scene_graph"] = [int(x) for x in step_dict["scene_graph"]]

                step_dict["frontiers"] = []

                update_success = tsdf_planner.update_frontier_map(pts=pts_normal, cfg=cfg.planner)
                if not update_success:
                    logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: update frontier map failed!")
                    break

                max_point_choice = tsdf_planner.get_next_choice(
                    pts=pts_normal,
                    angle=angle,
                    path_points=path_points,
                    pathfinder=pathfinder,
                    target_obj_id=target_obj_id,
                    cfg=cfg.planner,
                )
                if max_point_choice is None:
                    logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: no valid choice!")
                    break

                return_values = tsdf_planner.get_next_navigation_point(
                    choice=max_point_choice,
                    pts=pts_normal,
                    angle=angle,
                    path_points=path_points,
                    pathfinder=pathfinder,
                    cfg=cfg.planner,
                    save_visualization=cfg.save_visualization,
                )
                if return_values[0] is None:
                    logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: find next navigation point failed!")
                    break
                pts_normal, angle, pts_pix, fig, path_points = return_values

                # Turn to face each frontier point and get rgb image
                # print(f"Start to save {len(tsdf_planner.frontiers)} frontier observations")
                for i, frontier in enumerate(tsdf_planner.frontiers):
                    frontier_dict = {}
                    pos_voxel = frontier.position
                    pos_world = pos_voxel * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
                    pos_world = pos_normal_to_habitat(np.append(pos_world, floor_height))
                    frontier_dict["coordinate"] = pos_world.tolist()
                    # Turn to face the frontier point
                    if frontier.image is not None:
                        if 'prev_' in frontier.image:
                            new_name = f'{cnt_step}_{i}.png'
                            os.system(f"cp {os.path.join(episode_frontier_dir, frontier.image)} {os.path.join(episode_frontier_dir, new_name)}")
                            frontier.image = new_name
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
                        frontier.image = f"{cnt_step}_{i}.png"
                        frontier_dict["rgb_id"] = f"{cnt_step}_{i}.png"
                    step_dict["frontiers"].append(frontier_dict)

                # save the ground truth choice
                if type(max_point_choice) == Object:
                    choice_obj_id = max_point_choice.object_id
                    prediction = [float(scene_graph_obj_id == choice_obj_id) for scene_graph_obj_id in step_dict["scene_graph"]]
                    prediction += [0.0 for _ in range(len(step_dict["frontiers"]))]
                elif type(max_point_choice) == Frontier:
                    prediction = [0.0 for _ in range(len(step_dict["scene_graph"]))]
                    prediction += [float(ft == max_point_choice) for ft in tsdf_planner.frontiers]
                else:
                    raise ValueError("Invalid max_point_choice type")
                assert len(prediction) == len(step_dict["scene_graph"]) + len(step_dict["frontiers"]), f"{len(prediction)} != {len(step_dict['scene_graph'])} + {len(step_dict['frontiers'])}"
                if sum(prediction) != 1.0:
                    logging.info(f"Error! Prediction sum is not 1.0: {sum(prediction)}")
                    logging.info(max_point_choice)
                    logging.info(tsdf_planner.frontiers)
                    break
                assert sum(prediction) == 1.0, f"{sum(prediction)} != 1.0"
                step_dict["prediction"] = prediction

                # record previous choice
                step_dict["previous_choice"] = previous_choice_path  # this could be None or an image path of the frontier in last step
                if type(max_point_choice) == Frontier:
                    previous_choice_path = max_point_choice.image

                # Save step data
                with open(os.path.join(episode_data_dir, f"{cnt_step:04d}.json"), "w") as f:
                    json.dump(step_dict, f, indent=4)

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

                if cfg.save_frontier_video:
                    frontier_video_path = os.path.join(episode_data_dir, "frontier_video")
                    os.makedirs(frontier_video_path, exist_ok=True)
                    # if type(max_point_choice) == Frontier:
                    #     # img_path = os.path.join(episode_frontier_dir, max_point_choice.image)
                    #     # os.system(f"cp {img_path} {os.path.join(frontier_video_path, f'{cnt_step:04d}-frontier.png')}")
                    #     num_frontiers = len(tsdf_planner.frontiers)
                    #     side_length = int(np.sqrt(num_frontiers)) + 1
                    #     plt.figure()
                    #     for i in range(1, num_frontiers + 1):
                    #         plt.subplot(side_length, side_length, i)
                    #         img_path = os.path.join(episode_frontier_dir, tsdf_planner.frontiers[i].image)
                    #         img = plt.imread(img_path)
                    #         plt.imshow(img)
                    #         plt.xticks([])
                    #         plt.yticks([])
                    #         if max_point_choice.image == tsdf_planner.frontiers[i].image:
                    #             plt.title('chosen')
                    #     plt.imsave(frontier_video_path, f'{cnt_step}.png')
                    # else:  # navigating to the objects
                    #     if cfg.save_obs:
                    #         img_path = os.path.join(observation_save_dir, f"{cnt_step}-view_{total_views - 1}.png")
                    #         if os.path.exists(img_path):
                    #             os.system(f"cp {img_path} {os.path.join(frontier_video_path, f'{cnt_step:04d}-object.png')}")

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
                    global_caption = f"{question_data['question']}\n{question_data['answer']}"
                    if type(max_point_choice) == Object:
                        global_caption += '\nToward target object'
                    fig.suptitle(global_caption, fontsize=16)
                    plt.tight_layout(rect=(0., 0., 1., 0.95))
                    plt.savefig(os.path.join(frontier_video_path, f'{cnt_step}.png'))
                    plt.close()

                # update position and rotation
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)
                rotation = get_quaternion(angle, camera_tilt)
                explore_dist += np.linalg.norm(pts_pixs[-1] - pts_pixs[-2]) * tsdf_planner._voxel_size

                logging.info(f"Current position: {pts}, {explore_dist:.3f}/{max_explore_dist:.3f}")

            if target_found:
                metadata["episode_length"] = cnt_step
                with open(os.path.join(episode_data_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=4)
                logging.info(f"Question id {question_data['question_id']}-path {path_idx} finish with {cnt_step} steps")
                success_count += 1
            else:
                logging.info(f"Question id {question_data['question_id']}-path {path_idx} failed.")
                if cfg.del_fail_case:
                    os.system(f"rm -r {episode_data_dir}")

            logging.info(f"{question_ind}/{total_questions}: Success rate: {success_count}/{question_ind}")

        logging.info(f'Scene {scene_id} finish')

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
    parser.add_argument("--path_id_offset", default=0, type=int)
    parser.add_argument("--seed", default=None, type=int)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    if not os.path.exists(cfg.dataset_output_dir):
        os.makedirs(cfg.dataset_output_dir, exist_ok=True)
    logging_path = os.path.join(cfg.output_dir, "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    cfg.path_id_offset = args.path_id_offset
    if args.seed is not None:
        cfg.seed = args.seed

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg)
