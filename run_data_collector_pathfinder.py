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
from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM
from src.tsdf import TSDFPlanner, Frontier, Object
from habitat_sim.utils.common import d3_40_colors_rgb


def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    with open(os.path.join(cfg.question_data_path, "generated_questions.json")) as f:
        questions_data = json.load(f)
    all_scene_list = list(set([q["episode_history"] for q in questions_data]))
    logging.info(f"Loaded {len(questions_data)} questions.")

    # for each scene, answer each question
    question_ind = 0
    for scene_id in all_scene_list:
        all_questions_in_scene = [q for q in questions_data if q["episode_history"] == scene_id]

        ##########################################################
        # rand_q = np.random.randint(0, len(all_questions_in_scene) - 1)
        # all_questions_in_scene = all_questions_in_scene[rand_q:rand_q+1]
        # all_questions_in_scene = [q for q in all_questions_in_scene if q['question_id'] == '00324-DoSbsoo4EAg_240_cutting_board_878397']
        # all_questions_in_scene = all_questions_in_scene[6:]
        # all_questions_in_scene = [q for q in all_questions_in_scene if "00109" in q['question_id']]
        ##########################################################

        # load scene
        split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
        scene_mesh_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.glb")
        navmesh_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.navmesh")
        semantic_texture_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".semantic.glb")
        scene_semantic_annotation_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".semantic.txt")
        assert os.path.exists(scene_mesh_path) and os.path.exists(navmesh_path) and os.path.exists(semantic_texture_path) and os.path.exists(scene_semantic_annotation_path)

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
        #
        # # Load the semantic annotation
        # obj_id_to_class = {}
        # obj_id_to_room_id = {}
        # with open(scene_semantic_annotation_path, "r") as f:
        #     for line in f.readlines():
        #         if 'HM3D Semantic Annotations' in line:  # skip the first line
        #             continue
        #         line = line.strip().split(',')
        #         idx = int(line[0])
        #         class_name = line[2].replace("\"", "")
        #         room_id = int(line[3])
        #         obj_id_to_class[idx] = class_name
        #         obj_id_to_room_id[idx] = room_id
        # obj_id_to_class[0] = 'unannotated'

        for question_data in all_questions_in_scene:
            question_ind += 1

            target_obj_id = question_data['object_id']
            target_position = question_data['position']
            target_rotation = question_data['rotation']
            episode_data_dir = os.path.join(str(cfg.dataset_output_dir), f"{question_data['question_id']}")
            episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")

            # if the data has already generated, skip
            if os.path.exists(episode_data_dir) and os.path.exists(os.path.join(episode_data_dir, "metadata.json")):
                logging.info(f"Question id {question_data['question_id']} already exists")
                continue

            os.makedirs(episode_data_dir, exist_ok=True)
            os.makedirs(episode_frontier_dir, exist_ok=True)

            # get a navigation start point
            start_position, path_points, travel_dist = get_navigable_point_to(
                target_position, pathfinder, max_search=1000, min_dist=cfg.min_travel_dist
            )
            if start_position is None or path_points is None:
                logging.info(f"Cannot find a navigable path to the target object in question {question_data['question_id']}")
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
            floor_height = target_position[1]
            tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
            try:
                del tsdf_planner
            except:
                pass
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height_offset=0,
                pts_init=pos_habitat_to_normal(start_position),
                init_clearance=cfg.init_clearance * 2,
            )

            # convert path points to normal and drop y-axis for tsdf planner
            path_points = [pos_habitat_to_normal(p) for p in path_points]
            path_points = [p[:2] for p in path_points]

            # record the history of the agent's path
            pts_pixs = np.empty((0, 2))
            pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(start_position)[:2]))

            logging.info(f'Question id {question_data["question_id"]} finish initialization')

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
            num_step = int(travel_dist * cfg.max_step_dist_ratio)
            for cnt_step in range(num_step):

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

                # observe and update the TSDF
                for view_idx, ang in enumerate(all_angles):
                    logging.info(f"Step {cnt_step}, view {view_idx + 1}/{total_views}")

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

                    # check stop condition
                    target_obj_pix_ratio = np.sum(semantic_obs == target_obj_id) / (img_height * img_width)
                    if target_obj_pix_ratio > 0:
                        obj_pix_center = np.mean(np.argwhere(semantic_obs == target_obj_id), axis=0)
                        bias_from_center = (obj_pix_center - np.asarray([img_height // 2, img_width // 2])) / np.asarray([img_height, img_width])
                        # currently just consider that the object should be in around the horizontal center, not the vertical center
                        # due to the viewing angle difference
                        if target_obj_pix_ratio > cfg.stop_min_pix_ratio and np.abs(bias_from_center)[1] < cfg.stop_max_bias_from_center:
                            logging.info(f"Stop condition met at step {cnt_step} view {view_idx}")
                            target_found = True

                    # construct an frequency count map of each semantic id to a unique id
                    masked_ids = np.unique(semantic_obs[depth > 5.0])
                    semantic_obs = np.where(np.isin(semantic_obs, masked_ids), 0, semantic_obs)
                    tsdf_planner.increment_scene_graph(semantic_obs, object_id_to_bbox, min_pix_ratio=cfg.min_pix_ratio)

                    if cfg.save_obs:
                        observation_save_dir = os.path.join(episode_data_dir, 'observations')
                        os.makedirs(observation_save_dir, exist_ok=True)
                        if target_found:
                            plt.imsave(os.path.join(observation_save_dir, f"{cnt_step}-view_{view_idx}-target.png"), rgb)
                        else:
                            plt.imsave(os.path.join(observation_save_dir, f"{cnt_step}-view_{view_idx}.png"), rgb)
                        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
                        semantic_img.putpalette(d3_40_colors_rgb.flatten())
                        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
                        semantic_img = semantic_img.convert("RGBA")
                        if target_found:
                            semantic_img.save(os.path.join(observation_save_dir, f"{cnt_step}-view_{view_idx}-semantic-target.png"))
                        else:
                            semantic_img.save(os.path.join(observation_save_dir, f"{cnt_step}-view_{view_idx}--semantic.png"))

                    if target_found:
                        break

                    num_black_pixels = np.sum(np.sum(rgb, axis=-1) == 0)  # sum over channel first
                    if num_black_pixels < cfg.black_pixel_ratio * img_width * img_height:
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

                if target_found:
                    break

                # record current scene graph
                step_dict["scene_graph"] = list(tsdf_planner.simple_scene_graph.keys())
                step_dict["scene_graph"] = [int(x) for x in step_dict["scene_graph"]]

                step_dict["frontiers"] = []

                # determine the next point and move the agent
                return_values = tsdf_planner.find_next_pose_with_path(
                    pts=pts_normal,
                    angle=angle,
                    path_points=path_points,
                    pathfinder=pathfinder,
                    target_obj_id=target_obj_id,
                    flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                    save_visualization=cfg.save_visualization,
                    **cfg.planner,
                )
                if return_values[0] is None:
                    logging.info(f"Question id {question_data['question_id']} invalid!")
                    break

                pts_normal, angle, pts_pix, fig, path_points, max_point_choice = return_values

                # Turn to face each frontier point and get rgb image
                print(f"Start to save {len(tsdf_planner.frontiers)} frontier observations")
                for i, frontier in enumerate(tsdf_planner.frontiers):
                    frontier_dict = {}
                    pos_voxel = frontier.position
                    pos_world = pos_voxel * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
                    pos_world = pos_normal_to_habitat(np.append(pos_world, floor_height))
                    frontier_dict["coordinate"] = pos_world.tolist()
                    # Turn to face the frontier point
                    if frontier.image is not None:
                        frontier_dict["rgb_id"] = frontier.image
                    else:
                        view_frontier_direction = np.asarray([pos_world[0] - pts[0], 0., pos_world[2] - pts[2]])
                        default_view_direction = np.asarray([0., 0., -1.])
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
                assert len(prediction) == len(step_dict["scene_graph"]) + len(step_dict["frontiers"])
                assert sum(prediction) == 1.0
                step_dict["prediction"] = prediction

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

                # update position and rotation
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)
                rotation = get_quaternion(angle, camera_tilt)

            if target_found:
                metadata["episode_length"] = cnt_step
                with open(os.path.join(episode_data_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=4)
                logging.info(f"Question id {question_data['question_id']} finish with {cnt_step} steps")
            else:
                logging.info(f"Question id {question_data['question_id']} failed.")
                os.system(f"rm -r {episode_data_dir}")

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

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg)
