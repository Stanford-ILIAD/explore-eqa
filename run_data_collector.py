"""
Run EQA in Habitat-Sim with VLM exploration.

"""

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
)
from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM
from src.tsdf import TSDFPlanner
from habitat_sim.utils.common import d3_40_colors_rgb


def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    cfg.seed = np.random.randint(1000000)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    with open(cfg.init_pose_data_path) as f:
        init_pose_data = {}
        for row in csv.DictReader(f, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }
    logging.info(f"Loaded {len(questions_data)} questions.")

    for question_ind in tqdm(range(len(questions_data))):
        # Extract question
        question_data = questions_data[question_ind]
        scene = question_data["scene"]
        floor = question_data["floor"]
        scene_floor = scene + "_" + floor
        question = question_data["question"]
        choices = question_data["choices"]
        answer = question_data["answer"]
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]
        logging.info(f"\n========\nIndex: {question_ind} Scene: {scene} Floor: {floor}")
        logging.info(f"Question: {question} Choices: {choices}, Answer: {answer}")

        ######
        # load semantic object bbox data
        with open(os.path.join(cfg.semantic_bbox_data_path, f"{scene}.json")) as f:
            semantic_data = json.load(f)
        ######

        # Re-format the question to follow LLaMA style
        # vlm_question = question
        # vlm_pred_candidates = ["A", "B", "C", "D"]
        # for token, choice in zip(vlm_pred_candidates, choices):
        #     vlm_question += "\n" + token + "." + " " + choice

        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(cfg.dataset_output_dir, f"{question_ind:07d}")
        episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
        os.makedirs(episode_data_dir, exist_ok=True)
        os.makedirs(episode_frontier_dir, exist_ok=True)

        metadata = {}
        metadata["question"] = question
        metadata["scene"] = scene
        metadata["floor"] = floor
        metadata["init_pts"] = init_pts
        metadata["init_angle"] = init_angle

        result = {"question_ind": question_ind}

        # Set up scene in Habitat
        try:
            simulator.close()
        except:
            pass
        scene_mesh_dir = os.path.join(
            cfg.scene_data_path, scene, scene[6:] + ".basis" + ".glb"
        )
        navmesh_file = os.path.join(
            cfg.scene_data_path, scene, scene[6:] + ".basis" + ".navmesh"
        )
        sim_settings = {
            "scene": scene_mesh_dir,
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
        pathfinder.load_nav_mesh(navmesh_file)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
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

        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pos_habitat_to_normal(pts),
            init_clearance=cfg.init_clearance * 2,
        )

        # find an endpoint for the path
        pts_end = None
        path_points = None
        max_try = 1000
        try_count = 0
        max_distance_history = -1
        while True:
            try_count += 1
            if try_count > max_try:
                break

            pts_end_current = simulator.pathfinder.get_random_navigable_point()
            if np.abs(pts_end_current[1] - pts[1]) > 0.4:  # make sure the end point is on the same level
                continue

            path = habitat_sim.ShortestPath()
            path.requested_start = pts
            path.requested_end = pts_end_current
            found_path = simulator.pathfinder.find_path(path)
            # geodesic_distance = path.geodesic_distance
            # path_points = path.points  # list of points in the path
            if found_path:
                if path.geodesic_distance > max_distance_history:
                    max_distance_history = path.geodesic_distance
                    pts_end = pts_end_current
                    path_points = path.points

            if found_path and max_distance_history > 6:
                break

        assert pts_end is not None and path_points is not None
        assert np.array_equal(path_points[0], np.asarray(pts, dtype=np.float32)) and np.array_equal(path_points[-1], pts_end)
        init_orientation = path_points[1] - path_points[0]
        init_orientation[1] = 0
        # set the agent's orientation
        rotation = quat_to_coeffs(
            quat_from_two_vectors(np.array([0, 0, -1]), init_orientation)
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()
        angle, axis = quat_to_angle_axis(
            quat_from_two_vectors(np.array([0, 0, -1]), init_orientation)
        )
        # convert path points to normal
        path_points = [pos_habitat_to_normal(p) for p in path_points]
        # drop y coordinate
        path_points = [p[:2] for p in path_points]

        pts_pixs = np.empty((0, 2))  # for plotting path on the image
        # get the voxel coordinate of the init position
        pts_voxel = pos_habitat_to_normal(pts)
        pts_voxel = (pts_voxel[:2] - tsdf_planner._vol_origin[:2]) / tsdf_planner._voxel_size
        pts_pixs = np.vstack((pts_pixs, pts_voxel))
        pts_pixs = np.empty((0, 2))  # for plotting path on the image
        for cnt_step in range(num_step):

            step_dict = {}
            step_dict["agent_state"] = {}
            step_dict["agent_state"]["init_pts"] = list(pts)
            step_dict["agent_state"]["init_angle"] = list(rotation)

            logging.info(f"\n== step: {cnt_step}")

            # Save step info and set current pose
            step_name = f"step_{cnt_step}"
            logging.info(f"Current pts: {pts}")
            agent_state.position = pts
            agent_state.rotation = rotation
            agent.set_state(agent_state)
            pts_normal = pos_habitat_to_normal(pts)
            result[step_name] = {"pts": pts, "angle": angle}

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

            # construct an frequency count map of each semantic id to a unique id
            masked_ids = np.unique(semantic_obs[depth > 3.0])
            semantic_obs = np.where(np.isin(semantic_obs, masked_ids), 0, semantic_obs)
            tsdf_planner.increment_scene_graph(semantic_obs, semantic_data)

            step_dict["scene_graph"] = list(tsdf_planner.simple_scene_graph.keys())
            step_dict["scene_graph"] = [int(x) for x in step_dict["scene_graph"]]
            if cfg.save_obs:
                plt.imsave(
                    os.path.join(episode_data_dir, "{}.png".format(cnt_step)), rgb
                )
                semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
                semantic_img.putpalette(d3_40_colors_rgb.flatten())
                semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
                semantic_img = semantic_img.convert("RGBA")
                semantic_img.save(
                    os.path.join(episode_data_dir, "{}_semantic.png".format(cnt_step))
                )

            num_black_pixels = np.sum(
                np.sum(rgb, axis=-1) == 0
            )  # sum over channel first
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

                result[step_name]["smx_vlm_pred"] = np.ones((4)) / 4
                result[step_name]["smx_vlm_rel"] = np.array([0.01, 0.99])
            else:
                result[step_name]["smx_vlm_pred"] = np.ones((4)) / 4
                result[step_name]["smx_vlm_rel"] = np.array([0.01, 0.99])

            step_dict["frontiers"] = []
            # Determine next point
            if cnt_step < num_step:
                pts_normal, angle, pts_pix, fig, path_points = tsdf_planner.find_next_pose_with_path(
                    pts=pts_normal,
                    angle=angle,
                    path_points=path_points,
                    pathfinder=pathfinder,
                    flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                    **cfg.planner,
                )

                # Turn to face each frontier point and get rgb image
                print(f"Num Frontiers: {len(tsdf_planner.frontiers)}")
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
                    ### We still need to save ground truth for every step here! ###

                # Save step data
                with open(os.path.join(episode_data_dir, f"{cnt_step:04d}.json"), "w") as f:
                    json.dump(step_dict, f, indent=4)

                pts_pixs = np.vstack((pts_pixs, pts_pix))
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)

                # Add path to ax5, with colormap to indicate order
                ax5 = fig.axes[4]
                ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
                fig.tight_layout()
                if cfg.save_obs:
                    plt.savefig(
                        os.path.join(episode_data_dir, "{}_map.png".format(cnt_step + 1))
                    )
                    plt.close()
            rotation = quat_to_coeffs(
                quat_from_angle_axis(angle, np.array([0, 1, 0]))
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()
            
        metadata["episode_length"] = cnt_step
        with open(os.path.join(episode_data_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

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
