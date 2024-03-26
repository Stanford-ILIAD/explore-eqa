"""
Run a scene and navigate randomly in Habitat-Sim.

"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np

np.set_printoptions(precision=3)
import csv
import logging
import quaternion
import random
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from src.habitat import (
    make_simple_cfg,
)


def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width

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

    # Sample a random question
    question_data = random.choice(questions_data)
    scene = question_data["scene"]
    floor = question_data["floor"]
    scene_floor = scene + "_" + floor
    question = question_data["question"]
    choices = question_data["choices"]
    answer = question_data["answer"]
    init_pts = init_pose_data[scene_floor]["init_pts"]
    init_angle = init_pose_data[scene_floor]["init_angle"]
    logging.info(f"\n========\nScene: {scene} Floor: {floor}")
    logging.info(f"Question: {question} Choices: {choices}, Answer: {answer}")

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
    }
    sim_cfg = make_simple_cfg(sim_settings)
    simulator = habitat_sim.Simulator(sim_cfg)
    pathfinder = simulator.pathfinder
    pathfinder.seed(cfg.seed)
    pathfinder.load_nav_mesh(navmesh_file)
    agent = simulator.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    pts = init_pts
    angle = init_angle
    rotation = quat_to_coeffs(
        quat_from_angle_axis(angle, np.array([0, 1, 0]))
        * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
    ).tolist()

    # Run random navigation
    num_step = 10
    for cnt_step in range(num_step):
        logging.info(f"\n== step: {cnt_step} pts: {pts}")

        # Set current pose
        agent_state.position = pts
        agent_state.rotation = rotation
        agent.set_state(agent_state)

        # Update camera info
        sensor = agent.get_state().sensor_states["depth_sensor"]
        quaternion_0 = sensor.rotation
        translation_0 = sensor.position
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
        cam_pose[:3, 3] = translation_0

        # Get observation at current pose - skip black image, meaning robot is outside the floor
        obs = simulator.get_sensor_observations()
        rgb = obs["color_sensor"]
        depth = obs["depth_sensor"]

        # Sample random next location
        pts = pathfinder.get_random_navigable_point()
        angle = random.uniform(0, 2 * np.pi)
        rotation = quat_to_coeffs(
            quat_from_angle_axis(angle, np.array([0, 1, 0]))
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()


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
