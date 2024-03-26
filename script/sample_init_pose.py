"""
Sample the initial pose of the robot in different scenes

"""

import os, pickle, random, shutil, logging

os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from src.habitat import (
    make_simple_cfg,
    pos_habitat_to_normal,
)

seed = 42
camera_height = 1.3  # ?
camera_tilt = 0  # ?
img_width = 1280
img_height = 1280
hfov = 100  # ?
hfov_rad = hfov * np.pi / 180
vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * img_height / img_width)
fx = (1.0 / np.tan(hfov_rad / 2.0)) * img_width / 2.0
fy = (1.0 / np.tan(vfov_rad / 2.0)) * img_height / 2.0
cx = img_width // 2
cy = img_height // 2
cam_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
min_avg_depth_initial = 2.0
min_depth_initial = 4.0
init_clearance = 0.5

# Data
save_data_parent_dir = "?"

# Save name
save_name = "?"

# Get the list of scenes and floors - already filtered out skipped ones
split = "train"
floor_data_path = "../scene_floor_heights_train.pkl"
with open(floor_data_path, "rb") as f:
    scene_floor_data = pickle.load(f)
scene_names = sorted(list(scene_floor_data.keys()))
print(f"{len(scene_names)} scenes loaded.")
# print(scene_names)

# log
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(os.path.join(save_data_parent_dir, "log.txt"), mode="w+"),
        logging.StreamHandler(),
    ],
)

num_data = 10000  # basically all floors
cnt_data = 0
results_all = {}
for scene_ind in tqdm(range(len(scene_names))):
    scene_num = int(scene_names[scene_ind][:5])

    scene_name = scene_names[scene_ind]
    floor_data = scene_floor_data[scene_name]

    # get floors from points data
    num_floor = floor_data["num_point_cluster"]
    floors_height = list(floor_data["points"].keys())
    for floor in range(num_floor):
        # get floor height
        floor_height = floors_height[floor]

        # debug
        logging.info(
            f"\n====== Scene cnt: {scene_ind+1}/{len(scene_names)} Floor: {floor} Scene: {scene_name} ======"
        )
        scene_floor_name = f"{scene_name}_{floor}"

        # load scene
        try:
            simulator.close()
        except:
            pass
        flag_train_split = int(scene_name[2]) < 8
        if flag_train_split:
            root = "?"
        else:
            root = "?"
        scene_mesh_dir = os.path.join(
            root, scene_name, scene_name[6:] + ".basis" + ".glb"
        )
        navmesh_file = os.path.join(
            root, scene_name, scene_name[6:] + ".basis" + ".navmesh"
        )
        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": 0,
            "sensor_height": camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": hfov,
        }
        cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(seed)
        pathfinder.load_nav_mesh(navmesh_file)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()

        # Sample random initial state
        max_try = 1000
        cnt_try = 0
        flag_skip_scene = False
        while 1:
            cnt_try += 1
            if cnt_try == max_try:
                flag_skip_scene = True
                break

            # get a random point
            pts = pathfinder.get_random_navigable_point()
            pts_normal = pos_habitat_to_normal(pts)

            # check if on the desired floor
            if abs(pts[1] - floor_height) > 0.3:
                continue

            # check sufficient clearance
            if pathfinder.distance_to_closest_obstacle(pts) < init_clearance:
                continue

            # check depth and black pixels
            max_try_angle = 10
            cnt_try_angle = 0
            flag_next_point = False
            while 1:
                cnt_try_angle += 1
                if cnt_try_angle == max_try_angle:
                    # logging.info('Failed to find valid locations.')
                    flag_next_point = True
                    break

                angle = np.random.uniform(0, 2 * np.pi)
                rotation = quat_to_coeffs(
                    quat_from_angle_axis(angle, np.array([0, 1, 0]))
                    * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
                ).tolist()
                agent_state.position = pts
                agent_state.rotation = rotation
                agent.set_state(agent_state)
                obs = simulator.get_sensor_observations()
                rgb = obs["color_sensor"]
                num_black_pixels = np.sum(rgb == 0)
                if num_black_pixels > 0.2 * img_width * img_height:
                    # print('black')
                    continue
                depth = obs["depth_sensor"]
                depth_filtered = depth[depth > 0.0000001]
                # print(np.mean(depth_filtered), -np.percentile(-depth_filtered, 80))
                if (
                    # check zero-size array
                    depth_filtered.size == 0
                    or np.mean(depth_filtered) < min_avg_depth_initial
                    or np.max(depth_filtered) < min_depth_initial
                    # or -np.percentile(-depth_filtered, 80) < 1
                ):
                    # print('depth!')
                    continue
                break
            if not flag_next_point:
                break

        # quit
        if not flag_skip_scene:
            results_all[scene_floor_name] = {}
            results_all[scene_floor_name]["pts"] = pts
            results_all[scene_floor_name]["angle"] = angle

            # Get rgb
            plt.imsave(
                os.path.join(save_data_parent_dir, scene_floor_name, "init_dec19.png"),
                rgb,
            )

            # cnt
            cnt_data += 1
    if cnt_data >= num_data:
        break

# Finish
print(f"{cnt_data} data collected.")

# Save data
with open(os.path.join(save_data_parent_dir, save_name + ".pkl"), "wb") as f:
    pickle.dump(results_all, f)
