import os, pickle, random, shutil, logging, torch

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
from transformers import CLIPProcessor, CLIPModel

def get_similarity(img_feature, text_feature):
    img_feature = img_feature / img_feature.norm()
    text_feature = text_feature / text_feature.norm()
    return (img_feature @ text_feature.T).item()

# view 1
pts_1 = np.asarray([0., 0, 0.])
angle_1 = 60.0 / 180.0 * np.pi

# view 2
pts_2 = np.asarray([-3.5, 0, -0.5])
angle_2 = -75.0 / 180.0 * np.pi

# view 3
angle_3 = -30.0 / 180.0 * np.pi
pts_3 = np.asarray([-6., 0, 4.5])

# view 4
angle_4 = -45.0 / 180.0 * np.pi
pts_4 = np.asarray([-11., 0, 0.])

scene_path = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yuncong/data/00802-wcojb4TFT35'
save_dir = 'clip_experiments'
text = 'Where is the red pillow?'
os.makedirs(save_dir, exist_ok=True)

seed = 42
camera_height = 1.2
camera_tilt = 0
img_width = 1280
img_height = 1280
hfov = 100
hfov_rad = hfov * np.pi / 180
vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * img_height / img_width)
fx = (1.0 / np.tan(hfov_rad / 2.0)) * img_width / 2.0
fy = (1.0 / np.tan(vfov_rad / 2.0)) * img_height / 2.0
cx = img_width // 2
cy = img_height // 2
cam_intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
min_avg_depth_initial = 1.0  # smaller than before

# Load the scene
scene_name = scene_path.split("/")[-1].split("-")[-1]
scene_mesh_dir = os.path.join(scene_path, scene_name + '.basis' + '.glb')
navmesh_file = os.path.join(scene_path, scene_name + '.basis' + '.navmesh')
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

rotation_1 = quat_to_coeffs(
    quat_from_angle_axis(angle_1, np.array([0, 1, 0]))
    * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
).tolist()
agent_state.position = pts_1
agent_state.rotation = rotation_1
agent.set_state(agent_state)
obs = simulator.get_sensor_observations()
rgb_1 = obs["color_sensor"]  # (H, W, 4), uint8
# save image
plt.imsave(os.path.join(save_dir, 'view_1.png'), rgb_1)

rotation_2 = quat_to_coeffs(
    quat_from_angle_axis(angle_2, np.array([0, 1, 0]))
    * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
).tolist()
agent_state.position = pts_2
agent_state.rotation = rotation_2
agent.set_state(agent_state)
obs = simulator.get_sensor_observations()
rgb_2 = obs["color_sensor"]  # (H, W, 4), uint8
# save image
plt.imsave(os.path.join(save_dir, 'view_2.png'), rgb_2)

rotation_3 = quat_to_coeffs(
    quat_from_angle_axis(angle_3, np.array([0, 1, 0]))
    * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
).tolist()
agent_state.position = pts_3
agent_state.rotation = rotation_3
agent.set_state(agent_state)
obs = simulator.get_sensor_observations()
rgb_3 = obs["color_sensor"]  # (H, W, 4), uint8
# save image
plt.imsave(os.path.join(save_dir, 'view_3.png'), rgb_3)

rotation_4 = quat_to_coeffs(
    quat_from_angle_axis(angle_4, np.array([0, 1, 0]))
    * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
).tolist()
agent_state.position = pts_4
agent_state.rotation = rotation_4
agent.set_state(agent_state)
obs = simulator.get_sensor_observations()
rgb_4 = obs["color_sensor"]  # (H, W, 4), uint8
# save image
plt.imsave(os.path.join(save_dir, 'view_4.png'), rgb_4)






















