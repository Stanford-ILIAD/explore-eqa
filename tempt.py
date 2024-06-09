import os

os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np

import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from src.habitat import (
    make_simple_cfg,
    pos_habitat_to_normal,
)

all_dir = '../../../multisensory/MLLM/data/hm3d/train/'

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

# available_scene_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/scene_feature_dict'
# all_scenes = os.listdir(available_scene_dir)
# all_scenes = [scene_name for scene_name in all_scenes if os.path.isdir(os.path.join(available_scene_dir, scene_name))]
# all_scenes = [scene_name for scene_name in all_scenes if '008' not in scene_name]
all_scenes = os.listdir(all_dir)

total_scene = len(all_scenes)
success_count = 0
failed_scenes = []
for idx, scene_name in enumerate(all_scenes):
    try:
        simulator.close()
    except:
        pass

    scene_path = os.path.join(all_dir, scene_name)
    scene_name_short = scene_path.split("/")[-1].split("-")[-1]
    scene_mesh_dir = os.path.join(scene_path, scene_name_short + '.basis' + '.glb')
    navmesh_file = os.path.join(scene_path, scene_name_short + '.basis' + '.navmesh')
    assert os.path.exists(scene_mesh_dir) and os.path.exists(navmesh_file), f"{scene_mesh_dir}_{navmesh_file}"
    sim_settings = {
        "scene": scene_mesh_dir,
        "default_agent": 0,
        "sensor_height": camera_height,
        "width": img_width,
        "height": img_height,
        "hfov": hfov,
    }

    try:
        cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(cfg)
        pathfinder = simulator.pathfinder

        success = pathfinder.load_nav_mesh(navmesh_file)

    except:
        success = False

    if not success:
        print(f"{idx}/{total_scene}   Failed in loading navmesh: {scene_name}")
        failed_scenes.append(scene_name)
    else:
        success_count += 1
        print(f"{idx}/{total_scene}   Success: {scene_name}")

print(f"Failed: {'   '.join(failed_scenes)}")
print(f"Success: {success_count}/{total_scene}")