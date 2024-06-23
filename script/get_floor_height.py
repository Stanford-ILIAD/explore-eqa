import os, json, pickle, glob

os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import json
import numpy as np
from sklearn.cluster import KMeans
import habitat_sim  # takes time
from tqdm.notebook import tqdm

import sys
from pathlib import Path
# Get the directory of the current script
current_dir = Path(__file__).parent
# Get the parent directory
parent_dir = current_dir.parent
# Add the parent directory to sys.path
sys.path.append(str(parent_dir))
from src.habitat import (
    make_simple_cfg,
)

# Get scenes
scene_dir_train = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/train"
scene_dir_val = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/val"
scene_names_train = os.listdir(scene_dir_train)
scene_names = [name for name in scene_names_train if os.path.isdir(os.path.join(scene_dir_train, name))]
scene_names_val = os.listdir(scene_dir_val)
scene_names += [name for name in scene_names_val if os.path.isdir(os.path.join(scene_dir_val, name))]
print("Number of scenes:", len(scene_names))

def sample_random_points(sim, volume_sample_fac=1.0, significance_threshold=0.2):
    scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    scene_volume = scene_bb.size().product()
    points = np.array(
        [
            sim.pathfinder.get_random_navigable_point()
            for _ in range(int(scene_volume * volume_sample_fac))
        ]
    )

    hist, bin_edges = np.histogram(points[:, 1], bins="auto")
    significant_bins = (hist / len(points)) > significance_threshold
    l_bin_edges = bin_edges[:-1][significant_bins]
    r_bin_edges = bin_edges[1:][significant_bins]
    points_floors = {}
    for l_edge, r_edge in zip(l_bin_edges, r_bin_edges):
        points_floor = points[(points[:, 1] >= l_edge) & (points[:, 1] <= r_edge)]
        height = points_floor[:, 1].mean()
        points_floors[height] = points_floor
    return points_floors


# for each scene, load in habitat, get random navigable points, cluster them into floors,
# verify the number matches the topview
seed = 42
volume_sample_fac = 5.0
significance_threshold = 0.2
scene_floor_heights = {}
for scene_ind in tqdm(range(len(scene_names))):
    scene = scene_names[scene_ind]
    print(f"==== {scene} {scene_ind+1}/{len(scene_names)}====")

    # get number of floors based on topview
    # num_floor = len(glob.glob(os.path.join(topdown_dir, scene + "_floor_*.png")))
    num_floor = 1
    print("Number of floors from topview:", num_floor)

    # Load in habitat
    try:
        simulator.close()
    except:
        pass
    split = "train" if scene in scene_names_train else "val"
    if split == "train":
        scene_mesh_dir = os.path.join(scene_dir_train, scene, scene[6:] + ".basis" + ".glb")
        navmesh_file = os.path.join(scene_dir_train, scene, scene[6:] + ".basis" + ".navmesh")
    else:
        scene_mesh_dir = os.path.join(scene_dir_val, scene, scene[6:] + ".basis" + ".glb")
        navmesh_file = os.path.join(scene_dir_val, scene, scene[6:] + ".basis" + ".navmesh")
    sim_settings = {
        "scene": scene_mesh_dir,
        "default_agent": 0,
        "sensor_height": 1.5,
        "width": 480,
        "height": 480,
        "hfov": 100,
    }
    try:
        cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(seed)
        pathfinder.load_nav_mesh(navmesh_file)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
    except Exception as e:
        print("Error loading habitat:", e)
        continue

    # sample a lot of points
    points = sample_random_points(
        simulator,
        volume_sample_fac=volume_sample_fac,
        significance_threshold=significance_threshold,
    )
    num_point_cluster = len(points.keys())
    print("Number of point clusters:", num_point_cluster)

    # save
    scene_floor_heights[scene] = {
        "num_floor": num_floor,
        "num_point_cluster": num_point_cluster,
        "points": points,
    }
    if scene_ind % 20 == 0:
        # Save floor data
        with open("scene_floor_heights.pkl", "wb") as f:
            pickle.dump(scene_floor_heights, f)


# Save floor data
with open("scene_floor_heights.pkl", "wb") as f:
    pickle.dump(scene_floor_heights, f)
