import os, json, pickle, glob

os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import json
import numpy as np
from sklearn.cluster import KMeans
import habitat_sim  # takes time
from util.habitat import (
    make_simple_cfg,
)
from tqdm.notebook import tqdm

# Get scenes
scene_dir = "../hm3dsem/val"
scene_names = os.listdir(scene_dir)
topdown_dir = "../hm3dsem/topdown"


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
    num_floor = len(glob.glob(os.path.join(topdown_dir, scene + "_floor_*.png")))
    print("Number of floors from topview:", num_floor)

    # Load in habitat
    try:
        simulator.close()
    except:
        pass
    scene_mesh_dir = os.path.join(scene_dir, scene, scene[6:] + ".basis" + ".glb")
    navmesh_file = os.path.join(scene_dir, scene, scene[6:] + ".basis" + ".navmesh")
    sim_settings = {
        "scene": scene_mesh_dir,
        "default_agent": 0,
        "sensor_height": 1.5,
        "width": 480,
        "height": 480,
        "hfov": 100,
    }
    cfg = make_simple_cfg(sim_settings)
    simulator = habitat_sim.Simulator(cfg)
    pathfinder = simulator.pathfinder
    pathfinder.seed(seed)
    pathfinder.load_nav_mesh(navmesh_file)
    agent = simulator.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()

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


# Save floor data
with open("../hm3dsem/scene_floor_heights_val.pkl", "wb") as f:
    pickle.dump(scene_floor_heights, f)
