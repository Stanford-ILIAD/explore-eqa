import os, pickle

os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import json
import numpy as np
import habitat_sim  # takes time
from src.habitat import (
    make_simple_cfg,
    pos_habitat_to_normal,
)
from tqdm.notebook import tqdm

# Get scenes
split = "val"
scene_dir = "?"
scene_names = os.listdir(scene_dir)
print("Original number of scenes:", len(scene_names))

# load every scene in habitat
scene_size = {}
for scene_ind in tqdm(range(len(scene_names))):
    scene = scene_names[scene_ind]
    print(f"==== {scene} {scene_ind+1}/{len(scene_names)}====")

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
    pathfinder.seed(42)
    pathfinder.load_nav_mesh(navmesh_file)
    agent = simulator.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()

    # Get mesh boundaries - this is for the full scene
    scene_bnds = pathfinder.get_bounds()
    scene_lower_bnds_normal = pos_habitat_to_normal(scene_bnds[0])
    scene_upper_bnds_normal = pos_habitat_to_normal(scene_bnds[1])
    size = np.sum((scene_upper_bnds_normal[:2] - scene_lower_bnds_normal[:2]) ** 2)
    print("Scene size:", size)
    print("Scene boundaries:", scene_lower_bnds_normal, scene_upper_bnds_normal)

    # or use topdown map - more precise at height levels
    # https://gist.github.com/mathfac/9cf247d3ab0abf27d73e2a0cf63272aa

    # save
    scene_size[scene] = {
        "dim": [scene_lower_bnds_normal, scene_upper_bnds_normal],
        "size": size,
    }

# Save floor data
with open(f"?.pkl", "wb") as f:
    pickle.dump(scene_size, f)
