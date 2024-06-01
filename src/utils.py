import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
import habitat_sim
import base64


def display_sample(rgb_obs, semantic_obs, depth_obs, class_obs, save_path, title=None):
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")

    class_img = Image.new("P", (class_obs.shape[1], class_obs.shape[0]))
    class_img.putpalette(d3_40_colors_rgb.flatten())
    class_img.putdata((class_obs.flatten() % 40).astype(np.uint8))
    class_img = class_img.convert("RGBA")

    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

    arr = [rgb_img, depth_img, semantic_img, class_img]
    titles = ['rgb', 'depth', 'semantic', 'class']
    plt.figure(figsize=(16 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 4, i + 1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)

    if title is not None:
        plt.suptitle(title)

    plt.savefig(save_path)


def print_scene_recur(scene, limit_output=10):
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None


def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor, a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    sem_sensor_spec = habitat_sim.CameraSensorSpec()
    sem_sensor_spec.uuid = "semantic"
    sem_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem_sensor_spec.resolution = [settings["height"], settings["width"]]
    sem_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    sem_sensor_spec.hfov = settings["hfov"]

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, sem_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def pos_habitat_to_normal(pts):
    # -90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")