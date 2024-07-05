"""
Run EQA in Habitat-Sim with CLIP for exploration.

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
import pickle
import logging
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
from src.habitat import (
    make_simple_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
)
from src.geom import get_cam_intr, get_scene_bnds
from src.vlm import VLM
from src.tsdf import TSDFPlanner
from CLIP.clip import ClipWrapper, saliency_configs


def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

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

    # Load VLM
    vlm = VLM(cfg.vlm)

    # Run all questions
    cnt_data = 0
    results_all = []
    for question_ind in tqdm(range(len(questions_data))):

        # Extract question
        question_data = questions_data[question_ind]
        scene = question_data["scene"]
        floor = question_data["floor"]
        scene_floor = scene + "_" + floor
        question = question_data["question"]
        choices = question_data["choices"]
        choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
        answer = question_data["answer"]
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]
        logging.info(f"\n\n========\nInd: {question_ind} Scene: {scene} Floor: {floor}")
        logging.info(f"Question: {question} Choices: {choices}, Answer: {answer}")

        # Re-format the question to follow LLaMA style
        vlm_question = question
        vlm_pred_candidates = ["A", "B", "C", "D"]
        for token, choice in zip(vlm_pred_candidates, choices):
            vlm_question += "\n" + token + "." + " " + choice

        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(cfg.output_dir, str(question_ind))
        os.makedirs(episode_data_dir, exist_ok=True)
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

        # Run steps
        pts_pixs = np.empty((0, 2))  # for plotting path on the image
        for cnt_step in range(num_step):
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

            # Get observation at current pose - skip black image, meaning robot is outside the house
            obs = simulator.get_sensor_observations()
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]
            if cfg.save_obs:
                plt.imsave(
                    os.path.join(episode_data_dir, "{}.png".format(cnt_step)), rgb
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

                # Get VLM prediction
                rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
                prompt_question = (
                    vlm_question
                    + "\nAnswer with the option's letter from the given choices directly."
                )
                # logging.info(f"Prompt Pred: {prompt_text}")
                smx_vlm_pred = vlm.get_loss(
                    rgb_im, prompt_question, vlm_pred_candidates
                )
                logging.info(f"Pred - Prob: {smx_vlm_pred}")

                # Get VLM relevancy
                prompt_rel = f"\nConsider the question: '{question}'. Are you confident about answering the question with the current view? Answer with Yes or No."
                # logging.info(f"Prompt Rel: {prompt_text}")
                smx_vlm_rel = vlm.get_loss(rgb_im, prompt_rel, ["Yes", "No"])
                logging.info(f"Rel - Prob: {smx_vlm_rel}")

                # Get CLIP relevancy
                clip_grads = ClipWrapper.get_clip_saliency(
                    img=rgb,  # use original array
                    text_labels=[question],
                    prompts=["{}"],
                    **saliency_configs["ours"](img_height),
                )[0]
                clip_grad = clip_grads.cpu().numpy()[0]  # only one label
                vmin, vmax = 0.005, 0.020  # normalization
                cmap = plt.get_cmap("jet")
                clip_grad = np.clip(
                    (clip_grad - vmin) / (vmax - vmin), a_min=0.0, a_max=1.0
                )

                # Save CLIP image
                fig = plt.figure()
                plt.imshow(rgb)
                colored_clip_grad_rgba = cmap(clip_grad)
                colored_clip_grad_rgba[..., -1] = (
                    1 - clip_grad
                ) * 0.7  # for visualizing
                plt.imshow(colored_clip_grad_rgba)
                sm = plt.cm.ScalarMappable(
                    cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
                )
                plt.colorbar(sm, ax=plt.gca())
                plt.savefig(os.path.join(episode_data_dir, f"{cnt_step}_clip_grad.png"))
                plt.close()

                # Integrate semantics
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=cam_pose_tsdf,
                    sem_im=clip_grad,
                    obs_weight=1.0,
                )
                logging.info(f"Semantic: {np.max(clip_grad)}")

                # Save data
                result[step_name]["smx_vlm_pred"] = smx_vlm_pred
                result[step_name]["smx_vlm_rel"] = smx_vlm_rel
            else:
                logging.info("Skipping black image!")
                result[step_name]["smx_vlm_pred"] = np.ones((4)) / 4
                result[step_name]["smx_vlm_rel"] = np.array([0.01, 0.99])

            # Determine next point
            if cnt_step < num_step:
                pts_normal, angle, pts_pix, fig = tsdf_planner.find_next_pose(
                    pts=pts_normal,
                    angle=angle,
                    flag_no_val_weight=cnt_step < cfg.min_random_init_steps,
                    **cfg.planner,
                )
                pts_pixs = np.vstack((pts_pixs, pts_pix))
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)

                # Add path to ax5, with colormap to indicate order
                ax5 = fig.axes[4]
                ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)
                fig.tight_layout()
                plt.savefig(
                    os.path.join(episode_data_dir, "{}_map.png".format(cnt_step + 1))
                )
                plt.close()
            rotation = quat_to_coeffs(
                quat_from_angle_axis(angle, np.array([0, 1, 0]))
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()

        # Check if success using weighted prediction
        smx_vlm_all = np.empty((0, 4))
        relevancy_all = []
        candidates = ["A", "B", "C", "D"]
        for step in range(num_step):
            smx_vlm_pred = result[f"step_{step}"]["smx_vlm_pred"]
            smx_vlm_rel = result[f"step_{step}"]["smx_vlm_rel"]
            relevancy_all.append(smx_vlm_rel[0])
            smx_vlm_all = np.vstack((smx_vlm_all, smx_vlm_rel[0] * smx_vlm_pred))
        # use the max of the weighted predictions
        smx_vlm_max = np.max(smx_vlm_all, axis=0)
        pred_token = candidates[np.argmax(smx_vlm_max)]
        success_weighted = pred_token == answer
        # use the max of the relevancy
        max_relevancy = np.argmax(relevancy_all)
        relevancy_ord = np.flip(np.argsort(relevancy_all))
        pred_token = candidates[np.argmax(smx_vlm_all[max_relevancy])]
        success_max = pred_token == answer

        # Summary
        logging.info(f"\n== Trial Summary")
        logging.info(f"Scene: {scene}, Floor: {floor}")
        logging.info(f"Question:\n{vlm_question}\nAnswer: {answer}")
        logging.info(f"Success (weighted): {success_weighted}")
        logging.info(f"Success (max): {success_max}")
        logging.info(
            f"Top 3 steps with highest relevancy with value: {relevancy_ord[:3]} {[relevancy_all[i] for i in relevancy_ord[:3]]}"
        )
        for rel_ind in range(3):
            logging.info(f"Prediction: {smx_vlm_all[relevancy_ord[rel_ind]]}")

        # Save data
        results_all.append(result)
        cnt_data += 1
        if cnt_data % cfg.save_freq == 0:
            with open(
                os.path.join(cfg.output_dir, f"results_{cnt_data}.pkl"), "wb"
            ) as f:
                pickle.dump(results_all, f)

    # Save all data again
    with open(os.path.join(cfg.output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results_all, f)
    logging.info(f"\n== All Summary")
    logging.info(f"Number of data collected: {cnt_data}")


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
