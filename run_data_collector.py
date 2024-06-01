"""
Run EQA in Habitat-Sim with VLM exploration.

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
import json
import logging
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis
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

    # Run all questions
    cnt_data = 0
    results_all = []
    # questions_data = questions_data[10:]
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

        # Set data dir for this question - set initial data to be saved
        episode_data_dir = os.path.join(cfg.dataset_output_dir, str(question_ind))
        episode_frontier_dir = os.path.join(cfg.frontier_dir, str(question_ind))
        os.makedirs(episode_data_dir, exist_ok=True)
        os.makedirs(episode_frontier_dir, exist_ok=True)
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

            # Get observation at current pose - skip black image, meaning robot is outside the floor
            obs = simulator.get_sensor_observations()
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]
            semantic_obs = obs["semantic_sensor"]

            # construct an frequency count map of each semantic id to a unique id
            masked_ids = np.unique(semantic_obs[depth > 3.0])
            semantic_obs = np.where(np.isin(semantic_obs, masked_ids), 0, semantic_obs)
            tsdf_planner.increment_scene_graph(semantic_obs, semantic_data)
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

                # Get VLM prediction
                rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
                # prompt_question = (
                #     vlm_question
                #     + "\nAnswer with the option's letter from the given choices directly."
                # )
                # logging.info(f"Prompt Pred: {prompt_text}")
                # smx_vlm_pred = vlm.get_loss(
                #     rgb_im, prompt_question, vlm_pred_candidates
                # )
                # smx_vlm_pred = np.ones((4)) / 4
                # logging.info(f"Pred - Prob: {smx_vlm_pred}")

                # Get VLM relevancy
                # prompt_rel = f"\nConsider the question: '{question}'. Are you confident about answering the question with the current view?"
                # # logging.info(f"Prompt Rel: {prompt_text}")
                # # smx_vlm_rel = vlm.get_loss(rgb_im, prompt_rel, ["Yes", "No"])
                # smx_vlm_rel = np.array([0.01, 0.99])
                # logging.info(f"Rel - Prob: {smx_vlm_rel}")

                # Get frontier candidates
                prompt_points_pix = []
                if cfg.use_active:
                    prompt_points_pix, fig = (
                        tsdf_planner.find_prompt_points_within_view(
                            pts_normal,
                            img_width,
                            img_height,
                            cam_intr,
                            cam_pose_tsdf,
                            **cfg.visual_prompt,
                        )
                    )
                    fig.tight_layout()
                    plt.savefig(
                        os.path.join(
                            episode_data_dir, "{}_prompt_points.png".format(cnt_step)
                        )
                    )
                    plt.close()

                # Visual prompting
                draw_letters = ["1", "2", "3", "4"]  # always four
                fnt = ImageFont.truetype(
                    "data/Open_Sans/static/OpenSans-Regular.ttf",
                    30,
                )
                actual_num_prompt_points = len(prompt_points_pix)
                # if actual_num_prompt_points >= cfg.visual_prompt.min_num_prompt_points:
                if True:
                    rgb_im_draw = rgb_im.copy()
                    draw = ImageDraw.Draw(rgb_im_draw)
                    for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
                        # draw.ellipse(
                        #     (
                        #         point_pix[0] - cfg.visual_prompt.circle_radius,
                        #         point_pix[1] - cfg.visual_prompt.circle_radius,
                        #         point_pix[0] + cfg.visual_prompt.circle_radius,
                        #         point_pix[1] + cfg.visual_prompt.circle_radius,
                        #     ),
                        #     fill=(200, 200, 200, 255),
                        #     outline=(0, 0, 0, 255),
                        #     width=3,
                        # )
                        draw.text(
                            tuple(point_pix.astype(int).tolist()),
                            draw_letters[prompt_point_ind],
                            font=fnt,
                            fill=(255, 0, 0, 255),
                            anchor="mm",
                            font_size=15,
                        )

                    rgb_im_draw.save(
                        os.path.join(episode_data_dir, f"{cnt_step}_draw.png")
                    )

                    for prompt_point_ind, point_pix in enumerate(prompt_points_pix):
                        # logging.info(f"Prompt point {prompt_point_ind}: {point_pix}")
                        width = 640
                        height = 480
                        size = 100
                        rgb_im_draw_cropped = rgb_im.crop(
                            (
                                max(point_pix[0] - size, 0),
                                max(point_pix[1] - size, 0),
                                min(point_pix[0] + size, width),
                                min(point_pix[1] + size, height),
                            )
                        )
                        rgb_im_draw_cropped.save(
                            os.path.join(
                                episode_data_dir, f"{cnt_step}_draw_{prompt_point_ind}.png"
                            )
                        )

                        

                    logging.info(f"Figure saved")

                    # # get VLM reasoning for exploring
                    # if cfg.use_lsv:
                    #     prompt_lsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it.\nWhich direction (black letters on the image) would you explore then? Answer with a single letter."
                    #     # logging.info(f"Prompt Exp: {prompt_text}")
                    #     lsv = vlm.get_loss(
                    #         rgb_im_draw,
                    #         prompt_lsv,
                    #         draw_letters[:actual_num_prompt_points],
                    #     )
                    #     lsv *= actual_num_prompt_points / 3
                    # else:
                    #     lsv = (
                    #         np.ones(actual_num_prompt_points) / actual_num_prompt_points
                    #     )

                    # # base - use image without label
                    # if cfg.use_gsv:
                    #     prompt_gsv = f"\nConsider the question: '{question}', and you will explore the environment for answering it. Is there any direction shown in the image worth exploring? Answer with Yes or No."
                    #     # logging.info(f"Prompt Exp base: {prompt_gsv}")
                    #     gsv = vlm.get_loss(rgb_im, prompt_gsv, ["Yes", "No"])[0]
                    #     gsv = (
                    #         np.exp(gsv / cfg.gsv_T) / cfg.gsv_F
                    #     )  # scale before combined with lsv
                    # else:
                    #     gsv = 1
                    # sv = lsv * gsv
                    # logging.info(f"Exp - LSV: {lsv} GSV: {gsv} SV: {sv}")

                    # # Integrate semantics only if there is any prompted point
                    # tsdf_planner.integrate_sem(
                    #     sem_pix=sv,
                    #     radius=1.0,
                    #     obs_weight=1.0,
                    # )  # voxel locations already saved in tsdf class

                # Save data
                # result[step_name]["smx_vlm_pred"] = smx_vlm_pred
                # result[step_name]["smx_vlm_rel"] = smx_vlm_rel
                result[step_name]["smx_vlm_pred"] = np.ones((4)) / 4
                result[step_name]["smx_vlm_rel"] = np.array([0.01, 0.99])
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
                frontier_voxel_pts = [np.array(key) for key in tsdf_planner.frontiers.keys()]
                frontier_world_pts = [
                    frontier * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2] for frontier in frontier_voxel_pts
                ]
                # Turn to face each frontier point and get rgb image
                print(f"Num Frontiers: {len(frontier_world_pts)}")
                for i in range(len(frontier_world_pts)):
                    frontier_pt = frontier_world_pts[i]
                    # Turn to face the frontier point
                    print(frontier_pt, pts)
                    if tsdf_planner.frontiers[tuple(frontier_voxel_pts[i])] is not None:
                        original_path = os.path.join(episode_frontier_dir, tsdf_planner.frontiers[tuple(frontier_voxel_pts[i])])
                        if os.path.exists(original_path):
                            target_path = os.path.join(episode_frontier_dir, f"{cnt_step}_frontier_{i}.png")
                            os.system(f"cp {original_path} {target_path}")
                    else:
                        angle = np.arctan2(frontier_pt[1] - pts[1], frontier_pt[0] - pts[0])
                        rotation = quat_to_coeffs(
                            quat_from_angle_axis(angle, np.array([0, 1, 0]))
                            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
                        ).tolist()
                        agent_state.rotation = rotation
                        agent.set_state(agent_state)
                        # Get observation at current pose - skip black image, meaning robot is outside the floor
                        obs = simulator.get_sensor_observations()
                        rgb = obs["color_sensor"]
                        plt.imsave(
                            os.path.join(episode_frontier_dir, f"{cnt_step}_frontier_{i}.png"),
                            rgb,
                        )
                        tsdf_planner.frontiers[tuple(frontier_voxel_pts[i])] = f"{cnt_step}_frontier_{i}.png"
                        # rotate back to original angle
                        angle = init_angle
                        rotation = quat_to_coeffs(
                            quat_from_angle_axis(angle, np.array([0, 1, 0]))
                            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
                        ).tolist()
                        agent_state.rotation = rotation
                        agent.set_state(agent_state)

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
        # Option 1: use the max of the weighted predictions
        smx_vlm_max = np.max(smx_vlm_all, axis=0)
        pred_token = candidates[np.argmax(smx_vlm_max)]
        success_weighted = pred_token == answer
        # Option 2: use the max of the relevancy
        max_relevancy = np.argmax(relevancy_all)
        relevancy_ord = np.flip(np.argsort(relevancy_all))
        pred_token = candidates[np.argmax(smx_vlm_all[max_relevancy])]
        success_max = pred_token == answer

        # Episode summary
        logging.info(f"\n== Episode Summary")
        logging.info(f"Scene: {scene}, Floor: {floor}")
        logging.info(f"Question: {question}, Choices: {choices}, Answer: {answer}")
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
