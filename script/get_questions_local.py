import os
import pickle
import json
import logging
import random

os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

import numpy as np
import quaternion


from openai import AzureOpenAI
from PIL import Image

from src.utils import *
from script.background_prompts import background_prompts
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis, quat_from_two_vectors


if __name__ == "__main__":
    dataset_path = '/home/hanyang/code/3d_project/explore-eqa/data/versioned_data/hm3d-0.2/hm3d'
    scene_list = ['00324-DoSbsoo4EAg', '00606-W16Bm4ysK8v', '00669-DNWbUAJYsPy']
    scene_dataset_config_file = '/home/hanyang/code/3d_project/explore-eqa/data/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    bounding_box_dir = '/home/hanyang/code/3d_project/explore-eqa/data/hm3d_obj_bbox_merged'
    train_floor_data_path = "data/scene_floor_heights_train.pkl"
    val_floor_data_path = "data/scene_floor_heights_val.pkl"
    save_dir = 'generated_questions'
    os.makedirs(save_dir, exist_ok=True)
    fewshot_example_dir = 'fewshot_examples'
    # question_categories = ['object_recognition', 'object_state_recognition', 'attribute_recognition',
    #                        'functional_reasoning', 'object_localization', 'world_knowledge', 'spatial_understanding']
    question_categories = ['functional_reasoning']
    sys_prompt = "You are a helpful assistant."

    # set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, "generate_qa_local_output.txt"), mode="w"),
            logging.StreamHandler(),
        ],
    )

    # about scene filtering
    num_distance_range = 4
    min_clearance = 0.1
    min_observe_radius = 1.5
    max_observe_radius = 3.0
    max_rare_class_num = 2  # The rare class should have only 1 object in the whole house
    min_obj_pix_ratio = 1e-5  # The object should account for at least 1e-5 of the image to be counted

    min_rare_obj_pix_percentage = 0.01  # The rare object should account for at least 1% of the image to be counted
    max_rare_obj_pix_percentage = 0.06  # The rare object should account for at most 6% of the image to be counted
    min_observe_distance_ratio = 1.5  # Observe distance should be at least 1.5 times the object's size

    # about generating questions
    n_shot_positive = 4
    n_shot_negative = 4

    # about camera
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
    min_avg_depth_initial = 0.5  # smaller than before

    client = AzureOpenAI(
        azure_endpoint='https://yuncong.openai.azure.com',
        api_key=os.getenv('AZURE_OPENAI_KEY'),
        api_version='2024-02-15-preview',
    )

    scene_path_list = []
    for scene_name in scene_list:
        scene_id = int(scene_name.split('-')[0])
        if scene_id < 800:
            scene_path_list.append(os.path.join(dataset_path, 'train', scene_name))
        else:
            scene_path_list.append(os.path.join(dataset_path, 'val', scene_name))

    # traverse each scene
    for scene_path in scene_path_list:
        # Load the scene
        scene_name = scene_path.split("/")[-1].split("-")[-1]
        scene_mesh_dir = os.path.join(scene_path, scene_name + '.basis' + '.glb')
        navmesh_file = os.path.join(scene_path, scene_name + '.basis' + '.navmesh')
        scene_semantic_texture_file = os.path.join(scene_path, scene_name + ".semantic" + ".glb")
        scene_semantic_annotation_file = os.path.join(scene_path, scene_name + ".semantic" + ".txt")
        assert os.path.exists(scene_mesh_dir) and os.path.exists(navmesh_file)
        assert os.path.exists(scene_semantic_texture_file) and os.path.exists(scene_semantic_annotation_file)

        try:
            simulator.close()
        except:
            pass

        sim_settings = {
            "scene": scene_mesh_dir,
            "default_agent": 0,
            "sensor_height": camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": hfov,
            "scene_dataset_config_file": scene_dataset_config_file,
        }
        cfg = make_simple_cfg(sim_settings)
        simulator = habitat_sim.Simulator(cfg)

        scene = simulator.semantic_scene
        logging.info(f"Scene loaded: {scene_name}")

        # Load the semantic annotation
        obj_id_to_class = {}
        obj_id_to_room_id = {}
        with open(scene_semantic_annotation_file, "r") as f:
            for line in f.readlines():
                if 'HM3D Semantic Annotations' in line:  # skip the first line
                    continue
                line = line.strip().split(',')
                idx = int(line[0])
                class_name = line[2].replace("\"", "")
                room_id = int(line[3])
                obj_id_to_class[idx] = class_name
                obj_id_to_room_id[idx] = room_id
        obj_id_to_class[0] = 'unannotated'

        # get unique objects: find class that only have 1 object
        class_to_object = {}
        for obj_id in obj_id_to_class:
            class_name = obj_id_to_class[obj_id]
            if class_name not in class_to_object:
                class_to_object[class_name] = []
            class_to_object[class_name].append(obj_id)
        rare_class_list = [key for key in class_to_object if 0 < len(class_to_object[key]) <= max_rare_class_num]
        rare_class_list.remove('unannotated')
        logging.info(f'Rare classes: {rare_class_list}')

        # load the floor data
        scene_id = int(scene_path.split("/")[-1].split("-")[0])
        if scene_id < 800:
            floor_data_path = train_floor_data_path
        else:
            floor_data_path = val_floor_data_path
        with open(floor_data_path, "rb") as f:
            scene_floor_data = pickle.load(f)
        # get floors from points data
        floor_data = scene_floor_data[scene_path.split("/")[-1]]
        num_floor = floor_data["num_point_cluster"]
        floors_height = list(floor_data["points"].keys())

        # Load the navigable maps
        pathfinder = simulator.pathfinder
        pathfinder.seed(seed)
        pathfinder.load_nav_mesh(navmesh_file)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()

        # load bounding box data
        bounding_box_data = json.load(open(os.path.join(bounding_box_dir, scene_path.split("/")[-1] + ".json"), "r"))
        object_id_to_bbox = {int(item['id']): item['bbox'] for item in bounding_box_data}

        # load generated questions save file
        generated_questions_save_file = os.path.join(save_dir, f"generated_questions.json")
        if os.path.exists(generated_questions_save_file):
            generated_question_list = json.load(open(generated_questions_save_file, "r"))
            logging.info(f'Loaded {len(generated_question_list)} previously generated questions')
        else:
            generated_question_list = []

        # for each object within the rare class, get several views
        for rare_class in rare_class_list:
            # get all objects within this class
            rare_obj_list = class_to_object[rare_class]

            for rare_obj_id in rare_obj_list:
                # get the object's bounding box
                if rare_obj_id not in object_id_to_bbox:
                    continue
                obj_bbox = object_id_to_bbox[rare_obj_id]
                obj_bbox = np.asarray(obj_bbox)  # (2, 3)
                obj_bbox_center = np.mean(obj_bbox, axis=0) # (x, z, y)
                # change to x, y, z for habitat
                obj_bbox_center = obj_bbox_center[[0, 2, 1]]
                object_height = obj_bbox_center[1]
                # get the floor height of that object for finding a navigable point
                floor_height = [height for height in floors_height if height - 0.1 < object_height]  # allow 10cm error for some object's detected height might be not accurate
                if len(floor_height) == 0:
                    continue
                floor_height = max(floor_height)
                # get the object's max radius
                object_length = np.max(np.abs(obj_bbox[0] - obj_bbox[1]))

                valid_observations_all = []  # store valid observations for all distances
                for cnt_view in range(num_distance_range):
                    # calculate the observe radius
                    observe_radius = min_observe_radius + (max_observe_radius - min_observe_radius) * cnt_view / (num_distance_range - 1)

                    max_try = 100
                    count_try = 0
                    while True:
                        count_try += 1
                        if count_try > max_try:
                            logging.info(f'Cannot find a navigable point near the object {rare_obj_id}: {rare_class} in distance {observe_radius}!')
                            break

                        # get a random point
                        try:
                            pts = pathfinder.get_random_navigable_point_near(
                                circle_center=np.asarray([obj_bbox_center[0], floor_height, obj_bbox_center[2]]),
                                radius=observe_radius
                            )
                        except Exception as e:
                            logging.debug(f'{rare_obj_id}-{rare_class}-{count_try}: pathfinder cannot find a navigable point near the object!')
                            continue
                        # if pts has nan
                        if np.isnan(pts).any():
                            logging.debug(f'{rare_obj_id}-{rare_class}-{count_try}: pathfinder find a point with nan!')
                            continue
                        # check if on the desired floor
                        if abs(pts[1] - floor_height) > 0.3:
                            logging.debug(f'{rare_obj_id}-{rare_class}-{count_try}: Not on the desired floor!')
                            continue
                        actual_distance = np.linalg.norm(pts - np.asarray([obj_bbox_center[0], pts[1], obj_bbox_center[2]]))
                        if actual_distance < observe_radius * 0.9:
                            logging.debug(f'{rare_obj_id}-{rare_class}-{count_try}: actual distance/observe radius: {actual_distance}/{observe_radius}')
                            continue

                        # check sufficient clearance
                        if pathfinder.distance_to_closest_obstacle(pts) < min_clearance:  # was 0.1
                            logging.debug(f'{rare_obj_id}-{rare_class}-{count_try}: Not enough clearance!')
                            continue

                        # get viewing direction: towards the object
                        viewing_direction = obj_bbox_center - np.asarray([pts[0], camera_height + pts[1], pts[2]])
                        default_direction = np.array([0.0, 0.0, -1.0])
                        intermediate_direction = viewing_direction.copy()
                        intermediate_direction[1] = 0.0
                        rotation = quat_to_coeffs(
                            quat_from_two_vectors(intermediate_direction, viewing_direction) *
                            quat_from_two_vectors(default_direction, intermediate_direction)
                        ).tolist()
                        rotation_to_save = quat_to_coeffs(  # the actual rotation can only be about y-axis
                            quat_from_two_vectors(default_direction, intermediate_direction)
                        ).tolist()

                        agent_state.position = pts
                        agent_state.rotation = rotation
                        agent.set_state(agent_state)
                        obs = simulator.get_sensor_observations()

                        # filter out observations with too many black pixels
                        rgb = obs["color_sensor"]  # (H, W, 4), uint8
                        num_black_pixels = np.sum(rgb == 0)
                        if num_black_pixels > 0.1 * img_width * img_height:
                            logging.debug(f'{rare_obj_id}-{rare_class}-{count_try}: Too many black pixels!')
                            continue
                        depth = obs["depth_sensor"]  # (H, W), float32
                        max_depth = np.max(depth)
                        depth_filtered = depth[depth > 0.0000001]

                        if (
                                # check zero-size array
                                depth_filtered.size == 0
                                or np.mean(depth_filtered) < min_avg_depth_initial
                                # or np.max(depth_filtered) < min_depth_initial
                                or -np.percentile(-depth_filtered, 80) < 0.5
                        ):  # 20% quantile is too small
                            logging.debug(f'{rare_obj_id}-{rare_class}-{count_try}: Depth is not valid!')
                            continue

                        # get the number of occupied pixel of this rare object
                        rare_obj_pix_ratio = np.sum(obs["semantic"] == rare_obj_id) / (img_width * img_height)


                        ######################################
                        # do filtering
                        select_current_view = False
                        if min_rare_obj_pix_percentage < rare_obj_pix_ratio < max_rare_obj_pix_percentage:
                            if actual_distance > min_observe_distance_ratio * object_length:
                                select_current_view = True

                        if not select_current_view:
                            logging.debug(f'{rare_obj_id}-{rare_class}-{count_try}: filtered out by conditions!')
                            continue
                        ######################################


                        valid_observations_all.append({
                            "rare_obj_id": rare_obj_id,
                            "rare_class": rare_class,
                            "position": pts,
                            "rotation": rotation_to_save,
                            "rgb": obs["color_sensor"],
                            "depth": obs["depth_sensor"],
                            "semantic": obs["semantic"],
                        })

                        logging.info(f'Object id: {rare_obj_id}, Class: {rare_class}, View: {cnt_view} saved!')

                        break


                # if there are valid observations, use the last one to generate questions
                if len(valid_observations_all) > 0:
                    valid_observation = valid_observations_all[-1]
                    focus_class = valid_observation["rare_class"]
                    reference_img = valid_observation["rgb"]

                    question_category = random.choice(question_categories)

                    # construct the prompt
                    prompt_bg = background_prompts[question_category]
                    content = [{"type": "text", "text": prompt_bg}]

                    fewshot_data_dir = os.path.join(fewshot_example_dir, question_category)
                    fewshot_examples = json.load(open(os.path.join(fewshot_data_dir, 'questions.json'), "r"))

                    # load positive examples
                    positive_example_count = 0
                    for fewshot_ind, fewshot_example in enumerate(fewshot_examples):
                        if not fewshot_example['exist']:
                            continue

                        view_base_64_image_all = []
                        view_image_path = os.path.join(fewshot_data_dir, fewshot_example['file'])
                        view_base_64_image_all.append(encode_image(view_image_path))

                        question = fewshot_example['question']
                        answer = fewshot_example['answer']
                        reference_object = fewshot_example['reference_object']
                        fewshot_prompt = f"Reference object: {reference_object}\nQuestion: {question}\nAnswer: {answer}\n"

                        content += [{"type": "text", "text": f"Example {positive_example_count + 1}:\n"}]
                        content += [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high",
                                },
                            }
                            for base64_image in view_base_64_image_all
                        ]
                        content += [{"type": "text", "text": fewshot_prompt}]

                        positive_example_count += 1
                        if positive_example_count >= n_shot_positive:
                            break

                    # load negative examples
                    negative_example_count = 0
                    for fewshot_ind, fewshot_example in enumerate(fewshot_examples):
                        if fewshot_example['exist']:
                            continue

                        view_base_64_image_all = []
                        view_image_path = os.path.join(fewshot_data_dir, fewshot_example['file'])
                        view_base_64_image_all.append(encode_image(view_image_path))

                        reference_object = fewshot_example['reference_object']
                        explanation = fewshot_example['explanation']
                        fewshot_prompt = f"Reference object: {reference_object}\nNot proper to generate a question\nExplanation: {explanation}\n"

                        content += [{"type": "text", "text": f"Example {positive_example_count + negative_example_count + 1}:\n"}]
                        content += [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high",
                                },
                            }
                            for base64_image in view_base_64_image_all
                        ]
                        content += [{"type": "text", "text": fewshot_prompt}]

                        negative_example_count += 1
                        if negative_example_count >= n_shot_negative:
                            break

                    # generate a random question id
                    question_id = f'{scene_path.split("/")[-1]}_{rare_obj_id}_{rare_class}_{random.randint(0, 1000000)}'

                    # save the observation image
                    img_save_path = os.path.join(save_dir, f"{question_id}.png")
                    Image.fromarray(reference_img).save(img_save_path)

                    view_base_64_image = encode_image(img_save_path)
                    content += [
                        {
                            "type": "text",
                            "text": (
                                "Now can you generate a question and its answer following previous format "
                                "based on the view below:\n"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{view_base_64_image}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Reference object: {focus_class}\n"
                        }
                    ]

                    output = None
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                                {"role": "user", "content": content},
                            ],
                            max_tokens=500,
                            seed=42,
                            temperature=0.2
                        )

                        output = response.choices[0].message.content
                        logging.info(f"Output: {output}")
                    except Exception as e:
                        logging.info(f"Error: {e}")

                    if output is not None:
                        if "Question: " in output and "Answer: " in output:
                            gen_question = output.split("Question: ")[1].split('Answer: ')[0].strip()
                            gen_answer = output.split("Answer: ")[1].strip()

                            generated_question = {
                                "question_id": question_id,
                                "episode_history": scene_path.split('/')[-1],
                                "category": question_category.replace('_', ' '),
                                "question": gen_question,
                                "answer": gen_answer,
                                "object_id": rare_obj_id,
                                "class": rare_class
                            }
                            logging.info(f'Cateogry: {question_category}\nQuestion: {gen_question}\nAnswer: {gen_answer}')

                            generated_question_list.append(generated_question)

                            # save the generated question
                            with open(generated_questions_save_file, "w") as f:
                                json.dump(generated_question_list, f, indent=4)

                            generate_success = True
                        elif "Explanation: " in output:
                            gen_explanation = output.split("Explanation: ")[1].strip()

                            generated_question = {
                                "question_id": question_id,
                                "episode_history": scene_path.split('/')[-1],
                                "category": question_category.replace('_', ' '),
                                "question": "Not proper to generate a question",
                                "answer": gen_explanation,
                                "object_id": rare_obj_id,
                                "class": rare_class
                            }
                            logging.info(f'Cateogry: {question_category}\nCannot generate question\nExplanation: {gen_explanation}')

                            generated_question_list.append(generated_question)

                            # save the generated question
                            with open(generated_questions_save_file, "w") as f:
                                json.dump(generated_question_list, f, indent=4)

                        else:
                            logging.info(f"Invalid output for GPT: {output}")
                    else:
                        logging.info(f"Error in generating question for {rare_obj_id}-{rare_class}!")










