import os, logging, io, requests, base64, pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Get the list of scenes and floors
setup_dir = "questions"
floor_view_parent_dir = os.path.join(setup_dir, "?")
fewshot_floor_view_parent_dir = os.path.join(setup_dir, "?")
# scene_floor_names = sorted(os.listdir(floor_view_parent_dir))

# Get views info
views_data_path = os.path.join(floor_view_parent_dir, "?.pkl")
with open(views_data_path, "rb") as f:
    views_data = pickle.load(f)

# log
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(os.path.join(setup_dir, "?.txt"), mode="w+"),
        logging.StreamHandler(),
    ],
)

# cfg
api_key = "?"
num_view = 5
topdown_dir = "?"
dataset_name = "?"
vlm_T = 0.2
sys_prompt = "You are a helpful assistant."
prompt_bg = (
    "You will be shown some random views inside a house."
    " You will come up with a simple question"
    " based on the views, that the household owner may ask the robot."
    " Make sure the question has four options and a definitive answer."
    " Try to be creative in the question and make it sound like interesting scenarios"
    " when the household owner needs help."
    "\nNote:"
    " \n(1) The question is for a robot in the 3D scene, so do not refer to the"
    " image in"
    " the question."
    " \n(2) The views shown do not cover the full scene and there might be"
    " information of the scene missing, and thus do not refer to 'first/second room'"
    " or 'the room' or 'the table' or 'the view' or 'the area' in the question."
    " \nBelow are some examples of the views and the corresponding question."
    " \n(3) Do not ask about, for example, if a door is locked or not, or if the fan"
    " is on,"
    " since it is hard to tell from static images."
    " \nIn terms of types of questions to ask:"
    " \n(1) Focus on locations of the objects, especially those small, or misplaced, or"
    " those"
    " that the household owner possibly recently moved."
    " \n(2) Sometimes ask counting questions, such as 'how many chairs in...'."
    " \n(3) You can ask Yes/No type of questions (but just ask sometimes), such as 'is"
    " the fire distinguisher near the staircase?'"
    " but avoid multiple choices like 'Can't determine' or 'I am not sure.'"
    " \n(4) Do not ask questions that the household owner should know, like the color"
    " of the sofa, or type of plants."
)

# Run all
responses_all = {}
# for cnt, scene_floor_name in tqdm(enumerate(scene_floor_names)):
for cnt in range(len(views_data)):
    scene_name = views_data[cnt]["scene_name"]
    floor_ind = views_data[cnt]["floor"]
    scene_floor_name = f"{scene_name}_{floor_ind}"
    # scene_name, floor_ind = scene_floor_name.split("_")
    logging.info("=====================================")
    logging.info(f"Scene name: {scene_name}")
    logging.info(f"Floor: {floor_ind}")

    # Get view pts
    # pts_xy_all = [view_data[f'step_{step+1}']['pts'][[0,2]] for step in range(num_view)]

    # Initialize with the background prompt (not the system prompt, which is short)
    content = [{"type": "text", "text": prompt_bg}]

    # Add few-shot examples
    fewshot_data = {
        # '00750-E1NrAhMoqvB_0':
        #     ' Can you figure out where I left the red blanket?'
        #     ' A) on the bed B) on the sofa C) on the living room floor'
        #     ' D) in the bathroom. Answer: A) on the bed',
        "00475-g7hUFVNac26_1": (
            " Can you figure out where I left the small silver trash can?"
            " A) by the bedroom door B) in the kitchen C) in the bathroom"
            " D) by the living room sofa. Answer: A) by the bedroom door"
        ),
        "00034-6imZUJGRUq4_0": (
            " I forgot where I hung the clock in the basement. Can"
            " you tell me where is it? A) above the bed B) on the wall"
            " C) on the pillar D) next to TV. Answer: C) on the pillar"
        ),
        "00238-j6fHrce9pHR_0": (
            " Could you go and check how many wooden chairs I left in the garage?"
            " A) one B) two C) three D) four. Answer: D) four"
        ),
        # '00267-gQ3xxshDiCz_0':
        #     ' Can you figure out how many pillows that I left on the sofa?'
        #     ' A) one B) two C) three D) four. Answer: D) four'
    }
    for fewshot_ind, (fewshot_scene_floor, fewshot_prompt) in enumerate(
        fewshot_data.items()
    ):
        view_base_64_image_all = []
        for i in range(num_view):
            view_image_path = os.path.join(
                fewshot_floor_view_parent_dir, fewshot_scene_floor, f"{i}.png"
            )
            view_base_64_image_all.append(encode_image(view_image_path))
        content += [{"type": "text", "text": f"Example {fewshot_ind+1}:"}]
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
        content += [{"type": "text", "text": "Question:" + fewshot_prompt}]

    # Add current scene
    view_base_64_image_all = []
    for i in range(num_view):
        view_image_path = os.path.join(
            floor_view_parent_dir, scene_floor_name, f"{i}.png"
        )
        view_base_64_image_all.append(encode_image(view_image_path))
    content += [
        {
            "type": "text",
            "text": (
                "Now can you generate three questions and their answers"
                " based on the views below:"
            ),
        }
    ]
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

    # Debug info
    # print("System prompt:\n", sys_prompt)
    # print([c['text'] for c in content if isinstance(c, dict) and 'text' in c])
    # print(len(content))

    # Call
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Organization": "?",
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": content},
        ],
        "max_tokens": 500,
        "seed": 42,
        "temperature": vlm_T,
    }

    # Process response
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    ).json()
    try:
        output = response["choices"][0]["message"]["content"]
        logging.info(output)
    except:
        logging.info(f"Error: {response}")
    responses_all[scene_floor_name] = response

    # Save periodically
    if cnt % 100 == 0:
        with open(os.path.join(setup_dir, dataset_name + f"_{cnt}.pkl"), "wb") as f:
            pickle.dump(responses_all, f)

# Save
with open(os.path.join(setup_dir, dataset_name + ".pkl"), "wb") as f:
    pickle.dump(responses_all, f)
