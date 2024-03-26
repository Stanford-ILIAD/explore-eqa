import requests
import torch
import time

from PIL import Image
from pathlib import Path

from prismatic import load
from prismatic import available_model_names, available_models, get_model_description
from pprint import pprint

# # For gated LMs like Llama-2, make sure to request official access, and generate an access token
hf_token = ""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub)
model_id = "prism-dinosiglip+7b"
vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
# image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image = Image.open("results/clip_exp/0/0.png").convert("RGB")
# user_prompt = "What is going on in this image?"
question = "Is the lamp in the bedroom turned on?"
user_prompt = f"You are in a home environemnt. Consider the question: '{question}', and you will explore the environment for answering it.\nIs there any direction shown in the image worth exploring? Answer with Yes or No."

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate!
s1 = time.time()
# generated_text = vlm.generate(
#     image,
#     prompt_text,
#     do_sample=True,
#     temperature=0.4,
#     max_new_tokens=512,
#     min_length=1,
# )
losses = vlm.get_loss(
    image,
    prompt_text,
    return_string_probabilities=["Yes", "No"],
)[0]
# print("Generated text:", generated_text)
print("Loss:", losses)
print("Time used:", time.time() - s1)

# # List all Pretrained VLMs (by HF Hub IDs)
# pprint(available_models())

# # List all Pretrained VLMs + Descriptions (by explicit labels / names from paper figures)
# pprint(available_model_names())

# # Print and return a targeted description of a model (by name or ID)
# #   =>> See `prismatic/models/registry.py` for explicit schema
# description = get_model_description("Prism-DINOSigLIP 13B (Controlled)")
