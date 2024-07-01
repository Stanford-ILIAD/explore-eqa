from torch.utils.data.distributed import DistributedSampler
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from easydict import EasyDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader,Dataset,Subset
from itertools import chain
import random
import numpy as np

SCENE_TOKEN = "<scene>"
# FRONTIER_TOKEN = "<frontier>"
SELECT_TOKEN = "<select>"
SCENE_TOKEN = "<scene>"
VISUAL_TOKEN = "<visual>"
TACTILE_TOKEN = "<temperature>"
SOUND_TOKEN = "<sound>"
# TEMP_TOKEN = "<temperature>"
GET_VISUAL_TOKEN = "<observe>"
GET_TACTILE_TOKEN = "<touch>"
GET_SOUND_TOKEN = "<tap>"
SELECT_TOKEN = "<select>"

# because sos token is added, the max_length should be +1?

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, False)
    del checkpoint

def collate_wrapper(batch):
    max_length = max(b.length for b in batch) + 1
    max_scene_length = max(b.scene_feature.shape[0] for b in batch)
    # max_frontier_length = max(b.frontier_feature.shape[0] for b in batch)
    
    scene_feature = torch.zeros((len(batch), max_scene_length, 1024))
    scene_insert_loc = torch.zeros((len(batch), max_scene_length))
    
    for (j,b) in enumerate(batch):
        scene_feature[j, :b.scene_feature.shape[0]] = b.scene_feature
        # frontier_feature[j, :b.frontier_feature.shape[0]] = b.frontier_feature
        scene_insert_loc[j, :b.scene_insert_loc.shape[0]] = b.scene_insert_loc
    
    return EasyDict(
        input_ids=torch.cat([b.input_ids for b in batch])[...,:max_length],
        attention_mask=torch.cat([b.attention_mask for b in batch])[...,:max_length],
        scene_feature=scene_feature,
        scene_insert_loc=scene_insert_loc.to(torch.long),
        scene_length = torch.tensor([b.scene_length for b in batch]),
        max_scene_length = torch.tensor([b.scene_feature.shape[0] for b in batch])
    )

def load_scene_features(scene_dir, scene_id):
    scene = {}
    scene_fold = os.path.join(scene_dir, scene_id)
    for object_f in os.listdir(scene_fold):
        try:
            object_id = object_f[:-3]
            object_feature  = torch.load(os.path.join(scene_fold, object_f),
                                        map_location = 'cpu')
            scene[object_id] = object_feature
        except:
            continue
    return scene

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

def prepare_step_dict(step_dict):
    pass

def encode(model, image_processor, img):
    img = image_processor.preprocess(img, return_tensors='pt')['pixel_values']
    img = torch.cat([img], dim=0).half().cuda()
    img = model.encode_images(img)
    return img

# TODO: add prefiltering and parts from eval_dataset here

def get_item(tokenizer, step_dict):
    # load a whole episode and each step within it
    step = step_dict
    # episode = step_dict['episode']
    scene = step['scene']
    scene_feature_map = step['scene_feature_map']
    obj_map = step['obj_map']
    text =  f"Question: {step['question']}\n" 

    if step.get("use_egocentric_views") is True:
        text += "Followings are the egocentric views:\n "
        for i in range(len(step["egocentric_view_features"])):
            text += f"<scene> "
        egocentric_features = step["egocentric_view_features"]
        text += "/\n"
    
    text += f"Select the frontier/object that would help finding the answer of the question.\n"

    if step.get("use_action_memory") is True:
        text += f"Here is your selection in the previous step:\n "
        if step["memory_feature"] is None:
            text += f"No selection in the previous step. "
        else:
            text += f"<scene> "
        memory_feature = step["memory_feature"]
        text += "/\n"
    
    # replace scene graph in each steps with scene feature
    object_features = []
    remove_indices = []
    object_index = 0
    for i, sid in enumerate(step["scene_graph"]):
        if str(sid) not in scene_feature_map.keys() or sid not in obj_map.keys():
            remove_indices.append(i)
        else:
            object_feature = scene_feature_map[str(sid)]
            object_features.append(object_feature)
            class_name = obj_map[sid]
            text += f"object {object_index} {class_name} <scene> "
            object_index += 1
    if object_index == 0:
        text += f"No object available "
    text += "/\n"
    
    print("length object features", len(object_features))

    if len(object_features) == 0:
        # construct zero scene feature if all objects are missed
        object_features = None
    else:
        object_features = torch.stack(object_features, dim = 0)

    text += "Below are all the frontiers that we can explore:\n"
    if len(step['frontiers']) > 0:
        for i, frontier in enumerate(step['frontiers']):
            text += f"frontier {i} <scene> "
    else:
        text += f"No frontier available "
    text += "/\n"
    frontier_features = step["frontier_features"]
    text += "Answer: "
    
    if object_features is not None and frontier_features is not None:
        scene_feature = torch.cat([object_features, frontier_features], dim=0)
    elif object_features is not None:
        scene_feature = object_features
    else:
        scene_feature = frontier_features

    if step.get("use_egocentric_views") is not None:
        scene_feature = torch.cat([egocentric_features, scene_feature], dim=0)

    if step.get("memory_feature") is not None:
        scene_feature = torch.cat([memory_feature, scene_feature], dim=0)
   
    step["scene_feature"] = scene_feature
    # remove scene graph id --- remove this if we need to keep id
    print(text)
    
    text = tokenizer(text, return_tensors = "pt",
                        max_length = 1024,
                        truncation = True,
                        padding = 'max_length')
    input_ids = text["input_ids"]
    length = torch.nonzero(input_ids).shape[0]
    
    attention_mask = text["attention_mask"]
    
    # only pick the index of the first occured token
    scene_token_id = tokenizer(SCENE_TOKEN).input_ids[-1]
    scene_insert_loc = (input_ids == scene_token_id).nonzero()[:,1].reshape(-1)
                
    batch = [
        EasyDict(
            text = text,
            input_ids = input_ids,
            length = length,
            scene_length = len(scene_feature),
            attention_mask = attention_mask,
            scene_feature = scene_feature,
            scene_insert_loc = scene_insert_loc,
        )
    ]
    return collate_wrapper(batch)
