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


class ExploreDataset(Dataset):
    
    def __init__(self, src_path, 
                 tokenizer,
                 max_length,
                 scene_token = SCENE_TOKEN,
                 # frontier_token = FRONTIER_TOKEN,
                 select_token = SELECT_TOKEN):
        self.scene_dir = os.path.join(src_path,'scene_features')
        self.explore_dir = os.path.join(src_path,'exploration_data')
        self.tokenizer = tokenizer
        self.scene_token = scene_token
        self.scene_token_id = self.tokenizer(self.scene_token).input_ids[-1]
        # self.frontier_token = frontier_token
        # self.frontier_token_id = self.tokenizer.convert_tokens_to_ids(self.frontier_token)
        # self.select_token = select_token
        # self.select_token_id = self.tokenizer(select_token).input_ids[-1]
        self.max_length = max_length
        self.data = self.load_data()

    def load_data(self):
        # load scene feature into dict
        self.scenes = {}
        for scene in os.listdir(self.scene_dir):
            self.scenes[scene] = {}
            scene_fold = os.path.join(self.scene_dir, scene)
            # need to confirm: if object in different scene should have different features
            for object_f in os.listdir(scene_fold):
                object_id = object_f[:-3]
                object_feature  = torch.load(os.path.join(scene_fold, object_f),
                                             map_location = 'cpu')
                self.scenes[scene][object_id] = object_feature
            
        # load episode data: metadata is managed with self.episodes
        self.episodes= []
        data = []
        for i, episode in enumerate(os.listdir(self.explore_dir)):
            epi_path = os.path.join(self.explore_dir,episode)
            # load metadata
            with open(os.path.join(epi_path,'metadata.json'),'r') as f:
                metadata = json.load(f)
            self.episodes.append(metadata)
            
            # load step data
            steps_data = []
            for step in range(metadata["episode_length"]):
                with open(os.path.join(epi_path,f'{pad_zero(str(step),4)}.json')) as f:
                    stepdata = json.load(f)
                # link each step to its episode
                stepdata['episode_id'] = i
                # add paths for frontiers
                frontier_features = []
                stepdata['frontier_features'] = {}
                for frontier in stepdata["frontiers"]:
                    # placeholder for loading frontier feature
                    rgb_id = frontier['rgb_id']
                    frontier_folder = os.path.join(epi_path,'frontier_rbg')
                    # load frontier feature
                    # feature = torch.load(os.path.join(frontier_folder, rgb_id.replace(".png", ".pt")),
                    #                         map_location = 'cpu')
                    feature = os.path.join(frontier_folder, rgb_id.replace(".png", ".pt"))
                    # feature = torch.zeros(1024)
                    stepdata['frontier_features'][rgb_id] = feature
                    #front['rgb_id'] = os.path.join(epi_path,'frontier_rgb',front['rgb_id'])
                # remove frontier info, can be removed in case other features needed
                # del stepdata['frontiers']
                steps_data.append(stepdata)
            data.extend(steps_data)
        
        # link steps to episodes, which can be used for dataset split
        self.episode2step = defaultdict(list)
        for i in range(len(data)):
            self.episode2step[data[i]['episode_id']].append(i)
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        try:
            # load a whole episode and each step within it
            step = self.data[idx]
            episode = self.episodes[step['episode_id']]
            scene = self.scenes[episode['scene']]
            
            text =  f"Question: {episode['question']}\n" +\
                    f"Select the frontier/object that would help finding the answer of the question.\n"
            
            # replace scene graph in each steps with scene feature
            prediction = np.array(step['prediction'])
            object_features = []
            remove_indices = []
            text += "These are the objects already in our scene graph:\n"
            for i, sid in enumerate(step["scene_graph"]):
                if str(sid) not in scene.keys():
                    remove_indices.append(i)
                else:
                    object_features.append(scene[str(sid)])
                    # TODO: add object class
                    # text += f"object {i}: <scene> {step['scene_graph'][i]['class']}\n"
                    text += f"object_{i} <scene> "
            text += "/\n"

            prediction = np.delete(prediction, remove_indices)
            prediction = torch.tensor(prediction)
            # find the index of 1.0 in prediction
            prediction_index = np.where(prediction == 1.0)[0][0]
            if prediction_index < len(object_features):
                answer = f"object_{prediction_index}"
            else:
                answer = f"frontier_{prediction_index - len(object_features)}"
            
            # object_features = [scene[str(sid)] for sid in step["scene_graph"]
            #                     if str(sid) in scene.keys()]
            if len(object_features) == 0:
                # construct zero scene feature if all objects are missed
                object_features = torch.zeros((1,1024))
            else:
                object_features = torch.stack(object_features, dim = 0)

            text += "Below are all the frontiers that we can explore:\n"
            frontier_features = []
            for i, frontier in enumerate(step['frontiers']):
                frontier_features.append(
                    torch.load(step['frontier_features'][frontier['rgb_id']], map_location='cpu')
                )
                text += f"frontier_{i} <scene> "
            text += "/\n"
            frontier_features = torch.cat(frontier_features, dim = 0)

            text += "Answer: "
            text += answer + self.tokenizer.eos_token
            # test
            
            scene_feature = torch.cat([object_features, frontier_features], dim = 0)

            # if len(scene_feature) > 70:
            #     return self.__getitem__(idx-1)
                
            step["scene_feature"] = scene_feature
            # remove scene graph id --- remove this if we need to keep id
            
            # make sure all things are included
            assert self.max_length > len(text)
            assert self.max_length > len(scene_feature) # make sure that scene feature is never truncated

            
            text = self.tokenizer(text, return_tensors = "pt",
                                max_length = self.max_length,
                                truncation = True,
                                padding = 'max_length')
            input_ids = text["input_ids"]
            length = torch.nonzero(input_ids).shape[0]
            
            attention_mask = text["attention_mask"]

            # print(self.tokenizer.decode(input_ids[input_ids != self.tokenizer.pad_token_id]))
            
            # only pick the index of the first occured token
            scene_insert_loc = (input_ids == self.scene_token_id).nonzero()[:,1].reshape(-1)
            # frontier_insert_loc = (input_ids == self.frontier_token_id).nonzero()[:1,1].reshape(-1).tolist()
                        
            batch =  EasyDict(
                text = text,
                input_ids = input_ids,
                length = length,
                scene_length = len(scene_feature),
                attention_mask = attention_mask,
                scene_feature = scene_feature,
                scene_insert_loc = scene_insert_loc,
                # frontier_feature = step['frontier_features'],
                # frontier_insert_loc = frontier_insert_loc,
            )
        except:
            # print ("cannot find feature %d"%idx)
            return self.__getitem__(idx-1)
        # except:
        #     print(step['episode_id'])
        #     print(idx)
        #     print(step['scene_graph'])
    
    def collate_wrapper(self, batch):
        # because sos token is added, the max_length should be +1?
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
            # frontier_feature=frontier_feature,
            # frontier_insert_loc = list(chain.from_iterable([[[batch_idx, x] for x in b.frontier_insert_loc] for batch_idx, b in enumerate(batch)])),
            # scene_length = torch.tensor([b.scene_feature.shape[0] for b in batch]),
            # frontier_length = torch.tensor([b.frontier_feature.shape[0] for b in batch])
            max_scene_length = torch.tensor([b.scene_feature.shape[0] for b in batch])
        )