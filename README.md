
# Explore until Confident: Efficient Exploration for Embodied Question Answering

Allen Z. Ren, Jaden Clark, Anushri Dixit, Masha Itkina, Anirudha Majumdar, Dorsa Sadigh

Princeton University, Stanford University, Toyota Research Institute

[Project webpage](https://explore-eqa.github.io/)

## Installation
Set up the conda environment (Linux, Python 3.9):
```bash
conda env create -f environment.yml
conda activate explore-eqa
pip install -e .
```

Install the latest version of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) (headless with no Bullet physics) with:
```bash
conda install habitat-sim headless -c conda-forge -c aihabitat
```

<!-- Follow instructions at [https://github.com/TRI-ML/prismatic-vlms](https://github.com/TRI-ML/prismatic-vlms) to set up PrismaticVLM. -->
Set up [Prismatic VLM](https://github.com/TRI-ML/prismatic-vlms) with the submodule:
```bash
cd prismatic-vlms && pip install -e .
```

Download the **train** split (`hm3d-train-habitat-v0.2.tar`) of the HM3D dataset [here](https://github.com/matterport/habitat-matterport-3dresearch). You will be asked to request for access first.

(Optional) For running [CLIP](https://github.com/openai/CLIP)-based exploration:
```bash
cd CLIP && pip install -e .
```

## Dataset

We release the [HM-EQA](https://github.com/Stanford-ILIAD/explore-eqa/tree/master/data) dataset, which includes 500 questions about 267 scenes from the [HM-3D](https://aihabitat.org/datasets/hm3d-semantics/) dataset. They are available in [`data/`](https://github.com/Stanford-ILIAD/explore-eqa/tree/master/data).

## Usage
First specify `scene_data_path` in the config files with the path to the downloaded HM3D train split, and specify `hf_token` to be your Hugging Face user access token.Running the script below for the first time will download the VLM model, which assumes access to a GPU with sufficient VRAM for the chosen VLM.

Run our method (VLM-semantic exploration) in Habitat-Sim:
```bash
python run_vlm_exp.py -cf cfg/vlm_exp.yaml
```

Run CLIP-based exploration in Habitat-Sim:
```bash
python run_clip_exp.py -cf cfg/clip_exp.yaml
```

Load a scene (with the question from our dataset) in Habitat-Sim:
```bash
python test_scene.py -cf cfg/test_scene.yaml
```

## Scripts

We also share a few scripts that might be helpful:
- [script/sample_views_from_scene.py](https://github.com/Stanford-ILIAD/explore-eqa/tree/master/script/sample_views_from_scene.py): for sampling random views in a scene in Habitat-Sim. We used such images for generating EQA questions with GPT4-V.
- [script/get_floor_height.py](https://github.com/Stanford-ILIAD/explore-eqa/tree/master/script/get_floor_height.py): for getting the height of the floors in each scene of the HM-3D dataset, which is not available from the original dataset.
- [script/get_questions_gpt4v.py](https://github.com/Stanford-ILIAD/explore-eqa/tree/master/script/get_questions_gpt4v.py): for generating EQA questions with GPT4-V with random views of the scene and fewshot examples.
- [script/sample_init_pose.py](https://github.com/Stanford-ILIAD/explore-eqa/tree/master/script/sample_init_pose.py): for sampling valid initial poses of the robot in each scene.

## Acknowledgement

The CLIP-based exploration uses the CLIP multi-scale relevancy extractor from [Semantic Abstraction](https://github.com/real-stanford/semantic-abstraction).