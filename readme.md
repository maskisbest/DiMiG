## Overview

This repo provides the training and evaluation code for universal **image** perturbations against VLP models.

## Setup

### Install dependencies

We recommend **Python 3.8+** and a PyTorch build matching your CUDA version. Example (pip):

```bash
pip install torch torchvision transformers ruamel.yaml pillow tqdm pandas
```

### Prepare datasets

Download the datasets and set `image_root` in the config files:

- [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/)
- [MSCOCO](https://cocodataset.org/#home)

The annotation JSONs are already provided in `data_annotation/`.

### Prepare checkpoints

Download finetuned checkpoints and place them under `checkpoint/<MODEL>/<dataset>.pth`:

- [ALBEF](https://github.com/salesforce/ALBEF)
- [TCL](https://github.com/uta-smile/TCL)
- [BLIP](https://github.com/salesforce/BLIP)
- [X-VLM](https://github.com/zengyan-97/X-VLM)

For CLIP models (ViT-B/16, RN101), weights are downloaded automatically via the cache in `hf_cache/`:

- [CLIP](https://huggingface.co/openai/clip-vit-base-patch16)

## Reproduce the paper

### 1) Build BadCLIP candidates

```bash
python build_badclip_candidates.py \
  --config configs/Retrieval_flickr_train.yaml \
  --source_model ALBEF \
  --checkpoint_root ./checkpoint \
  --output ./output/ALBEF/flickr30k/badclip_candidates.jsonl
```

### 2) Train the image perturbation generator

```bash
python img_train.py \
  --config configs/Retrieval_flickr_train.yaml \
  --source_model ALBEF \
  --source_ckpt ./checkpoint \
  --badclip_candidates ./output/ALBEF/flickr30k/badclip_candidates.jsonl
```

### 3) Evaluate on Image-Text Retrieval

```bash
python eval.py \
  --config configs/Retrieval_flickr_test.yaml \
  --source_model ALBEF \
  --load_dir ./output
```
