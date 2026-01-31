import argparse
import contextlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
import torch
from tqdm import tqdm
from torchvision import transforms

from dataset import paired_dataset
from utils import load_model


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def prepare_text_inputs(tokenizer, texts, device, max_length=30):
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    class TextInput:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

        def to(self, dev):
            for k, v in self.__dict__.items():
                if torch.is_tensor(v):
                    setattr(self, k, v.to(dev))
            return self

        def __getitem__(self, key):
            return getattr(self, key)

        def __contains__(self, key):
            return hasattr(self, key)

    return TextInput(inputs).to(device)


def build_transforms(model_name, config, model=None):
    if model_name == "BLIP":
        return transforms.Compose(
            [
                transforms.Resize(
                    (config["image_res"], config["image_res"]),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # BLIP uses ImageNet stats
            ]
        )
    if model_name in ["ALBEF", "TCL", "XVLM"]:
        return transforms.Compose(
            [
                transforms.Resize(
                    (config["image_res"], config["image_res"]),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )
    n_px = model.visual.input_resolution if model_name in ["ViT-B/16", "RN101"] else 224
    return transforms.Compose(
        [
            transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
        ]
    )


def compute_text_embeds(model, tokenizer, texts, device, batch_size=256, max_length=30, dtype=None):
    use_dtype = dtype or (torch.float16 if device.type == "cuda" else torch.float32)
    out = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(chunks(texts, batch_size), desc="text embeds"):
            inputs = prepare_text_inputs(tokenizer, batch, device, max_length=max_length)
            ctx = torch.autocast(device_type=device.type, dtype=use_dtype) if device.type == "cuda" else contextlib.nullcontext()
            with ctx:
                features = model.inference_text(inputs)["text_feat"]
                features = features / features.norm(dim=-1, keepdim=True)
            out.append(features.cpu())
    return torch.cat(out, dim=0)


def compute_image_embeds(model, loader, device, dtype=None):
    use_dtype = dtype or (torch.float16 if device.type == "cuda" else torch.float32)
    rows = []
    ids = []
    model.eval()
    with torch.no_grad():
        for images, _, img_ids, _ in tqdm(loader, desc="image embeds"):
            images = images.to(device)
            ctx = torch.autocast(device_type=device.type, dtype=use_dtype) if device.type == "cuda" else contextlib.nullcontext()
            with ctx:
                feats = model.inference_image(images)["image_feat"]
                feats = feats / feats.norm(dim=-1, keepdim=True)
            rows.append(feats.cpu())
            ids.extend(img_ids)
    embeds = torch.zeros(len(loader.dataset), rows[0].shape[-1])
    offset = 0
    for block in rows:
        count = block.shape[0]
        embeds[ids[offset : offset + count]] = block
        offset += count
    return embeds


def _dir_cos(img_vec, txt_vec_a, txt_vec_b, eps=1e-6):
    """
    Directional cosine between (img->a) and (img->b).
    """
    va = txt_vec_a - img_vec
    vb = txt_vec_b - img_vec
    va = va / (va.norm(dim=-1, keepdim=True).clamp(min=eps))
    vb = vb / (vb.norm(dim=-1, keepdim=True).clamp(min=eps))
    return (va * vb).sum(dim=-1)


def select_candidates(image_embeds, text_embeds, img2txt, topk=64, seed=42, device=torch.device("cpu")):
    rng = random.Random(seed)
    num_imgs = image_embeds.size(0)
    num_txts = text_embeds.size(0)
    all_txt_ids = set(range(num_txts))

    text_gpu = text_embeds.to(device)
    results = []
    boundary_dir_fallback = 0
    chunk = 256
    for start in tqdm(range(0, num_imgs, chunk), desc="select candidates"):
        end = min(start + chunk, num_imgs)
        img_chunk = image_embeds[start:end].to(device)
        sim = img_chunk @ text_gpu.T
        furthest_sim = sim.clone()
        for row, img_id in enumerate(range(start, end)):
            banned = img2txt[img_id]
            if banned:
                furthest_sim[row, banned] = float("inf")

        # 最远样本：与图像最不相似（排除配对）
        furthest_idx = torch.argmin(furthest_sim, dim=1)

        for row, img_id in enumerate(range(start, end)):
            banned = set(img2txt[img_id])
            far_idx = furthest_idx[row].item()
            if not math.isfinite(furthest_sim[row, far_idx].item()) or far_idx in banned:
                far_idx = rng.choice(list(all_txt_ids - banned))

            # 计算 topk 最相近候选（排除配对），基于方向一致性筛边界样本
            boundary_sim = sim[row : row + 1].clone()
            if banned:
                boundary_sim[0, list(banned)] = -float("inf")
            _, top_idx = torch.topk(boundary_sim, k=min(topk, boundary_sim.size(1)), dim=1)

            boundary_choice = None
            img_vec = image_embeds[img_id].to(device)
            far_vec = text_embeds[far_idx].to(device)
            cos_lower, cos_upper = 1e-3, math.cos(math.radians(15)) - 1e-6  # (0, ~0.97)
            for idx in top_idx[0].tolist():
                if idx in banned:
                    continue
                cand_vec = text_embeds[idx].to(device)
                cos_val = _dir_cos(img_vec, cand_vec, far_vec)
                if cos_lower < cos_val < cos_upper:
                    boundary_choice = idx
                    break
            if boundary_choice is None:
                boundary_dir_fallback += 1
                # 回退到最相似的非配对
                for idx in top_idx[0].tolist():
                    if idx not in banned:
                        boundary_choice = idx
                        break
            if boundary_choice is None:
                boundary_choice = rng.choice(list(all_txt_ids - banned))

            pool = list(all_txt_ids - banned - {boundary_choice, far_idx})
            random_choice = rng.choice(pool) if pool else rng.choice(list(all_txt_ids - banned))

            results.append(
                {
                    "image_id": img_id,
                    "boundary_text_id": boundary_choice,
                    "furthest_text_id": far_idx,
                    "random_text_id": random_choice,
                }
            )
    return results, {"boundary_dir_fallback": boundary_dir_fallback, "total_images": num_imgs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/Retrieval_coco_train.yaml")
    parser.add_argument("--source_model", default="BLIP", type=str)
    parser.add_argument("--source_text_encoder", default="bert-base-uncased", type=str)
    parser.add_argument("--checkpoint_root", default="./checkpoint", type=str)
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="输出候选文件路径，默认 ./output/<source_model>/badclip_candidates.jsonl",
    )
    parser.add_argument("--text_batch_size", default=256, type=int)
    parser.add_argument("--image_batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int, help="DataLoader workers; set 0 to avoid shm issues")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pin_memory for DataLoader")
    parser.add_argument("--topk", default=64, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_length", default=30, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    model_ckpt = os.path.join(args.checkpoint_root, args.source_model, f"{config['dataset']}.pth")

    # 自动根据 source_model 设定默认输出目录
    if args.output is None:
        args.output = os.path.join("./output", args.source_model, config["dataset"], "badclip_candidates.jsonl")

    print("loading model...")
    model, _, tokenizer = load_model(args.source_model, model_ckpt, args.source_text_encoder, config, device)
    model = model.to(device)

    print("building dataset...")
    transform = build_transforms(args.source_model, config, model)
    dataset = paired_dataset(config["annotation_file"], transform, config["image_root"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.image_batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=args.pin_memory,
        shuffle=False,
    )

    print("computing text embeddings...")
    text_embeds = compute_text_embeds(
        model, tokenizer, dataset.text, device, batch_size=args.text_batch_size, max_length=args.max_length
    )
    print("computing image embeddings...")
    image_embeds = compute_image_embeds(model, loader, device)

    print("selecting candidates...")
    candidates, stats = select_candidates(
        image_embeds, text_embeds, dataset.img2txt, topk=args.topk, seed=args.seed, device=device
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in candidates:
            item["image_path"] = dataset.image[item["image_id"]]
            item["boundary_text"] = dataset.text[item["boundary_text_id"]]
            item["furthest_text"] = dataset.text[item["furthest_text_id"]]
            item["random_text"] = dataset.text[item["random_text_id"]]
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 辅助文件放在同一路径前缀下
    text_map_path = out_path.with_suffix(".texts.json")
    with text_map_path.open("w", encoding="utf-8") as f:
        json.dump({"texts": dataset.text}, f, ensure_ascii=False)

    sim_pos_path = out_path.with_suffix(".sim_pos_stats.jsonl")
    # 如果后续需要保存相似度统计，可在此写入；暂占位空文件便于统一路径
    with sim_pos_path.open("w", encoding="utf-8") as f:
        pass

    print(f"saved candidates to {out_path}")
    print(f"saved text list to {text_map_path}")
    print(f"saved sim_pos stats placeholder to {sim_pos_path}")
    print(f"boundary direction fallback: {stats['boundary_dir_fallback']} / {stats['total_images']}")


if __name__ == "__main__":
    main()
