import argparse
import json
import os
import random

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from generator import Generator
from dataset import paired_dataset
from gen_img_attack import Gen_img_Attacker, ImageAttacker
from utils import load_model


def load_badclip_candidates(path):
    # Load candidate ids per image.
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            mapping[int(item["image_id"])] = {
                "boundary": int(item["boundary_text_id"]),
                "furthest": int(item["furthest_text_id"]),
                "random": int(item["random_text_id"]),
            }
    return mapping


# Train the image perturbation generator.
def train(model, tokenizer, data_loader, device, args, save_dir, config, badclip_map, all_texts):
    print("Start train (single GPU)")
    model.float()
    model.eval()

    image_G_input_dim = 3
    image_G_output_dim = 3
    image_num_filters = [64, 128, 256]

    if args.source_model in ["ALBEF", "TCL", "BLIP", "XVLM"]:
        context_dim = 256
    else:
        context_dim = 512

    image_netG = Generator(
        image_G_input_dim,
        image_num_filters,
        image_G_output_dim,
        num_heads=1,
        context_dim=context_dim,
    )

    if args.start_epoch > 0:
        load_dir = os.path.join(args.load_dir, f"{args.source_model}-[{args.pos_weights_raw}]", config["dataset"])
        image_netG.load_state_dict(
            torch.load(os.path.join(load_dir, "image-model-{}.pth".format(args.start_epoch - 1)), map_location=device)
        )
    image_netG = image_netG.to(device)

    if args.source_model in ["ALBEF", "TCL", "XVLM"]:
        images_normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
    elif args.source_model == "BLIP":
        images_normalize = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )
    elif args.source_model in ["ViT-B/16", "RN101"]:
        images_normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
    elif args.source_model == "11":
        images_normalize = transforms.Normalize(
            (0.44430269, 0.42129134, 0.38488099),
            (0.28511056, 0.27731498, 0.28582974),
        )
    else:
        images_normalize = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )

    img_attacker = ImageAttacker(
        image_netG,
        images_normalize,
        args.temperature,
        model,
        eps=args.eps / 255,
        device=device,
        lr=args.lr,
        alpha=args.alpha,
    )
    attacker = Gen_img_Attacker(model, img_attacker, tokenizer)

    epoch_logs = []

    # Main training loop.
    for epoch in range(args.start_epoch, args.epochs):
        running_total_loss = 0
        running_cross_loss = 0
        running_single_loss = 0
        epoch_total_loss = 0
        epoch_cross_loss = 0
        epoch_single_loss = 0
        batch_count = 0

        for batch_idx, (images, texts_group, images_ids, text_ids_groups) in enumerate(tqdm(data_loader)):
            txt2img = []
            texts = []
            txt_id = 0
            img2txt = []
            for i in range(len(texts_group)):
                texts += texts_group[i]
                txt2img += [i] * len(text_ids_groups[i])
                img2txt.append([])
                for _ in range(len(texts_group[i])):
                    img2txt[i].append(txt_id)
                    txt_id += 1

            images = images.to(device)

            pos_texts = []
            for img_id in images_ids:
                if img_id not in badclip_map:
                    raise KeyError(f"badclip_candidates missing image id {img_id}")
                cand = badclip_map[img_id]
                pos_texts.extend(
                    [
                        all_texts[cand["furthest"]],
                        all_texts[cand["boundary"]],
                        all_texts[cand["random"]],
                    ]
                )
            pos_inputs = tokenizer(
                pos_texts, padding="max_length", truncation=True, max_length=30, return_tensors="pt"
            ).to(device)
            pos_txts_output = model.inference_text(pos_inputs)
            pos_txt_supervisions = pos_txts_output["text_feat"].view(len(images_ids), 3, -1)
            pos_weights = torch.tensor(args.pos_weights, device=device).unsqueeze(0).expand(len(images_ids), -1)

            img_loss, img_single_loss, img_cross_loss, img_perturbation_embeding = attacker.attack(
                images,
                texts,
                img2txt,
                txt2img,
                pos_txt_supervisions,
                pos_weights,
                device=device,
                max_length=30,
            )

            running_total_loss += img_loss.item()
            running_single_loss += img_single_loss.item()
            running_cross_loss += img_cross_loss.item()
            epoch_total_loss += img_loss.item()
            epoch_single_loss += img_single_loss.item()
            epoch_cross_loss += img_cross_loss.item()
            batch_count += 1

            if (batch_idx + 1) % 10 == 0:
                print(
                    "Epoch: {}  Batch: {}/{}  img single loss: {:.5f}  image cross loss: {:.5f}  image total loss: {:.5f}".format(
                        epoch,
                        batch_idx + 1,
                        len(data_loader),
                        running_single_loss / 10,
                        running_cross_loss / 10,
                        running_total_loss / 10,
                    )
                )
                running_total_loss = 0
                running_single_loss = 0
                running_cross_loss = 0

            del img_loss, img_single_loss, img_cross_loss, img_perturbation_embeding
            torch.cuda.empty_cache()

        if batch_count > 0:
            print(
                "Epoch: {}  Avg img single loss: {:.5f}  image cross loss: {:.5f}  image total loss: {:.5f}".format(
                    epoch,
                    epoch_single_loss / batch_count,
                    epoch_cross_loss / batch_count,
                    epoch_total_loss / batch_count,
                )
            )
            # Save generator checkpoint.
            attacker.img_attacker.save_model("{}/image-model-{}.pth".format(save_dir, epoch))

            epoch_logs.append(
                {
                    "epoch": epoch,
                    "avg_total_loss": epoch_total_loss / batch_count,
                    "avg_single_loss": epoch_single_loss / batch_count,
                    "avg_cross_loss": epoch_cross_loss / batch_count,
                }
            )

    if epoch_logs:
        # Persist loss curves and optional plot.
        log_path = os.path.join(save_dir, "loss_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(epoch_logs, f, ensure_ascii=False, indent=2)
        print(f"Saved loss log to {log_path}")
        if plt is not None:
            epochs = [e["epoch"] for e in epoch_logs]
            total = [e["avg_total_loss"] for e in epoch_logs]
            single = [e["avg_single_loss"] for e in epoch_logs]
            cross = [e["avg_cross_loss"] for e in epoch_logs]
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, total, label="total_loss")
            plt.plot(epochs, single, label="single_loss")
            plt.plot(epochs, cross, label="cross_loss")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Training loss per epoch")
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(save_dir, "loss_curve.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Saved loss curve to {plot_path}")
        else:
            print("matplotlib not installed; skipped plotting loss curve.")
    torch.cuda.empty_cache()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/Retrieval_flickr_train.yaml")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--target_batch_size", default=4, type=int)
    parser.add_argument("--badclip_candidates", default="", type=str)
    parser.add_argument("--source_model", default="ALBEF", type=str)
    parser.add_argument("--source_text_encoder", default="bert-base-uncased", type=str)
    parser.add_argument("--source_ckpt", default="./checkpoint/", type=str)
    parser.add_argument("--eps", type=int, default=12)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--load_dir", default="", type=str, help="checkpoint root for resumes")
    parser.add_argument(
        "--pos_weights",
        type=str,
        default="1.0,0.5,0.1",
        help="positive sample weights [furthest,boundary,random]",
    )
    args = parser.parse_args()
    args.pos_weights_raw = args.pos_weights.replace(" ", "")

    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    save_dir = os.path.join(args.save_dir, f"{args.source_model}-[{args.pos_weights_raw}]", config["dataset"])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        cudnn.benchmark = True

    print("Creating Source Model")
    source_ckpt = os.path.join(args.source_ckpt, args.source_model, "{}.pth".format(config["dataset"]))
    model, ref_model, tokenizer = load_model(args.source_model, source_ckpt, args.source_text_encoder, config, device)
    model = model.to(device)
    ref_model = ref_model.to(device)

    print("Creating dataset")
    if args.source_model in ["ALBEF", "TCL", "BLIP", "XVLM"]:
        s_test_transform = transforms.Compose(
            [
                transforms.Resize((config["image_res"], config["image_res"]), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )
    else:
        n_px = model.visual.input_resolution if hasattr(model, "visual") else config.get("image_res", 224)
        s_test_transform = transforms.Compose(
            [
                transforms.Resize(n_px, interpolation=Image.BICUBIC),
                transforms.CenterCrop(n_px),
                transforms.ToTensor(),
            ]
        )

    train_dataset = paired_dataset(config["annotation_file"], s_test_transform, config["image_root"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
    )

    badclip_map = load_badclip_candidates(args.badclip_candidates)
    all_texts = train_dataset.text

    pos_weights = [float(x) for x in args.pos_weights.split(",") if x.strip()]
    if len(pos_weights) != 3:
        raise ValueError(f"pos_weights needs 3 values [furthest,boundary,random], got {pos_weights}")
    args.pos_weights = pos_weights

    train(model, tokenizer, train_loader, device, args, save_dir, config, badclip_map, all_texts)


if __name__ == "__main__":
    main()
