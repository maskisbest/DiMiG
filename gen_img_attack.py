import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms


class Gen_img_Attacker:

    def __init__(self, model, img_attacker, tokenizer):
        self.model = model
        self.img_attacker = img_attacker
        self.tokenizer = tokenizer

    def attack(
        self,
        imgs,
        txts,
        img2txt,
        txt2img,
        pos_txt_supervision,
        pos_weights,
        device="cpu",
        max_length=30,
        **kwargs,
    ):
        # Extract clean features and run one attack step.
        with torch.no_grad():
            imgs_outputs = self.model.inference_image(self.img_attacker.normalization(imgs))
            img_supervisions = imgs_outputs["image_feat"]
            txts_input = self.tokenizer(
                txts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output["text_feat"]

        img_loss, img_single_loss, img_cross_loss, img_perturbation_embeding = (
            self.img_attacker.txt_guided_attack(
                self.model,
                imgs,
                img2txt,
                txt2img,
                device,
                txt_embeds=txt_supervisions,
                pos_txt_embeds=pos_txt_supervision,
                pos_weights=pos_weights,
            )
        )

        return img_loss, img_single_loss, img_cross_loss, img_perturbation_embeding


class ImageAttacker:
    def __init__(self, netG, normalization, temperature, model, eps, device="cuda", lr=2e-4, alpha=0.1):
        self.normalization = normalization
        self.eps = eps
        self.generator = netG.to(device) if netG is not None else None
        if self.generator is not None:
            self.optimG = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.temperature = temperature
        self.alpha = alpha
        self.model = model

    def get_generator(self):
        return self.generator

    def save_model(self, path):
        torch.save(self.generator.state_dict(), path)

    # Contrastive loss with weighted positive targets.
    def loss_func(self, adv_imgs_embeds, imgs_embeds, txts_embeds, txt2img, pos_txt_embeds, pos_weights, temperature):
        device = adv_imgs_embeds.device

        sim_pos = (adv_imgs_embeds.unsqueeze(1) * pos_txt_embeds).sum(dim=-1)
        sim_pos = torch.exp(sim_pos / temperature)
        loss_target = (sim_pos * pos_weights).sum(dim=1).mean()

        it_sim_matrix = torch.exp((adv_imgs_embeds @ txts_embeds.T) / temperature)
        it_labels = torch.zeros(it_sim_matrix.shape, device=device)
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        loss_untarget = (it_sim_matrix * it_labels).sum(-1).mean()

        eps = 1e-8
        loss = torch.log(loss_untarget / (loss_untarget + loss_target + eps))

        return loss

    def txt_guided_attack(
        self,
        model,
        imgs,
        img2txt,
        txt2img,
        device,
        txt_embeds=None,
        pos_txt_embeds=None,
        pos_weights=None,
    ):
        # Optimize the generator to craft image perturbations.
        model.eval()
        b, _, _, _ = imgs.shape

        imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        imgs = torch.clamp(imgs, 0.0, 1.0)

        self.generator.train()
        self.optimG.zero_grad()
        for p in model.parameters():
            p.requires_grad = True

        text_cond = []
        for i in range(len(img2txt)):
            text_cond.append(torch.mean(txt_embeds[img2txt[i]], dim=0))
        text_cond = torch.stack(text_cond, dim=0)

        with torch.enable_grad():
            x = Variable(imgs.to(device))
            img_noise = self.generator(imgs, text_cond)
            model_name = model.module.__class__.__name__ if hasattr(model, "module") else model.__class__.__name__
            if model_name in ["ViT-B/16", "RN101"]:
                img_noise = F.interpolate(img_noise, size=(224, 224), mode="bilinear")
            img_noise = torch.clamp(img_noise, -self.eps, self.eps)

            adv_imgs = x + img_noise.expand(imgs.size())

            if self.normalization is not None:
                adv_imgs_output = model.inference_image(self.normalization(adv_imgs))
                imgs_output = model.inference_image(self.normalization(x))
            else:
                adv_imgs_output = model.inference_image(adv_imgs)
                imgs_output = model.inference_image(x)

            adv_imgs_embeds = adv_imgs_output["image_feat"]
            imgs_embeds = imgs_output["image_feat"]
            criterion_MSE = torch.nn.MSELoss(reduce=True, size_average=False)
            loss_MSE = criterion_MSE(adv_imgs_embeds, imgs_embeds)

            loss_infoNCE = self.loss_func(
                adv_imgs_embeds[0:b],
                imgs_embeds[0:b],
                txt_embeds,
                txt2img,
                pos_txt_embeds,
                pos_weights,
                self.temperature,
            )

            loss = loss_infoNCE - self.alpha * loss_MSE
        loss.backward()
        self.optimG.step()

        return loss, loss_infoNCE, loss_MSE, img_noise

    def get_scaled_imgs(self, imgs, scales=None, device="cuda"):
        # Multi-scale augmentation helper.
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])
        reverse_transform = transforms.Resize(ori_shape, interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio * ori_shape[0]), int(ratio * ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape, interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)

            reversed_imgs = reverse_transform(scaled_imgs)
            result.append(reversed_imgs)

        return torch.cat([imgs] + result, 0)
