import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import utils
from utils import load_model
from dataset import paired_dataset
import torch.nn.functional as F

from generator import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/Retrieval_flickr_test.yaml')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--text_encoder', default='bert-base-uncased', type=str)
parser.add_argument('--source_model', default='ALBEF', type=str)
parser.add_argument('--checkpoint', default='./checkpoint', type=str)
parser.add_argument('--original_rank_index_path', default='./std_eval_idx/')
parser.add_argument('--scales', type=str, default='0.5,0.75,1.25,1.5')
parser.add_argument('--load_dir', type=str,default='')

parser.add_argument('--epoch_img', type=int, default=49)
parser.add_argument('--pos_weights', type=str, default='1.0,0.5,0.1', help='positive sample weights order [furthest,boundary,random]')
parser.add_argument('--output', type=str, default='')
parser.add_argument('--eps', type=float, default= 12/255 )
args = parser.parse_args()



model_list = ['ALBEF', 'TCL', 'BLIP', 'XVLM', 'ViT-B/16', 'RN101']

record = dict()
for model_name in model_list:
    record[model_name] = dict()

# Run image-attack evaluation for retrieval.
def retrieval_eval(record, tokenizer, blip_tokenizer, target_transform, data_loader, device, config):


    for model_name in model_list:
        record[model_name]['model'].float()
        record[model_name]['model'].eval()


    source_model = record[args.source_model]['model']


    image_G_input_dim = 3
    image_G_output_dim = 3
    image_num_filters = [64, 128, 256]
    if args.source_model in ['ALBEF', 'TCL', 'BLIP', 'XVLM']:
        context_dim = 256
    elif args.source_model == 'ViT-B/16':
        context_dim = 512
    elif args.source_model == 'RN101':
        context_dim = 512
    else:
        context_dim = 512
    weights_tag = f"{args.source_model}-[{args.pos_weights.replace(' ', '')}]"
    load_dir = os.path.join(args.load_dir, weights_tag, config['dataset'])


    image_generator = Generator(image_G_input_dim, image_num_filters, image_G_output_dim, num_heads=1, context_dim=context_dim)
    image_generator.load_state_dict(torch.load(os.path.join(load_dir, f'image-model-{args.epoch_img}.pth'), map_location=device))
    image_generator = image_generator.to(device)
    image_generator.eval()
    print('Computing features for evaluation adv...')

    if args.source_model in ['ALBEF', 'TCL']:

        images_normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )
    elif args.source_model in ['BLIP', 'XVLM']:

        images_normalize = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    elif args.source_model in ['ViT-B/16', 'RN101']:

        images_normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )
    else:

        images_normalize = transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )

    print('Prepare memory')
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)

    for model_name in model_list:
        if model_name == 'BLIP':
            record[model_name]['text_ids'] = []
        if model_name in ['ALBEF', 'TCL', 'BLIP']:
            record[model_name]['image_feats'] = torch.zeros(num_image, config['embed_dim'])
            record[model_name]['image_embeds'] = torch.zeros(num_image, 577, 768)
            record[model_name]['text_feats'] = torch.zeros(num_text, config['embed_dim'])
            record[model_name]['text_embeds'] = torch.zeros(num_text, 30, 768)
            record[model_name]['text_atts'] = torch.zeros(num_text, 30).long()
        elif model_name == 'XVLM':
            record[model_name]['image_feats'] = torch.zeros(num_image, config['embed_dim'])
            record[model_name]['image_embeds'] = torch.zeros(num_image, 145, 1024)
            record[model_name]['text_feats'] = torch.zeros(num_text, config['embed_dim'])
            record[model_name]['text_embeds'] = torch.zeros(num_text, 30, 768)
            record[model_name]['text_atts'] = torch.zeros(num_text, 30).long()
        else:
            record[model_name]['image_feats'] = torch.zeros(num_image, record[model_name]['model'].visual.output_dim)
            record[model_name]['text_feats'] = torch.zeros(num_text, record[model_name]['model'].visual.output_dim)


    print('Forward')
    for batch_idx, (images, texts_group, images_ids, text_ids_groups) in enumerate(tqdm(data_loader)):
        texts_ids = []
        txt2img = []
        texts = []
        img2txt = []
        txt_id = 0
        for i in range(len(texts_group)):
            texts += texts_group[i]
            texts_ids += text_ids_groups[i]
            txt2img += [i]*len(text_ids_groups[i])
            img2txt.append([])
            for j in range(len(texts_group[i])):
                img2txt[i].append(txt_id)
                txt_id = txt_id + 1



        txts_input = tokenizer(texts, padding='max_length', truncation=True, max_length=30,
                                                     return_tensors="pt").to(device)
        txts_output = source_model.inference_text(txts_input)
        txt_embeds = txts_output['text_feat']


        text_cond = []
        for i in range(len(img2txt)):
            text_cond.append(torch.mean(txt_embeds[img2txt[i]], dim=0))
        text_cond = torch.stack(text_cond, dim=0).to(device)


        images = images.to(device)
        img_noise = image_generator(images, text_cond)
        if args.source_model in ['ViT-B/16', 'RN101']:
            img_noise = F.interpolate(img_noise, size=(224, 224), mode='bilinear')
        img_noise = torch.clamp(img_noise, -args.eps, args.eps)
        adv_imgs = images + img_noise.expand(images.size())


        adv_texts = texts

        with torch.no_grad():
            t_adv_img_list = []
            for itm in adv_imgs:
                t_adv_img_list.append(target_transform(itm))
            t_adv_imgs = torch.stack(t_adv_img_list, 0).to(device)


            for model_name in model_list:
                if model_name in ['ALBEF', 'TCL', 'XVLM', 'BLIP']:
                    if args.source_model in ['ALBEF', 'TCL', 'XVLM', 'BLIP']:
                        adv_images = adv_imgs
                    else:
                        adv_images = t_adv_imgs
                    adv_images_norm = images_normalize(adv_images)


                    if model_name == 'BLIP':

                        adv_texts_input = blip_tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
                        record[model_name]['text_ids'].append(adv_texts_input.input_ids)
                    else:

                        adv_texts_input = tokenizer(adv_texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)

                    output_img = record[model_name]['model'].inference_image(adv_images_norm)
                    output_txt = record[model_name]['model'].inference_text(adv_texts_input)

                    record[model_name]['image_feats'][images_ids] = output_img['image_feat'].cpu().detach()
                    record[model_name]['image_embeds'][images_ids] = output_img['image_embed'].cpu().detach()
                    record[model_name]['text_feats'][texts_ids] = output_txt['text_feat'].cpu().detach()
                    record[model_name]['text_embeds'][texts_ids] = output_txt['text_embed'].cpu().detach()
                    record[model_name]['text_atts'][texts_ids] = adv_texts_input.attention_mask.cpu().detach()
                else:
                    if args.source_model in ['ALBEF', 'TCL', 'XVLM', 'BLIP']:
                        adv_images = t_adv_imgs
                    else:
                        adv_images = adv_imgs

                    adv_images_norm = images_normalize(adv_images)

                    output = record[model_name]['model'].inference(adv_images_norm, adv_texts)

                    record[model_name]['image_feats'][images_ids] = output['image_feat'].cpu().float().detach()
                    record[model_name]['text_feats'][texts_ids] = output['text_feat'].cpu().float().detach()
    record['BLIP']['text_ids'] = torch.cat(record['BLIP']['text_ids'], dim=0)

    for model_name in model_list:
        if model_name in ['ALBEF', 'TCL', 'XVLM', 'BLIP']:
            record[model_name]['score_matrix_i2t'], record[model_name]['score_matrix_t2i'] = retrieval_score(record, model_name, num_image, num_text, config, device=device)
        else:
            sims_matrix = record[model_name]['image_feats'] @ record[model_name]['text_feats'].t()
            record[model_name]['score_matrix_i2t'] = sims_matrix.cpu().numpy()
            record[model_name]['score_matrix_t2i'] = sims_matrix.t().cpu().numpy()
    return

@torch.no_grad()

# Compute refined i2t/t2i scores with cross-encoder.
def retrieval_score(record, model_name, num_image, num_text, config, device=None):
    if device is None:
        device = record[model_name]['image_embeds'].device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation Direction Similarity With Image Attack:'


    image_feats = record[model_name]['image_feats']
    text_feats = record[model_name]['text_feats']
    image_embeds = record[model_name]['image_embeds']
    text_embeds = record[model_name]['text_embeds']
    text_atts = record[model_name]['text_atts']
    model = record[model_name]['model']


    sims_matrix = image_feats @ text_feats.t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_embeds[i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        if model_name == 'BLIP':
            text_ids = record[model_name]['text_ids']
            output, _ = model.text_encoder(text_ids[topk_idx].to(device),
                                            attention_mask=text_atts[topk_idx].to(device),
                                            encoder_hidden_states=encoder_output,
                                            encoder_attention_mask=encoder_att,
                                            return_dict=True
                                            )
        else:
            output = model.text_encoder(encoder_embeds=text_embeds[topk_idx].to(device),
                                        attention_mask=text_atts[topk_idx].to(device),
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        mode='fusion'
                                        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        if model_name == 'BLIP':
            text_ids = record[model_name]['text_ids']
            output, _ = model.text_encoder(text_ids[i].repeat(config['k_test'], 1).to(device),
                                            attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device),
                                            encoder_hidden_states=encoder_output,
                                            encoder_attention_mask=encoder_att,
                                            return_dict=True
                                            )
        else:
            output = model.text_encoder(encoder_embeds=text_embeds[i].repeat(config['k_test'], 1, 1).to(device),
                                        attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device),
                                        encoder_hidden_states=encoder_output,
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        mode='fusion'
                                        )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

@torch.no_grad()

# Compute attack success rates for retrieval.
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img, model_name, config):

    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]

        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank





    after_attack_tr1 = np.where(ranks < 1)[0]
    after_attack_tr5 = np.where(ranks < 5)[0]
    after_attack_tr10 = np.where(ranks < 10)[0]

    original_rank_index_path = os.path.join(args.original_rank_index_path, config['dataset'])
    origin_tr1 = np.load(f'{original_rank_index_path}/{model_name}_tr1_rank_index.npy')
    origin_tr5 = np.load(f'{original_rank_index_path}/{model_name}_tr5_rank_index.npy')
    origin_tr10 = np.load(f'{original_rank_index_path}/{model_name}_tr10_rank_index.npy')

    asr_tr1 = round(100.0 * len(np.setdiff1d(origin_tr1, after_attack_tr1)) / len(origin_tr1), 2)
    asr_tr5 = round(100.0 * len(np.setdiff1d(origin_tr5, after_attack_tr5)) / len(origin_tr5), 2)
    asr_tr10 = round(100.0 * len(np.setdiff1d(origin_tr10, after_attack_tr10)) / len(origin_tr10), 2)


    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]






    after_attack_ir1 = np.where(ranks < 1)[0]
    after_attack_ir5 = np.where(ranks < 5)[0]
    after_attack_ir10 = np.where(ranks < 10)[0]

    origin_ir1 = np.load(f'{original_rank_index_path}/{model_name}_ir1_rank_index.npy')
    origin_ir5 = np.load(f'{original_rank_index_path}/{model_name}_ir5_rank_index.npy')
    origin_ir10 = np.load(f'{original_rank_index_path}/{model_name}_ir10_rank_index.npy')

    asr_ir1 = round(100.0 * len(np.setdiff1d(origin_ir1, after_attack_ir1)) / len(origin_ir1), 2)
    asr_ir5 = round(100.0 * len(np.setdiff1d(origin_ir5, after_attack_ir5)) / len(origin_ir5), 2)
    asr_ir10 = round(100.0 * len(np.setdiff1d(origin_ir10, after_attack_ir10)) / len(origin_ir10), 2)

    eval_result = {'txt_r1': asr_tr1,
                   'txt_r5': asr_tr5,
                   'txt_r10': asr_tr10,
                   'img_r1': asr_ir1,
                   'img_r5': asr_ir5,
                   'img_r10': asr_ir10}
    return eval_result


# Orchestrate evaluation and print results.
def eval_asr(record, ref_model, tokenizer, blip_tokenizer, target_transform, data_loader, device, config):

    for model_name in model_list:
        record[model_name]['model'] = record[model_name]['model'].to(device)
    ref_model = ref_model.to(device)

    print("Start eval")
    start_time = time.time()

    retrieval_eval(record, tokenizer, blip_tokenizer, target_transform, data_loader, device, config)

    result = {}
    for model_name in model_list:
        if model_name in ['ALBEF', 'TCL', 'XVLM', 'BLIP']:
            record[model_name]['result'] = itm_eval(record[model_name]['score_matrix_i2t'], record[model_name]['score_matrix_t2i'], data_loader.dataset.img2txt, data_loader.dataset.txt2img, model_name, config)
        elif model_name == 'ViT-B/16':
            record[model_name]['result'] = itm_eval(record[model_name]['score_matrix_i2t'], record[model_name]['score_matrix_t2i'], data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'CLIP_ViT', config)
        else:
            record[model_name]['result'] = itm_eval(record[model_name]['score_matrix_i2t'], record[model_name]['score_matrix_t2i'], data_loader.dataset.img2txt, data_loader.dataset.txt2img, 'CLIP_CNN', config)
        print('Performance on {}: {}'.format(model_name, record[model_name]['result']))

        i2t = model_name + ' i2t'
        t2i = model_name + ' t2i'
        result[i2t] = []
        result[t2i] = []
        result[i2t].append(record[model_name]['result']['txt_r1'])
        result[i2t].append(record[model_name]['result']['txt_r5'])
        result[i2t].append(record[model_name]['result']['txt_r10'])
        result[t2i].append(record[model_name]['result']['img_r1'])
        result[t2i].append(record[model_name]['result']['img_r5'])
        result[t2i].append(record[model_name]['result']['img_r10'])
    torch.cuda.empty_cache()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


    print_attack_summary(record, model_list)


# Summarize ASR across models.
def print_attack_summary(record, model_list):
    print("\n" + "="*80)
    print("MULTIMODAL ATTACK SUMMARY")
    print("="*80)

    for model_name in model_list:
        if 'result' in record[model_name]:
            result = record[model_name]['result']
            print(f"\n{model_name} Model Attack Results:")
            print(f"  Image->Text (I2T) Attack Success Rate:")
            print(f"    Top-1: {result['txt_r1']:.2f}%")
            print(f"    Top-5: {result['txt_r5']:.2f}%")
            print(f"    Top-10: {result['txt_r10']:.2f}%")
            print(f"  Text->Image (T2I) Attack Success Rate:")
            print(f"    Top-1: {result['img_r1']:.2f}%")
            print(f"    Top-5: {result['img_r5']:.2f}%")
            print(f"    Top-10: {result['img_r10']:.2f}%")


            avg_asr = (result['txt_r1'] + result['txt_r5'] + result['txt_r10'] +
                      result['img_r1'] + result['img_r5'] + result['img_r10']) / 6
            print(f"  Average ASR: {avg_asr:.2f}%")

    print("\n" + "="*80)
    print("Note: Higher ASR values indicate more successful attacks")
    print("="*80)


config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
device = torch.device('cuda')


seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True

tokenizer = None
blip_tokenizer = None
print("Creating Source Model")
for model_name in model_list:
    record[model_name]['ckpt'] = os.path.join(args.checkpoint, model_name, '{}.pth'.format(config['dataset']))
    if model_name == 'BLIP':
        record[model_name]['model'], ref_model, blip_tokenizer = load_model(model_name, record[model_name]['ckpt'], args.text_encoder, config, device)
    else:
        record[model_name]['model'], ref_model, tokenizer = load_model(model_name, record[model_name]['ckpt'], args.text_encoder, config, device)


if tokenizer is None:
    tokenizer = blip_tokenizer


print("Creating dataset")

if args.source_model in ['ALBEF', 'TCL', 'BLIP', 'XVLM']:
    source_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])

    n_px = 224
    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
    ])
else:
    target_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])

    n_px = 224
    source_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
    ])

test_dataset = paired_dataset(config['annotation_file'], source_transform, config['image_root'])
test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            num_workers=4, collate_fn=test_dataset.collate_fn)

eval_asr(record, ref_model, tokenizer, blip_tokenizer, target_transform, test_loader, device, config)
