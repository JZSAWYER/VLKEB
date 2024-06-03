"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from easyeditor.evaluate import portability_evaluate

from .processor.base_dataset import BaseDataset
from .processor.blip_processors import BlipImageEvalProcessor
from ..trainer.utils import dict_to
from PIL import Image
# from ..easyeditor.dataset.evaluate_vqa import QuestionProcessor
import random
import typing
import torch
import transformers
import re
from transformers import AutoTokenizer
from tqdm import tqdm

class VQADataset(BaseDataset):
    def __init__(self, data_dir: str, size:  typing.Optional[int] = None, config=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        vis_processor = None
        # get tokenizer and vis_processor
        if "qwen-vl" in config.model_name.lower():
            vis_processor = BlipImageEvalProcessor(image_size=448, mean=None, std=None)
        elif "owl-2" in config.model_name.lower():
            # from ..trainer.mPLUG_Owl2.mplug_owl2.model.builder import load_pretrained_model
            # from ..trainer.mPLUG_Owl2.mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
            from transformers.models.clip.image_processing_clip import CLIPImageProcessor
            vis_processor = CLIPImageProcessor.from_pretrained(config.name, trust_remote_code=True)
            # model_name = get_model_name_from_path(config.name)
            # tokenizer, _, vis_processor, _ = load_pretrained_model(config.name, None, 'mPLUG_Owl2', load_8bit=False, load_4bit=False, device=f"cuda:{config.device}")
        else:
            vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            if config.model_name == 'qwen-vl':
                tokenizer = AutoTokenizer.from_pretrained(config.name, trust_remote_code=True)
            elif config.model_name == "owl-2":
                tokenizer = AutoTokenizer.from_pretrained(config.name, use_fast=False, trust_remote_code=True)
            else:
                tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        self.prompt = "Question: {} Short answer: "

        data = []
        for i, record in tqdm(enumerate(self.annotation), desc=f"Processing with {data_dir.split('/')[-1]}", total=len(self.annotation)):
            
            if record['alt'] == "":
                continue
            
            if config.model_name == 'qwen-vl':
                image = os.path.join(self.vis_root, record["image"])
                rephrase_image = os.path.join(self.rephrase_root, record["image_rephrase"])
                locality_image = os.path.join(self.vis_root, record['m_loc'])
            elif config.model_name == "owl-2":
                from ..trainer.mPLUG_Owl2.mplug_owl2.mm_utils import process_images
                image_path = os.path.join(self.vis_root, record["image"])
                rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
                locality_image_path = os.path.join(self.vis_root, record['m_loc'])
                    
                _image = Image.open(image_path).convert('RGB')
                max_edge = max(_image.size) # We recommand you to resize to squared image for BEST performance.
                image = process_images([_image.resize((max_edge, max_edge))], self.vis_processor)
                # new_images = []
                # _image = vis_processor.preprocess(_image.resize((max_edge, max_edge)), return_tensors='pt')['pixel_values'][0]
                # new_images.append(_image)
                # if all(x.shape == new_images[0].shape for x in new_images):
                #     new_images = torch.stack(new_images, dim=0)
                # image = new_images
                # image = new_images.to(f"cuda:{config.device}", dtype=torch.float16)

                _image = Image.open(rephrase_image_path).convert('RGB')
                max_edge = max(_image.size) # We recommand you to resize to squared image for BEST performance.
                # new_images = []
                # _image = vis_processor.preprocess(_image.resize((max_edge, max_edge)), return_tensors='pt')['pixel_values'][0]
                # new_images.append(_image)
                # if all(x.shape == new_images[0].shape for x in new_images):
                #     new_images = torch.stack(new_images, dim=0)
                rephrase_image = process_images([_image.resize((max_edge, max_edge))], self.vis_processor)
                # rephrase_image = new_images.to(f"cuda:{config.device}", dtype=torch.float16)

                _image = Image.open(locality_image_path).convert('RGB')
                max_edge = max(_image.size) # We recommand you to resize to squared image for BEST performance.
                # new_images = []
                # _image = vis_processor.preprocess(_image.resize((max_edge, max_edge)), return_tensors='pt')['pixel_values'][0]
                # new_images.append(_image)
                # if all(x.shape == new_images[0].shape for x in new_images):
                #     new_images = torch.stack(new_images, dim=0)
                # locality_image = new_images
                locality_image = process_images([_image.resize((max_edge, max_edge))], self.vis_processor)
                # locality_image = new_images.to(f"cuda:{config.device}", dtype=torch.float16)
            else:
                image_path = os.path.join(self.vis_root, record["image"])
                rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
                locality_image_path = os.path.join(self.vis_root, record['m_loc'])
                
                image = Image.open(image_path).convert("RGB")
                # Check if rephrase_image_path exists before opening
                if os.path.exists(rephrase_image_path):
                    # print(f"Rephrased image found: {rephrase_image_path}")
                    rephrase_image = Image.open(rephrase_image_path).convert("RGB")
                else:
                    continue  # Skip the rest of the loop and proceed with the next record
                locality_image = Image.open(locality_image_path).convert("RGB")

                image = self.vis_processor(image)
                rephrase_image = self.vis_processor(rephrase_image)  
                locality_image = self.vis_processor(locality_image)  
                      
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': image,
                'image_rephrase': rephrase_image,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }
            
            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']
            
            item['multimodal_locality_image'] = locality_image
            item['multimodal_locality_prompt'] = record['m_loc_q']
            item['multimodal_locality_ground_truth'] = record['m_loc_a']
            if ('port_new' in record):
                item['port_q'] = []
                item['port_a'] = []
                item['hop'] = []
                for k in record['port_new']:
                    item['port_q'].append(k['Q&A']['Question'])
                    item['port_a'].append(k['Q&A']['Answer'])
                    item['hop'].append(k['port_type'])
            else:
                continue
            data.append(item)
            
        if size is not None:
            data = data[:size]        
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]
        
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = torch.stack(image, dim=0)
        edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl":
            edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(s)) for s in src]
            edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        # # edit_outer
        edit_outer = {}
        edit_outer['image'] = torch.stack(image, dim=0)
        edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_outer['prompts_len'] = [len(self.tok.encode(r)) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl":
            edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_outer_image['prompts_len'] = [len(self.tok.encode(s)) for s in src]
            edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        # # loc
        loc = {}
        loc['image'] = None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, return_tensors="pt",)["input_ids"]
        
        # # m_loc
        loc_image = {}
        loc_image['image'] = torch.stack(m_loc_image, dim=0)
        loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, return_tensors="pt",)["input_ids"]

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)
        
        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            "cond": cond
        }
        return dict_to(batch, self.config.device)
    
    def collate_fn_qwen(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]
        if ('port_q' in batch[0]):
            port_q = batch[0]['port_q']
            port_a = batch[0]['port_a']
            hop = batch[0]['hop']
            
        
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = image
        edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        text = ''
        num_images = 1
        text += f'Picture {num_images}: '
        text += '<img>' + image[0] + '</img>'
        text += '\n'
        text += src[0] + " " + trg[0]
        edit_inner['inputs'] = self.tok(text, return_tensors='pt')["input_ids"]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl":
            edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(s)) for s in src]
            edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        # # edit_outer
        edit_outer = {}
        edit_outer['image'] = image
        edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        text = ''
        num_images = 1
        text += f'Picture {num_images}: '
        text += '<img>' + image[0] + '</img>'
        text += '\n'
        text += rephrase[0] + " " + trg[0]
        edit_outer['inputs'] = self.tok(text, return_tensors='pt')["input_ids"]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl":
            edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_outer['prompts_len'] = [len(self.tok.encode(r)) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = image_rephrase
        edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg 
        text = ''
        num_images = 1
        text += f'Picture {num_images}: '
        text += '<img>' + image_rephrase[0] + '</img>'
        text += '\n'
        text += src[0] + " " + trg[0]
        edit_outer_image['inputs'] = self.tok(text, return_tensors='pt')["input_ids"]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl":
            edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_outer_image['prompts_len'] = [len(self.tok.encode(s)) for s in src]
            edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        # # loc
        loc = {}
        loc['image'] = None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        text = ''
        text += loc_q[0] + " " + loc_a[0]
        loc['inputs'] = self.tok(text, return_tensors='pt')["input_ids"]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl":
            loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, return_tensors="pt",)["input_ids"]
        
        # # m_loc
        loc_image = {}
        loc_image['image'] = m_loc_image
        loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        text = ''
        num_images = 1
        text += f'Picture {num_images}: '
        text += '<img>' + m_loc_image[0] + '</img>'
        text += '\n'
        text += m_loc_q[0] + " " + m_loc_a[0]
        loc_image['inputs'] = self.tok(text, return_tensors='pt')["input_ids"]
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl":
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, return_tensors="pt",)["input_ids"]

        #Portability
        port = []
        if ('port_q' in batch[0]):
            cnt = 0
            for _q, _a in zip(port_q, port_a):
                port_new = {}
                port_new['text_input'] = [_q + _a]
                port_new['target'] = _a
                text = ''
                num_images = 1
                text += f'Picture {num_images}: '
                text += '<img>' + image[0] + '</img>'
                text += '\n'
                text += _q + " " + _a
                port_new['inputs'] = self.tok(text, return_tensors='pt')["input_ids"]
                port_new['prompts_len'] = [len(self.tok.encode(_q))]
                port_new['labels'] = self.tok(' ' + _a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
                port_new['hop'] = hop[cnt]
                port.append(port_new)
                cnt +=1

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            # padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)
        
        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            "cond": cond,
            "port": port
        }
        return dict_to(batch, self.config.device)
    
    def collate_fn_inf(self, batch, MAX_WORDS=50):
        images = torch.stack([_['image'] for _ in batch], dim=0)
        prompts = []
        _format_ = "<image><ImageHere>Question:{} Short answer:"
        for _ in batch:
            prompt = _['prompt']
            prompt = re.sub(
                r"([.!\"()*#:;~])",
                "",
                prompt.lower(),
            )
            prompt = prompt.rstrip(" ")
            question_words = prompt.split(" ")
            if len(question_words) > MAX_WORDS:
                prompt = " ".join(question_words[: MAX_WORDS])
            prompt = _format_.format(prompt)
            prompts.append(prompt)
        
        # prompt = QuestionProcessor(prompt)        
    
        #passing images and prompts to infmlmm to predict
        inf_pred = {}
        inf_pred['image'] = images
        inf_pred['prompts'] = prompts
        trg = [b['target'] for b in batch]
        inf_pred['labels'] = trg
        # inf_pred['tok'] = self.tok

        src = [b['prompt'] for b in batch]
        cond = [b['cond'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = torch.stack(image, dim=0)
        edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['ques'] = src
        edit_inner['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "infmllm":
            edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(s)) for s in src]
            edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "infmllm":
            edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_outer_image['prompts_len'] = [len(self.tok.encode(s)) for s in src]
            edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)
        
        batch = {
            "inf_pred": inf_pred,
            "edit_inner": edit_inner,
            # "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            # "loc": loc,
            # "loc_image": loc_image,
            "cond": cond
        }
        return dict_to(batch, self.config.device)


    def collate_fn_owl(self, batch):
        
        from ..trainer.mPLUG_Owl2.mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from ..trainer.mPLUG_Owl2.mplug_owl2.conversation import conv_templates, SeparatorStyle
        from ..trainer.mPLUG_Owl2.mplug_owl2.model.builder import load_pretrained_model
        from ..trainer.mPLUG_Owl2.mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]
        if ('port_q' in batch[0]):
            port_q = batch[0]['port_q']
            port_a = batch[0]['port_a']
            hop = batch[0]['hop']


        # image_tensor = image[0] # Image Path
        # query = src[0]

        # model_name = get_model_name_from_path(self.config.name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.name, use_fast=False)

        # conv = conv_templates["mplug_owl2"].copy()

        # inp = DEFAULT_IMAGE_TOKEN + query
        # conv.append_message(conv.roles[0], inp)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(f"cuda:{self.config.device}")

        # with torch.inference_mode():
        #     logits = model(
        #         input_ids,
        #         images=image_tensor)
        
        # edit_inner
        edit_inner = {}
        edit_inner['image'] = image[0]
        edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        conv = conv_templates["mplug_owl2"].copy()
        inp = DEFAULT_IMAGE_TOKEN + src[0] + " " + trg[0]
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        edit_inner['input_ids'] = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl" or self.config.model_name == "owl-2":
            edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(s)) for s in src]
            edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        # # edit_outer
        edit_outer = {}
        edit_outer['image'] = image[0]
        edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        conv = conv_templates["mplug_owl2"].copy()
        inp = DEFAULT_IMAGE_TOKEN + rephrase[0] + " " + trg[0]
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        edit_outer['input_ids'] = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl" or self.config.model_name == "owl-2":
            edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_outer['prompts_len'] = [len(self.tok.encode(r)) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            
        # edit_outer_image
        edit_outer_image = {}
        edit_outer_image['image'] = image_rephrase[0]
        edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        conv = conv_templates["mplug_owl2"].copy()
        inp = DEFAULT_IMAGE_TOKEN + src[0] + " " + trg[0]
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        edit_outer_image['input_ids'] = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl" or self.config.model_name == "owl-2":
            edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            edit_outer_image['prompts_len'] = [len(self.tok.encode(s)) for s in src]
            edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
        
        # # loc
        loc = {}
        loc['image'] = torch.zeros(1, 3, 448, 448)
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        conv = conv_templates["mplug_owl2"].copy()
        inp = DEFAULT_IMAGE_TOKEN + loc_q[0] + " " + loc_a[0]
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        loc['input_ids'] = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl" or self.config.model_name == "owl-2":
            loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, return_tensors="pt",)["input_ids"]
        
        # # m_loc
        loc_image = {}
        loc_image['image'] = m_loc_image[0]
        loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        conv = conv_templates["mplug_owl2"].copy()
        inp = DEFAULT_IMAGE_TOKEN + m_loc_q[0] + " " + m_loc_a[0]
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        loc_image['input_ids'] = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2" or self.config.model_name == "qwen-vl" or self.config.model_name == "owl-2":
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        else:
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, return_tensors="pt",)["input_ids"]


        # #Portability
        # port = []
        # if ('port_q' in batch[0]):
        #     cnt = 0
        #     for _q, _a in zip(port_q, port_a):
        #         port_new = {}
        #         port_new['image'] = image[0]
        #         port_new['text_input'] = [self.prompt.format(_q) + _a]
        #         port_new['target'] = _a
        #         conv = conv_templates["mplug_owl2"].copy()
        #         inp = DEFAULT_IMAGE_TOKEN + _q + " " + _a
        #         conv.append_message(conv.roles[0], inp)
        #         conv.append_message(conv.roles[1], None)
        #         prompt = conv.get_prompt()
        #         port_new['input_ids'] = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        #         port_new['prompts_len'] = [len(self.tok.encode(self.prompt.format(_q), add_special_tokens=False))]
        #         port_new['labels'] = self.tok(_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        #         port_new['hop'] = hop[cnt]
        #         port.append(port_new)
        #         cnt += 1

        port = []
        if ('port_q' in batch[0]):
            cnt = 0
            for _q, _a in zip(port_q, port_a):
                port_new = {}
                port_new['image'] = image[0]
                # port_new['text_input'] = [self.prompt.format(q) + a for q, a in zip(_q, _a)]
                port_new['text_input'] = [" ".join([_q, _a]) ]
                port_new['target'] = _a
                # conv = conv_templates["mplug_owl2"].copy()
                # inp = DEFAULT_IMAGE_TOKEN + _q[0] + " " + _a[0]
                # inp = DEFAULT_IMAGE_TOKEN + _q + " " + _a
                # conv.append_message(conv.roles[0], inp)
                # conv.append_message(conv.roles[1], None)
                # prompt = conv.get_prompt()
                prompt = DEFAULT_IMAGE_TOKEN + _q + " " + _a
                port_new['input_ids'] = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
                port_new['prompts_len'] = [len(self.tok.encode(prompt, add_special_tokens=False))]
                port_new['labels'] = self.tok(_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
                # print(self.tok(_a, add_special_tokens=False, return_tensors="pt",)["input_ids"])
                # print(self.tok(' '+_a, add_special_tokens=False, return_tensors="pt",)["input_ids"])
                port_new['hop'] = hop[cnt]
                port.append(port_new)
                cnt += 1  
                port.append(port_new)


        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)
        
        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            "cond": cond,
            "port": port
        }
        return dict_to(batch, self.config.device)
