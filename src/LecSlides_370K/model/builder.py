#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, model_args, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    model_map = {
        'llava_phi': LlavaPhiForCausalLM,
        'mix': MixedLlavaPhiForCausalLM,
        'mix_decouple': MixedLlavaPhiForCausalLMDecoupleLayout,
        'merge_phi': MergeLlavaPhiForCausalLM,
        'merge_phi_wattn': MergeLlavaPhiForCausalLMWTG,
        'udop_phi': UDOPLlavaPhiForCausalLM,
        'udop_phi_merge':UDOPLlavaPhiForCausalLMMerging,
        'udop_t5': UDOP2T5,
        'udop_t5_single': UDOP2T5Single,
        'udop_t5_single_order': UDOP2T5SingleOrder,
        'udop_t5_order': UDOP2T5SingleOrderMerge,
        'layoutlm_t5': LayoutLM2T5,
    }

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    model_map_name =  getattr(model_args,'model_map_name','llava_phi')
    if model_base is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = model_map[model_map_name].from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                      **kwargs)
        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        print(f'loading model saved in {model_path}')
        if 't5' in model_map_name:
            tokenizer = AutoTokenizer.from_pretrained('/home/emzhang/data/t5-large', use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = model_map[model_map_name].from_pretrained(
            model_path,
        )

    image_processor = None

    # if 'llava' in model_name.lower():
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if model_map_name in ['llava_phi','merge_phi']:
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor
    elif 't5' in model_map_name:
        vision_tower = model.vision_tower
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor
    else:
        vision_tower = model.get_vision_tower()
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor



    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
