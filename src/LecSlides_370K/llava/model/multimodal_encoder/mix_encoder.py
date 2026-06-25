import os
from llava.model.nougat import NougatModel,NougatConfig
import torch
import torch.nn as nn
from transformers import NougatProcessor, VisionEncoderDecoderModel
from transformers import AutoConfig,AutoModel, AutoProcessor
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import LayoutLMv3Model

class MixDocVisionTower(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('using mix encoder')
        self.ocr_free_path = args.ocr_free_tower
        self.ocr_base_path = args.ocr_base_tower
        self.args = args
        self.init_ocr_free_encoder(self.ocr_free_path)
        self.init_ocr_base_encoder(self.ocr_base_path)
        # nougat = VisionEncoderDecoderModel.from_pretrained(self.ocr_free_path)
        # self.ocr_free_model = nougat.encoder

    def init_ocr_base_encoder(self, pretrain_path):
        self.ocr_base_model = LayoutLMv3Model.from_pretrained(pretrain_path)
        self.image_processor['ocr_base'] = AutoProcessor.from_pretrained(pretrain_path, apply_ocr=False)
        self.ocr_base_config = self.ocr_base_model.config
        self.ocr_base_model.requires_grad_(False)





    def init_ocr_free_encoder(self,pretrain_path):
        # nougat
        # nougat = NougatModel.from_pretrained(pretrain_path)
        # ckpt = torch.load(os.path.join(pretrain_path,'pytorch_model.bin'))
        # msg = nougat.load_state_dict(ckpt)
        # print(f'load ocr free encoder weights: {msg}')
        # self.ocr_free_model = nougat.encoder
        # self.ocr_free_config = AutoConfig.from_pretrained('/data/emzhang/data/nougat-base').encoder
        # self.image_processor = {}
        # self.image_processor['ocr_free'] = NougatProcessor.from_pretrained('/data/emzhang/data/nougat-base')
        # self.ocr_free_model.requires_grad_(False)
        # self.ocr_free_name = 'my_nougat'

        #ViT
        self.ocr_free_model = CLIPVisionModel.from_pretrained(pretrain_path)
        self.image_processor = {}
        self.ocr_free_config = self.ocr_free_model.config
        self.image_processor['ocr_free'] = CLIPImageProcessor.from_pretrained(pretrain_path)
        self.ocr_free_model.requires_grad_(False)
        self.select_layer = self.args.mm_vision_select_layer
        self.select_feature = getattr(self.args, 'mm_vision_select_feature', 'patch')
        self.ocr_free_name = 'clip_vit'






        # nougat hf
        # def get_w(weights, keyword):
        #     return {k[len(keyword) + 1:]: v for k, v in weights.items() if k.startswith(keyword)}
        # config = AutoConfig.from_pretrained(pretrain_path)
        # ckpt = torch.load(os.path.join(pretrain_path,'pytorch_model.bin'))
        # encode_weights = get_w(ckpt, 'encoder')
        # model = AutoModel.from_config(config.encoder)
        # msg = model.load_state_dict(encode_weights)
        # print(f'load ocr free encoder weights: {msg}')
        # return model

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, ocr_free_images, ocr_ids, ocr_attn, ocr_bbox, ocr_base_images):

        clip_output = self.clip_forward(ocr_free_images)
        layoutlm_output = self.ocr_base_model(input_ids=ocr_ids,attention_mask=ocr_attn,bbox=ocr_bbox,pixel_values=ocr_base_images).last_hidden_state

        return {'ocr_free_feature':clip_output,'ocr_base_feature':layoutlm_output}

        # if self.ocr_free_name == 'clip_vit':
        #     return self.clip_forward(images)
        # if type(images) is list:
        #     image_features = []
        #     for image in images:
        #         # image_forward_out = self.ocr_free_model(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))[0]
        #         image_forward_out = self.ocr_free_model(image.unsqueeze(0))
        #         image_features.append(image_forward_out)
        # else:
        #     # image_forward_out = self.ocr_free_model(images.to(device=self.device, dtype=self.dtype))[0]
        #     image_forward_out = self.ocr_free_model(images)
        #     image_features = image_forward_out
        # return image_features

    @torch.no_grad()
    def clip_forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.ocr_free_model(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.ocr_free_model(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dtype(self):
        return self.ocr_free_model.dtype

    @property
    def device(self):
        return self.ocr_free_model.device

    @property
    def hidden_size(self):
        return self.ocr_free_config.hidden_size

    @property
    def config(self):
        return self.ocr_free_config


