from typing import List, Optional, Tuple, Union
from addict import Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ..multimodal_projector.builder import build_vision_projector
from ..multimodal_encoder.builder import build_vision_tower, build_merging_vision_tower
from transformers import AutoConfig, AutoModelForCausalLM, OPTConfig, OPTModel, OPTForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import LayoutLMv3Model
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import PhiModel, PhiForCausalLM, PhiConfig, AutoProcessor
from ..multimodal_encoder.mix_encoder import MixDocVisionTower
from transformers import CLIPTextModel, AutoTokenizer
import math
from llava.model.multimodal_encoder.UDOP import UDOPEncoder, UDOPEncoderOrder, UdopUnimodelForConditionalGeneration, BaseModelOutputWithVisionEmbeds, UdopUnimodelForConditionalGenerationOrder
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import LayoutLMv3Model
import inspect
import time
import copy

@dataclass
class CausalOutputWithOther(CausalLMOutputWithPast):
    attn_loss: Optional[torch.FloatTensor] = None
    llm_loss: Optional[torch.FloatTensor] = None
class LlavaConfig(PhiConfig):
    model_type = "llava_phi"

class LlavaPhiModel(LlavaMetaModel, PhiModel):
    config_class = LlavaConfig

    def __init__(self, config: PhiConfig):
        super(LlavaPhiModel, self).__init__(config)

class MergeLlavaPhiModel(PhiModel):
    config_class = LlavaConfig

    def __init__(self, config):
        super(MergeLlavaPhiModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_merging_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_merging_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            print('have contained mm_projector, skip init projector')
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

class MixLlavaPhiModel(PhiModel):
    def __init__(self, config: PhiConfig):
        super(MixLlavaPhiModel, self).__init__(config)

        if hasattr(config, "ocr_free_tower"):
            self.vision_tower = MixDocVisionTower(config)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    def initialize_vision_modules(self, model_args, fsdp=None):
        ocr_free_tower = model_args.ocr_free_tower
        ocr_base_tower = model_args.ocr_base_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.ocr_free_tower = ocr_free_tower
        self.config.ocr_base_tower = ocr_base_tower

        if self.get_vision_tower() is None:
            vision_tower = MixDocVisionTower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            print('have contained vision tower, skip init projector')
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            # vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.ocr_free_size = vision_tower.ocr_free_config.hidden_size
        self.config.ocr_base_size = vision_tower.ocr_base_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            print('have contained mm_projector, skip init projector')
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            msg = self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print(f'load mm_projector: {msg}')

class UDOPLlavaPhiModel(PhiModel):
    def __init__(self, config: PhiConfig):
        super(UDOPLlavaPhiModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = UDOPEncoder(config.mm_vision_tower)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = UDOPEncoder(model_args.vision_tower)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = 1024
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            print('have contained mm_projector, skip init projector')
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))



class LlavaPhiForCausalLM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.model = LlavaPhiModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

class MergeLlavaPhiForCausalLM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.model = MergeLlavaPhiModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def merging_feature(self,image_features, key_idx=None):
        # direct merging selected layers
        new_features = []
        for bs, feature in enumerate(image_features):
            use_feat_pool = []
            cur_feature = self.model.vision_tower.feature_select(feature)
            mean_feature = torch.mean(cur_feature,dim=0)
            use_feat_pool.append(mean_feature)
            # add key feature
            if key_idx is not None:
                for idx in key_idx[bs]:
                    use_feat_pool.append(cur_feature[idx])
            use_feature = torch.concat(use_feat_pool,dim=0)
            new_features.append(use_feature)
        if not any(x.shape != new_features[0].shape for x in new_features):
            output_features = torch.stack(new_features,dim=0)
            return output_features
        else:
            max_len = max(x.shape[0] for x in new_features)

            new_features_align = []
            for feature in new_features:
                cur_new_embed = torch.cat((feature,
                                           torch.zeros((max_len - feature.shape[0], feature.shape[1]),
                                                       dtype=feature.dtype, device=feature.device)), dim=0)
                new_features_align.append(cur_new_embed)
            output_features = torch.stack(new_features_align, dim=0)
            return output_features



    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, evidence):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.merging_feature(image_features, key_idx=evidence)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, evidence
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        # if type(images) is list or images.ndim == 5:
        #     concat_images = torch.cat([image for image in images], dim=0)
        #     image_features = self.encode_images(concat_images)
        #     split_sizes = [image.shape[0] for image in images]
        #     image_features = torch.split(image_features, split_sizes, dim=0)
        #     image_features = [x.flatten(0, 1) for x in image_features]
        # else:
        image_features = self.encode_images(images, evidence) #  [2,256,4096]

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                print(f'error, do not contain image')
                raise NotImplementedError
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        evidence = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, evidence)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "evidence": kwargs.get("evidence", None),
            }
        )
        return model_inputs

class MergeLlavaPhiForCausalLMWTG(MergeLlavaPhiForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(MergeLlavaPhiForCausalLMWTG, self).__init__(config)

        is_text_encoder = getattr(config, 'is_text_encoder', False)
        self.is_text_encoder = is_text_encoder
        if is_text_encoder:
            print('text bert are in model, init directly')
            self.init_text_encoder(self.config.mm_vision_tower)
        self.post_init()

    def merging_feature(self,image_features, text_input_ids, key_idx=None):
        # direct merging selected layers
        attention_mask = text_input_ids.ne(-100)
        text_input_ids[text_input_ids==-100] = self.text_tokenizer.pad_token_id
        if text_input_ids.shape[1] > 77:
            text_input_ids = text_input_ids[:,:77]
            attention_mask = attention_mask[:,:77]
            for idx in range(text_input_ids.shape[0]):
                text_input_ids[idx,-1] = self.text_tokenizer.eos_token_id
        text_outputs = self.text_encoder(input_ids=text_input_ids,
                                         attention_mask=attention_mask)
        pooled_text_output = text_outputs.pooler_output.unsqueeze(1)
        new_features = []
        # extract vision embeds
        pooled_image_output = []
        for bs, feature in enumerate(image_features):
            pooled_image_output.append(feature['pooler_output'])
        if not any(x.shape != pooled_image_output[0].shape for x in pooled_image_output):
            pooled_image_output = torch.stack(pooled_image_output,dim=0)
            imag_max_len = None
        else:
            imag_max_len = max(x.shape[0] for x in pooled_image_output)

            new_pooled_image_output_align = []
            for feature in pooled_image_output:
                cur_new_embed = torch.cat((feature,
                                           torch.zeros((imag_max_len - feature.shape[0], feature.shape[1]),
                                                       dtype=feature.dtype, device=feature.device)), dim=0)
                new_pooled_image_output_align.append(cur_new_embed)
            pooled_image_output = torch.stack(new_pooled_image_output_align, dim=0)

        project_pooled_text_output = F.normalize(self.text_projection(pooled_text_output),dim=-1)
        project_pooled_image_output = F.normalize(self.vision_projection(pooled_image_output),dim=-1)
        attn_map = torch.einsum('bld,bcd->blc', project_pooled_text_output, project_pooled_image_output)
        attn_map = attn_map / math.sqrt(self.config.projection_dim)
        attn_prob = nn.Softmax(dim=-1)(attn_map)

        for bs, feature in enumerate(image_features):
            use_feat_pool = []
            cur_feature = self.model.vision_tower.feature_select(feature)
            if imag_max_len is not None and cur_feature.shape[0] != imag_max_len:
                print('now is padding image feature')
                cur_feature = torch.cat((cur_feature,
                                           torch.zeros((imag_max_len - cur_feature.shape[0], cur_feature.shape[1], cur_feature.shape[2]),
                                                       dtype=cur_feature.dtype, device=cur_feature.device)), dim=0)
            attn_feature = torch.einsum('lc,cxd->lxd', attn_prob[bs], cur_feature).squeeze(0)
            use_feat_pool.append(attn_feature)
            # add key feature
            # if key_idx is not None:
            #     for idx in key_idx[bs]:
            #         use_feat_pool.append(cur_feature[idx])
            use_feature = torch.concat(use_feat_pool,dim=0)
            new_features.append(use_feature)
        if not any(x.shape != new_features[0].shape for x in new_features):
            output_features = torch.stack(new_features,dim=0)
            return output_features, attn_map
        else:
            max_len = max(x.shape[0] for x in new_features)

            new_features_align = []
            for feature in new_features:
                cur_new_embed = torch.cat((feature,
                                           torch.zeros((max_len - feature.shape[0], feature.shape[1]),
                                                       dtype=feature.dtype, device=feature.device)), dim=0)
                new_features_align.append(cur_new_embed)
            output_features = torch.stack(new_features_align, dim=0)
            return output_features, attn_map

    def init_text_encoder(self, clip_path):
        if not self.is_text_encoder:
            print('Initialize text encoder...')
            self.config.is_text_encoder = True

        self.text_encoder = CLIPTextModel.from_pretrained(clip_path)
        text_tokenizer = AutoTokenizer.from_pretrained(clip_path)
        self.model.vision_tower.image_processor = {
            'ocr_free': self.model.vision_tower.image_processor,
            'text_tokenizer': text_tokenizer
        }
        self.text_tokenizer = text_tokenizer
        self.config.projection_dim = 768
        self.vision_projection = nn.Linear(self.model.vision_tower.hidden_size,self.config.projection_dim)
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size,self.config.projection_dim)



        # self.ocr_base_model = LayoutLMv3Model.from_pretrained(ocr_base_path)
        # self.ocr_base_model.requires_grad_(False)
        # ocr_base_processor = AutoProcessor.from_pretrained(ocr_base_path, apply_ocr=False)
        # # if not isinstance(self.model.vision_tower.image_processor,dict):
        # self.model.vision_tower.image_processor = {
        #     'ocr_free': self.model.vision_tower.image_processor,
        #     'ocr_base': ocr_base_processor
        # }
        # self.ocr_base_mlp = nn.Linear(self.ocr_base_model.config.hidden_size,self.config.hidden_size)

    def encode_images(self, images, evidence, text_input_ids):
        image_features = self.get_model().get_vision_tower()(images)
        image_features, attn_map = self.merging_feature(image_features, text_input_ids, key_idx=evidence)
        image_features = self.get_model().mm_projector(image_features)
        return image_features, attn_map

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, evidence, text_input_ids
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None

        # if type(images) is list or images.ndim == 5:
        #     concat_images = torch.cat([image for image in images], dim=0)
        #     image_features = self.encode_images(concat_images)
        #     split_sizes = [image.shape[0] for image in images]
        #     image_features = torch.split(image_features, split_sizes, dim=0)
        #     image_features = [x.flatten(0, 1) for x in image_features]
        # else:
        image_features, attn_map = self.encode_images(images, evidence, text_input_ids) #  [2,256,4096]

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                print(f'error, do not contain image')
                raise NotImplementedError
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, attn_map

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "evidence": kwargs.get("evidence", None),
                "text_input_ids": kwargs.get("text_input_ids", None),
            }
        )
        return model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        evidence = None,
        text_input_ids = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, attn_map = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, evidence, text_input_ids)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        total_loss = None
        loss_attn = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            attn_logits = attn_map.squeeze(1).float()
            target_attn = torch.zeros_like(attn_logits).to(attn_logits.device)
            for bs in range(target_attn.shape[0]):
                for evi in evidence[bs]:
                    target_attn[bs, evi] = 1
            num_sample = attn_logits.shape[0] * attn_logits.shape[1]
            num_pos = target_attn.sum()
            pos_weight = (num_sample - num_pos) / num_pos
            # attn_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(attn_logits.device))
            attn_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(attn_logits.device))
            loss_attn = attn_loss_func(attn_logits, target_attn)

            total_loss = loss + loss_attn
            # total_loss = loss





            # calculate bce loss for attn

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalOutputWithOther(
            loss=total_loss,
            logits=logits,
            llm_loss=loss.detach() if loss is not None else None,
            attn_loss=loss_attn.detach() if loss_attn is not None else None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class MixedLlavaPhiForCausalLMDecoupleLayout(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.model = LlavaPhiModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        is_layoutlm = getattr(config, 'layoutlm_train', False)
        self.is_layoutlm = is_layoutlm
        if is_layoutlm:
            print('layoutlm are in model, init directly')
            self.init_layoutlm('/data/emzhang/data/layoutlmv3-base')
        # Initialize weights and apply final processing
        self.post_init()

    def init_layoutlm(self, ocr_base_path):
        if not self.is_layoutlm:
            print('Initialize layoutlm...')
            self.config.layoutlm_train = True
        self.ocr_base_model = LayoutLMv3Model.from_pretrained(ocr_base_path)
        self.ocr_base_model.requires_grad_(False)
        ocr_base_processor = AutoProcessor.from_pretrained(ocr_base_path, apply_ocr=False)
        # if not isinstance(self.model.vision_tower.image_processor,dict):
        self.model.vision_tower.image_processor = {
            'ocr_free': self.model.vision_tower.image_processor,
            'ocr_base': ocr_base_processor
        }
        self.ocr_base_mlp = nn.Linear(self.ocr_base_model.config.hidden_size,self.config.hidden_size)

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encoder_layout_image(self,ocr_ids, ocr_attn, ocr_bbox, ocr_base_images):
        layoutlm_output = self.ocr_base_model(input_ids=ocr_ids, attention_mask=ocr_attn, bbox=ocr_bbox,
                                              pixel_values=ocr_base_images).last_hidden_state
        layoutlm_features = self.ocr_base_mlp(layoutlm_output)
        return layoutlm_features

    def encode_images_wocr(self, ocr_free_images, ocr_ids, ocr_attn, ocr_bbox, ocr_base_images):
        image_features = self.get_model().get_vision_tower()(ocr_free_images, ocr_ids, ocr_attn, ocr_bbox, ocr_base_images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        ocr_ids = None,
        ocr_attn = None,
        ocr_bbox = None,
        ocr_base_images = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if ocr_ids is not None:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal_wocr(input_ids, attention_mask, past_key_values, labels,images, ocr_ids, ocr_attn, ocr_bbox, ocr_base_images)
        else:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_labels_for_multimodal_wocr(
        self, input_ids, attention_mask, past_key_values, labels, images, ocr_ids, ocr_attn, ocr_bbox, ocr_base_images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            # image_features = self.encode_images_wocr(images, ocr_ids, ocr_attn, ocr_bbox, ocr_base_images) #  [2,256,4096]
            image_features = self.encode_images(images) #  [2,256,4096]
            layout_features = self.encoder_layout_image(ocr_ids, ocr_attn, ocr_bbox, ocr_base_images)
            image_features = torch.concat([image_features,layout_features],dim=1)


        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                print(f'error, do not contain image')
                raise NotImplementedError
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "ocr_ids": kwargs.get("ocr_ids", None),
                "ocr_attn": kwargs.get("ocr_attn", None),
                "ocr_bbox": kwargs.get("ocr_bbox", None),
                "ocr_base_images": kwargs.get("ocr_base_images", None),
            }
        )
        return model_inputs

class MixedLlavaPhiForCausalLM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.model = MixLlavaPhiModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_wocr(self, ocr_free_images, ocr_ids, ocr_attn, ocr_bbox, ocr_base_images):
        image_features = self.get_model().get_vision_tower()(ocr_free_images, ocr_ids, ocr_attn, ocr_bbox, ocr_base_images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        ocr_ids = None,
        ocr_attn = None,
        ocr_bbox = None,
        ocr_base_images = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if ocr_ids is not None:
            # ocr_bbox[ocr_bbox > 990] = 990
            # ocr_bbox[ocr_bbox < 0] = 0
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal_wocr(input_ids, attention_mask, past_key_values, labels,images, ocr_ids, ocr_attn, ocr_bbox, ocr_base_images)
            # except:
            #     print(ocr_ids)
            #     print(ocr_bbox)
            #     print(ocr_bbox.shape)
            #     raise ValueError
        else:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_labels_for_multimodal_wocr(
        self, input_ids, attention_mask, past_key_values, labels, images, ocr_ids, ocr_attn, ocr_bbox, ocr_base_images
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images_wocr(images, ocr_ids, ocr_attn, ocr_bbox, ocr_base_images) #  [2,256,4096]

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "ocr_ids": kwargs.get("ocr_ids", None),
                "ocr_attn": kwargs.get("ocr_attn", None),
                "ocr_bbox": kwargs.get("ocr_bbox", None),
                "ocr_base_images": kwargs.get("ocr_base_images", None),
            }
        )
        return model_inputs

class UDOPLlavaPhiForCausalLM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.model = UDOPLlavaPhiModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images_wocr(self, images, ocr_ids, seg_data, visual_seg_data):
        pad_token = self.model.vision_tower.image_processor.pad_token_id
        attn_mask = ocr_ids.ne(pad_token).to(dtype=images.dtype, device=images.device)
        image_features = self.get_model().get_vision_tower()(input_ids=ocr_ids, attention_mask=attn_mask,image=images, seg_data=seg_data, visual_seg_data=visual_seg_data)
        image_features = image_features.last_hidden_state
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        ocrs = None,
        seg_data = None,
        visual_seg_data = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal_wocr(input_ids, attention_mask, past_key_values, labels,images, ocrs, seg_data, visual_seg_data)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_labels_for_multimodal_wocr(
        self, input_ids, attention_mask, past_key_values, labels, images, ocrs, seg_data, visual_seg_data
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        image_features = self.encode_images_wocr(images, ocrs, seg_data, visual_seg_data) #  [2,256,4096]

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                print(f'error, do not contain image')
                raise NotImplementedError
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "ocrs": kwargs.get("ocrs", None),
                "seg_data": kwargs.get("seg_data", None),
                "visual_seg_data": kwargs.get("visual_seg_data", None),
            }
        )
        return model_inputs

class UDOPLlavaPhiForCausalLMMerging(UDOPLlavaPhiForCausalLM):
    def encode_images_wocr(self, images, ocr_ids, seg_data, visual_seg_data):
        image_features_pool = []
        for image, ocr_id, seg_data_, visual_seg_data_ in zip(images, ocr_ids, seg_data, visual_seg_data):
            pad_token = self.model.vision_tower.image_processor.pad_token_id
            attn_mask = ocr_id.ne(pad_token).to(dtype=image.dtype, device=image.device)
            image_features = self.get_vision_tower()(input_ids=ocr_id, attention_mask=attn_mask,image=image, seg_data=seg_data_, visual_seg_data=visual_seg_data_)
            image_features = image_features.last_hidden_state

            # mean pooling
            mean_feature = torch.mean(image_features, dim=0)
            image_features_pool.append(mean_feature)

        if any(x.shape != image_features_pool[0].shape for x in image_features_pool):
            max_token_len = max(x.shape[0] for x in image_features_pool)
            new_pooled_image_feature_align = []
            for feature in image_features_pool:
                cur_new_embed = torch.cat((feature,
                                           torch.zeros((max_token_len - feature.shape[0], feature.shape[1]),
                                                       dtype=feature.dtype, device=feature.device)), dim=0)
                new_pooled_image_feature_align.append(cur_new_embed)
            pool_image_features = torch.stack(new_pooled_image_feature_align, dim=0)
        else:
            pool_image_features = torch.stack(image_features_pool,dim=0)


        # image_features = self.get_model().mm_projector(pool_image_features)
        return pool_image_features

class UDOP2T5(UdopUnimodelForConditionalGenerationOrder):
    def __init__(self, config):
        super(UDOP2T5, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = UDOPEncoder(config._name_or_path)
            # self.mm_projector = build_vision_projector(config)
        # udop = UdopUnimodelForConditionalGeneration.from_pretrained(config._name_or_path)
        # self.decoder = udop.decoder

    def encode_images_wocr(self, images, ocr_ids, seg_data, visual_seg_data, question):
        image_features_pool = []
        image_attn_pool = []
        for image, ocr_id, seg_data_, visual_seg_data_, question_ in zip(images, ocr_ids, seg_data, visual_seg_data, question):
            pad_token = self.vision_tower.image_processor.pad_token_id
            attn_mask = ocr_id.ne(pad_token).to(dtype=image.dtype, device=image.device)
            features = self.vision_tower(input_ids=ocr_id, attention_mask=attn_mask,image=image, seg_data=seg_data_, visual_seg_data=visual_seg_data_, question=question_)
            image_features = features.last_hidden_state
            image_attns = features.attention_mask

            # mean pooling
            # mean_feature = torch.mean(image_features, dim=0)
            # image_attn = torch.sum(image_attns,dim=0)
            # image_attn[image_attn > 0] = 1

            # concat
            mean_feature = image_features.reshape(-1,image_features.shape[2])
            image_attn = image_attns.reshape(-1)
            # truncate
            mean_feature = mean_feature[:1024,...]
            image_attn = image_attn[:1024]


            image_features_pool.append(mean_feature)
            image_attn_pool.append(image_attn)

        if any(x.shape != image_features_pool[0].shape for x in image_features_pool):
            max_token_len = max(x.shape[0] for x in image_features_pool)
            new_pooled_image_feature_align = []
            new_pooled_image_attn_align = []
            for feature in image_features_pool:
                cur_new_embed = torch.cat((feature,
                                           torch.zeros((max_token_len - feature.shape[0], feature.shape[1]),
                                                       dtype=feature.dtype, device=feature.device)), dim=0)
                new_pooled_image_feature_align.append(cur_new_embed)
            for feature in image_attn_pool:
                cur_new_embed = torch.cat((feature,
                                           torch.zeros((max_token_len - feature.shape[0]),
                                                       dtype=feature.dtype, device=feature.device)), dim=0)
                new_pooled_image_attn_align.append(cur_new_embed)
            pool_image_features = torch.stack(new_pooled_image_feature_align, dim=0)
            pool_image_attn = torch.stack(new_pooled_image_attn_align, dim=0)
        else:
            pool_image_features = torch.stack(image_features_pool,dim=0)
            pool_image_attn = torch.stack(image_attn_pool,dim=0)

        return pool_image_features, pool_image_attn

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        ocrs = None,
        seg_data = None,
        visual_seg_data = None,
        question=None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        if encoder_outputs is None:
            if question is None:
                question = [None] * len(ocrs)
            image_features, pool_image_attn = self.encode_images_wocr(images, ocrs, seg_data, visual_seg_data, question)
            encoder_outputs = BaseModelOutputWithVisionEmbeds(
                last_hidden_state=image_features,
                attention_mask=pool_image_attn,
            )
        else:
            image_features = encoder_outputs.last_hidden_state
            pool_image_attn = encoder_outputs.attention_mask
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        if decoder_input_ids is None and labels is None:
            return encoder_outputs
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=image_features,
            encoder_attention_mask=pool_image_attn,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )

    def get_encoder(self):
        return self
    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ):
        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(encoder, "_hf_hook"):
            encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)

        return model_kwargs
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = UDOPEncoder(model_args.vision_tower)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = 1024
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type


class UDOP2T5Single(UDOP2T5):
    def encode_images_wocr(self, images, ocr_ids, seg_data, visual_seg_data):
        pad_token = self.vision_tower.image_processor.pad_token_id
        attn_mask = ocr_ids.ne(pad_token).to(dtype=images.dtype, device=images.device)
        features = self.vision_tower(input_ids=ocr_ids, attention_mask=attn_mask,image=images, seg_data=seg_data, visual_seg_data=visual_seg_data)
        image_features = features.last_hidden_state
        image_attn_mask = features.attention_mask
        return image_features, image_attn_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        ocrs = None,
        seg_data = None,
        visual_seg_data = None
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        if encoder_outputs is None:
            image_features, pool_image_attn = self.encode_images_wocr(images, ocrs, seg_data, visual_seg_data)
            encoder_outputs = BaseModelOutputWithVisionEmbeds(
                last_hidden_state=image_features,
                attention_mask=pool_image_attn,
            )
        else:
            image_features = encoder_outputs.last_hidden_state
            pool_image_attn = encoder_outputs.attention_mask
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        if decoder_input_ids is None and labels is None:
            return encoder_outputs
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=image_features,
            encoder_attention_mask=pool_image_attn,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )

@dataclass
class Seq2SeqLMOutputWithLoss(Seq2SeqLMOutput):
    lm_loss: torch.FloatTensor = None
    rl_loss: torch.FloatTensor = None

class UDOP2T5SingleOrder(UDOP2T5):
    def __init__(self, config):
        super(UDOP2T5, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = UDOPEncoderOrder(config._name_or_path)
            # self.mm_projector = build_vision_projector(config)
        # udop = UdopUnimodelForConditionalGeneration.from_pretrained(config._name_or_path)
        # self.decoder = udop.decoder

    def encode_images_wocr(self, images, ocr_ids, seg_data, visual_seg_data, element_data, text_dist_graph, graph_shape, relation_label, question):
        pad_token = self.vision_tower.image_processor.pad_token_id
        attn_mask = ocr_ids.ne(pad_token).to(dtype=images.dtype, device=images.device)
        features = self.vision_tower(input_ids=ocr_ids, attention_mask=attn_mask,image=images, seg_data=seg_data, visual_seg_data=visual_seg_data, element_data=element_data, text_dist_graph=text_dist_graph, graph_shape=graph_shape, relation_label=relation_label, question=question)
        image_features = features.last_hidden_state
        image_attn_mask = features.attention_mask
        predict_relation = features.predict_relation
        return image_features, image_attn_mask, predict_relation

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = UDOPEncoderOrder(model_args.vision_tower)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            print('have vision tower, skip init')
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = 1024
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        ocrs = None,
        seg_data = None,
        visual_seg_data = None,
        element_data=None,
        text_dist_graph = None,
        graph_shape = None,
        relation_label = None,
        question = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        if encoder_outputs is None:
            image_features, pool_image_attn, predict_relation = self.encode_images_wocr(images, ocrs, seg_data, visual_seg_data, element_data, text_dist_graph, graph_shape, relation_label, question)
            encoder_outputs = BaseModelOutputWithVisionEmbeds(
                last_hidden_state=image_features,
                attention_mask=pool_image_attn,
            )
        else:
            image_features = encoder_outputs.last_hidden_state
            pool_image_attn = encoder_outputs.attention_mask
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        if decoder_input_ids is None and labels is None:
            return encoder_outputs
        # print(image_features.shape)
        # print(f'\n\n\n enming \n\n\n')
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=image_features,
            encoder_attention_mask=pool_image_attn,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        rl_loss = torch.tensor(0, device=lm_logits.device)
        lm_loss = torch.tensor(0, device=lm_logits.device)
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            if relation_label is not None and predict_relation is not None:
                relation_loss = nn.BCELoss(reduction='none')

                rl_loss = relation_loss(predict_relation.sigmoid(), relation_label)
                attn_mask = -100 * torch.ones(rl_loss.shape[0], rl_loss.shape[1], rl_loss.shape[2]).to(
                    dtype=rl_loss.dtype, device=rl_loss.device)
                for bs in range(rl_loss.shape[0]):
                    attn_mask[bs, :graph_shape[bs], :graph_shape[bs]] = 0
                attn_mask = (attn_mask != -100)
                rl_loss = rl_loss * attn_mask
                rl_loss = torch.sum(rl_loss) / torch.sum(attn_mask)
                loss = lm_loss + rl_loss
            else:
                loss = lm_loss




        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutputWithLoss(
            loss=loss,
            lm_loss=lm_loss.detach(),
            rl_loss=rl_loss.detach(),
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )

class UDOP2T5SingleOrderMerge(UDOP2T5):
    def __init__(self, config):
        super(UDOP2T5, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = UDOPEncoderOrder(config._name_or_path)
            # self.mm_projector = build_vision_projector(config)
        # udop = UdopUnimodelForConditionalGeneration.from_pretrained(config._name_or_path)
        # self.decoder = udop.decoder

    def encode_images_wocr(self, images, ocr_ids, seg_data, visual_seg_data, element_data, text_dist_graph, graph_shape, relation_label, question):
        image_features_pool = []
        image_attn_pool = []
        for image, ocr_id, seg_data_, visual_seg_data_,  element_data_, text_dist_graph_, graph_shape_, relation_label_, question_ in zip(images, ocr_ids, seg_data, visual_seg_data, element_data, text_dist_graph, graph_shape, relation_label, question):
            pad_token = self.vision_tower.image_processor.pad_token_id
            attn_mask = ocr_id.ne(pad_token).to(dtype=image.dtype, device=image.device)
            features = self.vision_tower(input_ids=ocr_id, attention_mask=attn_mask, image=image, seg_data=seg_data_,
                                         visual_seg_data=visual_seg_data_, element_data=element_data_,
                                         text_dist_graph=text_dist_graph_, graph_shape=graph_shape_,
                                         relation_label=relation_label_, question=question_)
            image_features = features.last_hidden_state
            image_attns = features.attention_mask

            # mean pooling
            # mean_feature = torch.mean(image_features, dim=0)
            # image_attn = torch.sum(image_attns,dim=0)
            # image_attn[image_attn > 0] = 1

            # concat
            mean_feature = image_features.reshape(-1,image_features.shape[2])
            image_attn = image_attns.reshape(-1)
            # truncate
            mean_feature = mean_feature[:1024,...]
            image_attn = image_attn[:1024]


            image_features_pool.append(mean_feature)
            image_attn_pool.append(image_attn)

        if any(x.shape != image_features_pool[0].shape for x in image_features_pool):
            max_token_len = max(x.shape[0] for x in image_features_pool)
            new_pooled_image_feature_align = []
            new_pooled_image_attn_align = []
            for feature in image_features_pool:
                cur_new_embed = torch.cat((feature,
                                           torch.zeros((max_token_len - feature.shape[0], feature.shape[1]),
                                                       dtype=feature.dtype, device=feature.device)), dim=0)
                new_pooled_image_feature_align.append(cur_new_embed)
            for feature in image_attn_pool:
                cur_new_embed = torch.cat((feature,
                                           torch.zeros((max_token_len - feature.shape[0]),
                                                       dtype=feature.dtype, device=feature.device)), dim=0)
                new_pooled_image_attn_align.append(cur_new_embed)
            pool_image_features = torch.stack(new_pooled_image_feature_align, dim=0)
            pool_image_attn = torch.stack(new_pooled_image_attn_align, dim=0)
        else:
            pool_image_features = torch.stack(image_features_pool,dim=0)
            pool_image_attn = torch.stack(image_attn_pool,dim=0)

        return pool_image_features, pool_image_attn


    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = UDOPEncoderOrder(model_args.vision_tower)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            print('skip init vision tower')
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = 1024
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        ocrs = None,
        seg_data = None,
        visual_seg_data = None,
        element_data=None,
        text_dist_graph = None,
        graph_shape = None,
        relation_label = None,
        question=None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        if encoder_outputs is None:
            if question is None:
                question = [None] * len(ocrs)
            image_features, pool_image_attn = self.encode_images_wocr(images, ocrs, seg_data, visual_seg_data, element_data, text_dist_graph, graph_shape, relation_label, question)
            encoder_outputs = BaseModelOutputWithVisionEmbeds(
                last_hidden_state=image_features,
                attention_mask=pool_image_attn,
            )
        else:
            image_features = encoder_outputs.last_hidden_state
            pool_image_attn = encoder_outputs.attention_mask
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        if decoder_input_ids is None and labels is None:
            return encoder_outputs
        # print(image_features.shape)
        # print(f'\n\n\n enming \n\n\n')
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=image_features,
            encoder_attention_mask=pool_image_attn,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        rl_loss = torch.tensor(0, device=lm_logits.device)
        lm_loss = torch.tensor(0, device=lm_logits.device)
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666




        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutputWithLoss(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )

class LayoutLM2T5(UdopUnimodelForConditionalGeneration):
    def __init__(self, config):
        super(LayoutLM2T5, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = LayoutLMv3Model.from_pretrained(config.mm_vision_tower)
            self.vision_tower.image_processor = AutoProcessor.from_pretrained(config.mm_vision_tower, apply_ocr=False)
            if 'base' in config.mm_vision_tower:
                self.mm_projector = nn.Linear(768, 1024)
            elif 'large' in config.mm_vision_tower:
                self.mm_projector = nn.Linear(1024, 1024)
            else:
                raise NotImplementedError
            # self.mm_projector = build_vision_projector(config)
        udop = UdopUnimodelForConditionalGeneration.from_pretrained(config._name_or_path)
        self.decoder = udop.decoder

    def encode_images_wocr(self, images, ocr_ids, ocr_attn, ocr_bbox):
        image_features_pool = []
        image_attn_pool = []
        for image, ocr_id, ocr_attn_, ocr_bbox_ in zip(images, ocr_ids, ocr_attn, ocr_bbox):
            features = self.vision_tower(input_ids=ocr_id, attention_mask=ocr_attn_, bbox=ocr_bbox_,
                                              pixel_values=image)
            image_features = features.last_hidden_state
            image_attns = torch.ones(image_features.shape[0],image_features.shape[1]).to(image_features.device)

            # mean pooling
            # mean_feature = torch.mean(image_features, dim=0)
            # image_attn = torch.sum(image_attns,dim=0)
            # image_attn[image_attn > 0] = 1

            # concat
            mean_feature = image_features.reshape(-1,image_features.shape[2])
            image_attn = image_attns.reshape(-1)
            # truncate
            mean_feature = mean_feature[:1024,...]
            image_attn = image_attn[:1024]


            image_features_pool.append(mean_feature)
            image_attn_pool.append(image_attn)

        if any(x.shape != image_features_pool[0].shape for x in image_features_pool):
            max_token_len = max(x.shape[0] for x in image_features_pool)
            new_pooled_image_feature_align = []
            new_pooled_image_attn_align = []
            for feature in image_features_pool:
                cur_new_embed = torch.cat((feature,
                                           torch.zeros((max_token_len - feature.shape[0], feature.shape[1]),
                                                       dtype=feature.dtype, device=feature.device)), dim=0)
                new_pooled_image_feature_align.append(cur_new_embed)
            for feature in image_attn_pool:
                cur_new_embed = torch.cat((feature,
                                           torch.zeros((max_token_len - feature.shape[0]),
                                                       dtype=feature.dtype, device=feature.device)), dim=0)
                new_pooled_image_attn_align.append(cur_new_embed)
            pool_image_features = torch.stack(new_pooled_image_feature_align, dim=0)
            pool_image_attn = torch.stack(new_pooled_image_attn_align, dim=0)
        else:
            pool_image_features = torch.stack(image_features_pool,dim=0)
            pool_image_attn = torch.stack(image_attn_pool,dim=0)

        pool_image_features = self.mm_projector(pool_image_features)

        return pool_image_features, pool_image_attn

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        ocr_ids = None,
        ocr_attn = None,
        ocr_bbox = None
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        if encoder_outputs is None:
            image_features, pool_image_attn = self.encode_images_wocr(images, ocr_ids, ocr_attn, ocr_bbox)
            encoder_outputs = BaseModelOutputWithVisionEmbeds(
                last_hidden_state=image_features,
                attention_mask=pool_image_attn,
            )
        else:
            image_features = encoder_outputs.last_hidden_state
            pool_image_attn = encoder_outputs.attention_mask
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        if decoder_input_ids is None and labels is None:
            return encoder_outputs
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=image_features,
            encoder_attention_mask=pool_image_attn,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ):
        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(encoder, "_hf_hook"):
            encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)

        return model_kwargs
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = LayoutLMv3Model.from_pretrained(model_args.vision_tower)
            vision_tower.image_processor = AutoProcessor.from_pretrained(model_args.vision_tower, apply_ocr=False)
            if 'base' in model_args.vision_tower:
                self.mm_projector = nn.Linear(768, 1024)
            elif 'large' in model_args.vision_tower:
                self.mm_projector = nn.Linear(1024, 1024)
            else:
                raise NotImplementedError

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = 1024
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type





AutoConfig.register("llava_phi", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaPhiForCausalLM)