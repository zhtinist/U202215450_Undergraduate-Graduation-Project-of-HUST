import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple
from dataclasses import dataclass

import torch
from torch import nn
from torch import Tensor
import math
from transformers import T5Config, T5PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.modeling_t5 import T5Block, T5ForConditionalGeneration, T5LayerNorm

from llava.model.multimodal_encoder.udop_relative_pos import (
    RelativePositionBias1D,
    RelativePositionBiasAggregated,
    RelativePositionBiasBase,
    create_relative_bias,
)
from llava.model.multimodal_encoder.mae import mae_model
from transformers import AutoTokenizer, AutoConfig
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import copy
import torch.nn.functional as F
logger = logging.getLogger(__name__)
DEFAULT_T5_TOKENIZER_PATH = os.environ.get("LECSLIDES_T5_TOKENIZER_PATH", "/home/emzhang/data/t5-large")
def _get_clone(module):
    return copy.deepcopy(module)


class GraphAttentionLayer(nn.Module):
    def __init__(self, num_attention_heads=16, hidden_size=1024, hidden_dropout_prob=0.0):
        super(GraphAttentionLayer, self).__init__()
        self.num_attention_heads = int(num_attention_heads / 2)
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.final = nn.Linear(hidden_size, self.all_head_size)
        self.init_weights()
        with torch.no_grad():
            nn.init.zeros_(self.final.weight)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            seq_inputs,
            graph_mask,
    ):
        mixed_query_layer = self.query(seq_inputs)

        key_layer = self.transpose_for_scores(self.key(seq_inputs))
        value_layer = self.transpose_for_scores(self.value(seq_inputs))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + graph_mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = self.final(context_layer)


        return outputs


class SubLayerConnection(nn.Module):
    def __init__(self, hidden_size=1024, hidden_dropout_prob=0.0):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-05)
        self.dropout = nn.Dropout(p=hidden_dropout_prob)
        self.size = hidden_size
        # self.lamda = torch.nn.Parameter(torch.zeros(1))
        self.init_weights()

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x, graph_mask, sublayer):
        # print(self.lamda)
        # return (1 - self.lamda) * x + self.lamda * self.dropout(sublayer(self.norm(x), graph_mask))
        # return x + self.dropout(sublayer(self.norm(x), graph_mask))
        return x + self.dropout(self.norm(sublayer(x, graph_mask)))
        # return self.norm(x + self.dropout(sublayer(x, graph_mask)))

class CellEmbeddings(nn.Module):
    def __init__(self, max_2d_position_embeddings=501, hidden_size=1024, ccat=False):
        super(CellEmbeddings, self).__init__()
        self.ccat = ccat
        self.max_2d_position_embeddings = max_2d_position_embeddings
        if ccat:
            self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size // 4)
            self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size // 4)
        else:
            self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
            self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)

    def forward(self, bbox):
        bbox = torch.clip(bbox, 0.0, 1.0)
        bbox = (bbox * (self.max_2d_position_embeddings-1)).long()
        bbox[bbox > (self.max_2d_position_embeddings-1)] = self.max_2d_position_embeddings-1

        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        if self.ccat:
            embeddings = torch.cat(
                [
                    left_position_embeddings,
                    upper_position_embeddings,
                    right_position_embeddings,
                    lower_position_embeddings
                ],
                dim=-1)
        else:
            embeddings = (
                left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
            )

        return embeddings

def pad_sequence(seq, target_len, pad_value=0):
    if isinstance(seq, torch.Tensor):
        n = seq.shape[0]
    else:
        n = len(seq)
        seq = torch.tensor(seq)
    m = target_len - n
    if m > 0:
        ret = torch.stack([pad_value] * m).to(seq)
        seq = torch.cat([seq, ret], dim=0)
    return seq[:target_len]


def collate_vlembed(inputs_patches, inputs_embeds, seg_data, visual_segdata, vis_special_token=None,
                    attention_mask=None, num_patches=14, max_len=0):
    L = num_patches
    ocr_points_x = torch.clip(torch.floor((seg_data[:, :, 0] + seg_data[:, :, 2]) / 2.0 * L).long(), 0, L - 1)
    ocr_points_y = torch.clip(torch.floor((seg_data[:, :, 1] + seg_data[:, :, 3]) / 2.0 * L).long(), 0, L - 1) * L
    ocr_points = ocr_points_x + ocr_points_y
    target_seg = (seg_data.mean(-1) == 0.0) | (seg_data.mean(-1) == 1.0)
    repeated_vision_embeds = torch.gather(inputs_patches, 1,
                                          ocr_points.unsqueeze(-1).repeat(1, 1, inputs_patches.size(-1)))
    repeated_vision_embeds[target_seg] = 0.0
    if repeated_vision_embeds.size(-1) != inputs_embeds.size(-1):
        dim_diff = inputs_embeds.size(-1) - repeated_vision_embeds.size(-1)
        if dim_diff > 0:
            repeated_vision_embeds = torch.nn.functional.pad(repeated_vision_embeds, (0, dim_diff))
        else:
            repeated_vision_embeds = repeated_vision_embeds[..., :inputs_embeds.size(-1)]
    inputs_embeds += repeated_vision_embeds

    patch_inds = torch.full_like(inputs_patches[:, :, 0], True).bool()
    ind = torch.cat([torch.arange(len(ocr_points))[:, None].repeat(1, ocr_points.size(-1))[:, :, None].to(ocr_points),
                     ocr_points[:, :, None]], -1).flatten(0, 1)
    rows, cols = zip(*ind)
    patch_inds[rows, cols] = False

    input_vision_patches = [inputs_patches[i][patch_inds[i]] for i in range(len(patch_inds))]
    visual_segdata = [visual_segdata[i][patch_inds[i]] for i in range(len(patch_inds))]
    if attention_mask is not None:
        visual_attention_mask = [torch.tensor([1] * len(item)).to(attention_mask) for item in visual_segdata]

    if max_len == 0:
        max_len = inputs_patches.size(1)
    else:
        max_len = max_len - inputs_embeds.size(1)
    inputs_vision_patches = torch.stack(
        [pad_sequence(item, max_len, torch.zeros_like(inputs_patches[0, 0])) for item in input_vision_patches])
    if inputs_vision_patches.size(-1) != inputs_embeds.size(-1):
        dim_diff = inputs_embeds.size(-1) - inputs_vision_patches.size(-1)
        if dim_diff > 0:
            inputs_vision_patches = torch.nn.functional.pad(inputs_vision_patches, (0, dim_diff))
        else:
            inputs_vision_patches = inputs_vision_patches[..., :inputs_embeds.size(-1)]
    visual_segdata = torch.stack(
        [pad_sequence(item, max_len, torch.zeros_like(seg_data[0, 0])) for item in visual_segdata])
    if attention_mask is not None:
        visual_attention_mask = torch.stack(
            [pad_sequence(item, max_len, torch.zeros_like(attention_mask[0, 0])) for item in visual_attention_mask])

    if vis_special_token is not None:
        if vis_special_token.size(-1) != inputs_vision_patches.size(-1):
            dim_diff = inputs_vision_patches.size(-1) - vis_special_token.size(-1)
            if dim_diff > 0:
                vis_special_token = torch.nn.functional.pad(vis_special_token, (0, dim_diff))
            else:
                vis_special_token = vis_special_token[..., :inputs_vision_patches.size(-1)]
        inputs_vision_patches += vis_special_token

    inputs_embeds = torch.cat([inputs_embeds, inputs_vision_patches], 1)
    seg_data = torch.cat([seg_data, visual_segdata], 1)
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, visual_attention_mask], 1)
    return inputs_embeds, seg_data, attention_mask

def iou(box1, box2):
    """
    计算两个边界框(box)的交并比(IOU)。

    参数:
    box1, box2 -- 边界框，格式为[x1, y1, x2, y2]，其中(x1, y1)是左上角坐标，(x2, y2)是右下角坐标。

    返回:
    iou -- 两个边界框的IoU值。
    """
    # 计算两个边界框的交集坐标
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    # 计算交集的宽度和高度
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)

    # 如果交集的面积为0，则IoU也为0
    if inter_width == 0 or inter_height == 0:
        return 0.0

    # 计算交集的面积
    intersection = inter_width * inter_height

    # 计算两个边界框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集的面积
    union = box1_area + box2_area - intersection

    # 计算IoU
    iou = intersection / union

    return iou

def check_ocr_element(element_pos, ocr_pos, mode='inside'):
    # import ipdb;ipdb.set_trace()
    # element_pos = torch.tensor(element_pos).to(device=ocr_pos.device)
    if mode == 'iou':
        flag = iou(ocr_pos,element_pos) > 0.6
    elif mode == 'inside':
        ct_x = (ocr_pos[0] + ocr_pos[2]) / 2
        ct_y = (ocr_pos[1] + ocr_pos[3]) / 2
        flag = ct_x > element_pos[0] and ct_x < element_pos[2] and ct_y > element_pos[1] and ct_y < element_pos[3]
    else:
        raise NotImplementedError

    return flag

def collate_vlembed_wElement(element_patches, element_data, inputs_patches, inputs_embeds, seg_data, visual_segdata, vis_special_token=None,
                    attention_mask=None, num_patches=14, max_len=0):
    L = num_patches
    ocr_points_x = torch.clip(torch.floor((seg_data[:, :, 0] + seg_data[:, :, 2]) / 2.0 * L).long(), 0, L - 1)
    ocr_points_y = torch.clip(torch.floor((seg_data[:, :, 1] + seg_data[:, :, 3]) / 2.0 * L).long(), 0, L - 1) * L
    ocr_points = ocr_points_x + ocr_points_y
    target_seg = (seg_data.mean(-1) == 0.0) | (seg_data.mean(-1) == 1.0)
    repeated_vision_embeds = torch.gather(inputs_patches, 1,
                                          ocr_points.unsqueeze(-1).repeat(1, 1, inputs_patches.size(-1)))
    repeated_vision_embeds[target_seg] = 0.0
    element_ocr = torch.zeros(inputs_embeds.shape[0],inputs_embeds.shape[1]).bool()
    for bs, (element_patch, element_data_, seg_data_) in enumerate(zip(element_patches, element_data, seg_data)):
        if element_patch is not None:
            for ocr_id, ocr_pos in enumerate(seg_data_):
                for ele_id, element_pos in enumerate(element_data_):
                    if check_ocr_element(element_pos, ocr_pos):
                        repeated_vision_embeds[bs,ocr_id] = element_patch[ele_id]
                        element_ocr[bs,ocr_id] = True
                        break

    element_max_len = max([len(patch) if patch is not None else 0 for patch in element_patches])

    inputs_embeds += repeated_vision_embeds

    patch_inds = torch.full_like(inputs_patches[:, :, 0], True).bool()
    ind = torch.cat([torch.arange(len(ocr_points))[:, None].repeat(1, ocr_points.size(-1))[:, :, None].to(ocr_points),
                     ocr_points[:, :, None]], -1).flatten(0, 1)
    rows, cols = zip(*ind)
    patch_inds[rows, cols] = False

    input_vision_patches = [inputs_patches[i][patch_inds[i]] for i in range(len(patch_inds))]
    visual_segdata = [visual_segdata[i][patch_inds[i]] for i in range(len(patch_inds))]

    for i in range(len(input_vision_patches)):
        if element_patches[i] is not None:
            input_vision_patches[i] = torch.cat([input_vision_patches[i], element_patches[i]], dim=0)
            cur_pos = torch.tensor(element_data[i]).to(dtype=visual_segdata[i].dtype, device=visual_segdata[i].device)
            visual_segdata[i] = torch.cat([visual_segdata[i], cur_pos], dim=0)
    # print(element_data)

    if attention_mask is not None:
        visual_attention_mask = [torch.tensor([1] * len(item)).to(attention_mask) for item in visual_segdata]

    if max_len == 0:
        max_len = inputs_patches.size(1) + element_max_len
    else:
        max_len = max_len - inputs_embeds.size(1)
    inputs_vision_patches = torch.stack(
        [pad_sequence(item, max_len, torch.zeros_like(inputs_patches[0, 0])) for item in input_vision_patches])
    visual_segdata = torch.stack(
        [pad_sequence(item, max_len, torch.zeros_like(seg_data[0, 0])) for item in visual_segdata])
    if attention_mask is not None:
        visual_attention_mask = torch.stack(
            [pad_sequence(item, max_len, torch.zeros_like(attention_mask[0, 0])) for item in visual_attention_mask])

    if vis_special_token is not None:
        inputs_vision_patches += vis_special_token

    inputs_embeds = torch.cat([inputs_embeds, inputs_vision_patches], 1)
    seg_data = torch.cat([seg_data, visual_segdata], 1)
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, visual_attention_mask], 1)
    return inputs_embeds, seg_data, attention_mask

@dataclass
class BaseModelOutputWithVisionEmbeds(BaseModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_embeds: torch.FloatTensor = None
    attention_mask: torch.FloatTensor = None
    seg_data: torch.FloatTensor = None
    predict_relation: torch.FloatTensor = None


@dataclass
class VisSeq2SeqLMOutput(BaseModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    image_output: Optional[Tuple[torch.FloatTensor]] = None
    image_target: Optional[Tuple[torch.FloatTensor]] = None
    image_mask_label: Optional[Tuple[torch.FloatTensor]] = None


class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual


class T52dStack(T5PreTrainedModel):
    """
    Almost exact copy of transformers T5Stack with the modification
    of passing `position_bias` in the forward method
    """

    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self._max_length = config.max_length

        setattr(config, 'output_attentions', True)
        if self.is_decoder:
            dec_trunc = getattr(config, "truncate_decoder_after_layer", None)
            self.num_layers = (
                dec_trunc if dec_trunc else config.num_layers
            )
        else:
            enc_trunc = getattr(config, "truncate_encoder_after_layer", None)
            self.num_layers = (
                enc_trunc if enc_trunc else config.num_layers
            )

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(self.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.dropout = nn.Dropout(config.dropout_rate)

        if not self.is_decoder:
            self.cell2dembedding = CellEmbeddings(config.max_2d_position_embeddings, config.hidden_size)

        # get weights from encoder position bias
        self.relative_bias = self._get_relative_bias(config)

        # tie weights of original position bias of encoder
        for bias in self.relative_bias.biases:
            if isinstance(bias, RelativePositionBias1D):
                self._tie_or_clone_weights(
                    bias.relative_attention_bias, self.block[0].layer[0].SelfAttention.relative_attention_bias
                )

        self.init_weights()

    @staticmethod
    def _get_relative_bias(config: T5Config) -> RelativePositionBiasAggregated:
        relative_bias_list = create_relative_bias(config)
        return RelativePositionBiasAggregated(relative_bias_list)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            ids_keep=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cross_attn_head_mask=None,
            position_bias=None,  # modified line,
            inputs_patches=None,  # modified line,
            seg_data=None,  # modified line,
            visual_seg_data=None,  # modified line,
            num_patches=None,  # modified line,
            special_vis_token=None,  # modified line,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = True  # False #True #output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ======================================================
        # input embeddings processing

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None and torch.numel(input_ids) > 0:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is None and input_ids is not None and torch.numel(input_ids) == 0:
            input_ids = torch.full((4, 1024), self.config.pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
            attention_mask = torch.zeros((4, 1024), device=input_ids.device, dtype=input_ids.dtype)
            seg_data = torch.zeros((4, 1024, 4), device=input_ids.device, dtype=input_ids.dtype)
            input_shape = input_ids.size()
            position_bias = torch.zeros_like(
                self.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)
            )
            # encoder_attention_mask = attention_mask
            logger.warning('Empty batch')
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        if inputs_patches is not None:
            # ===========================
            # combine OCR text and visual embed
            inputs_embeds, seg_data, attention_mask = collate_vlembed(inputs_patches, inputs_embeds, seg_data,
                                                                      visual_seg_data, special_vis_token,
                                                                      attention_mask, num_patches, 0)
            input_shape = inputs_embeds.size()[:-1]

        if not self.is_decoder:
            inputs_embeds += self.cell2dembedding(seg_data)

        batch_size, seq_length = input_shape

        # ======================================================
        # input masking/pos embed processing

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(self)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        if self.is_decoder:  # modified lines
            position_bias = None
        else:
            position_bias = self.relative_bias(
                attention_mask=attention_mask, seg_data=seg_data
            )
            position_bias = position_bias + extended_attention_mask
        encoder_decoder_position_bias = None

        # ======================================================
        # model inferencing

        hidden_states = inputs_embeds

        hidden_states = self.dropout(hidden_states)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            if use_cache is False:  # MP fixes
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithVisionEmbeds(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            attention_mask=attention_mask,
            seg_data=seg_data,
        )

class T52dStackOrder(T52dStack):
    def __init__(self, config, embed_tokens=None):
        super(T52dStackOrder,self).__init__(config,embed_tokens=embed_tokens)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self._max_length = config.max_length

        setattr(config, 'output_attentions', True)
        if self.is_decoder:
            dec_trunc = getattr(config, "truncate_decoder_after_layer", None)
            self.num_layers = (
                dec_trunc if dec_trunc else config.num_layers
            )
        else:
            enc_trunc = getattr(config, "truncate_encoder_after_layer", None)
            self.num_layers = (
                enc_trunc if enc_trunc else config.num_layers
            )

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(self.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.dropout = nn.Dropout(config.dropout_rate)

        if not self.is_decoder:
            self.cell2dembedding = CellEmbeddings(config.max_2d_position_embeddings, config.hidden_size)

        # get weights from encoder position bias
        self.relative_bias = self._get_relative_bias(config)

        # tie weights of original position bias of encoder
        for bias in self.relative_bias.biases:
            if isinstance(bias, RelativePositionBias1D):
                self._tie_or_clone_weights(
                    bias.relative_attention_bias, self.block[0].layer[0].SelfAttention.relative_attention_bias
                )

        self.init_weights()
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            ids_keep=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cross_attn_head_mask=None,
            position_bias=None,  # modified line,
            inputs_patches=None,  # modified line,
            seg_data=None,  # modified line,
            visual_seg_data=None,  # modified line,
            num_patches=None,  # modified line,
            element_patches=None,
            element_data=None,
            special_vis_token=None,  # modified line,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = True  # False #True #output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ======================================================
        # input embeddings processing

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None and torch.numel(input_ids) > 0:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is None and input_ids is not None and torch.numel(input_ids) == 0:
            input_ids = torch.full((4, 1024), self.config.pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
            attention_mask = torch.zeros((4, 1024), device=input_ids.device, dtype=input_ids.dtype)
            seg_data = torch.zeros((4, 1024, 4), device=input_ids.device, dtype=input_ids.dtype)
            input_shape = input_ids.size()
            position_bias = torch.zeros_like(
                self.get_extended_attention_mask(attention_mask, input_shape, attention_mask.device)
            )
            # encoder_attention_mask = attention_mask
            logger.warning('Empty batch')
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

            # reorder input sequence

        if inputs_patches is not None:
            # ===========================
            # combine OCR text and visual embed
            inputs_embeds, seg_data, attention_mask = collate_vlembed_wElement(element_patches, element_data, inputs_patches, inputs_embeds, seg_data,
                                                                      visual_seg_data, special_vis_token,
                                                                      attention_mask, num_patches, 0)
            input_shape = inputs_embeds.size()[:-1]

        if not self.is_decoder:
            inputs_embeds += self.cell2dembedding(seg_data)

        batch_size, seq_length = input_shape

        # ======================================================
        # input masking/pos embed processing

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(self)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None

        if self.is_decoder:  # modified lines
            position_bias = None
        else:
            position_bias = self.relative_bias(
                attention_mask=attention_mask, seg_data=seg_data
            )
            position_bias = position_bias + extended_attention_mask
        encoder_decoder_position_bias = None

        # ======================================================
        # model inferencing

        hidden_states = inputs_embeds

        hidden_states = self.dropout(hidden_states)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            if use_cache is False:  # MP fixes
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)

            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithVisionEmbeds(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            attention_mask=attention_mask,
            seg_data=seg_data,
        )

class UdopUnimodelForConditionalGeneration(T5ForConditionalGeneration):
    """
    Copied from original T5ForConditionalGeneration class with signature extended with 2D data.
    :param config: a `T5Config` instance
    """

    def __init__(self, config):
        super(UdopUnimodelForConditionalGeneration, self).__init__(config)

        # get max length of decoder part, for T5 decoder lenght depends
        # on the task and it can be modified by passing `_max_decoder_length` to the model/config
        self._max_decoder_length = config.max_decoder_length if hasattr(config, "max_decoder_length") else 256
        # Backward-compatible defaults for checkpoints without custom UDOP fields.
        if not hasattr(config, "max_2d_position_embeddings"):
            setattr(config, "max_2d_position_embeddings", 1024)
        if not hasattr(config, "mae_version"):
            # Infer MAE backbone size from hidden size to avoid
            # loading a base-patch vision tower for large UDOP checkpoints.
            inferred_mae = "mae_vit_large_patch16" if getattr(config, "d_model", 768) >= 1024 else "mae_vit_base_patch16"
            setattr(config, "mae_version", inferred_mae)
        if not hasattr(config, "mae_checkpoint"):
            setattr(config, "mae_checkpoint", "")
        if not hasattr(config, "image_size"):
            setattr(config, "image_size", 224)

        self.config.decoder_start_token_id = self.config.pad_token_id

        self.encoder = T52dStack(self.encoder.config, self.shared)
        self.decoder = T52dStack(self.decoder.config, self.shared)

        self.init_weights()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        mae_model_tmp = mae_model(config.mae_version, config.mae_checkpoint, config.image_size, config.vocab_size,
                                  config.max_2d_position_embeddings)

        self.patch_embed = mae_model_tmp.patch_embed
        self.embed_dim = mae_model_tmp.embed_dim
        self.pos_embed = mae_model_tmp.pos_embed
        self.special_vis_token = mae_model_tmp.special_vis_token

    @staticmethod
    def get_required_segment_levels() -> Sequence[str]:
        return ["tokens"]

    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, RelativePositionBiasBase):
            factor = self.config.initializer_factor
            d_model = self.config.d_model
            module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Tensor = None,
            decoder_input_ids: Optional[Tensor] = None,
            decoder_attention_mask: Optional[Tensor] = None,
            encoder_outputs: Optional[Tensor] = None,
            past_key_values: Optional[Tensor] = None,
            image: Optional[Tensor] = None,
            ids_keep: Optional[Tensor] = None,
            ids_restore: Optional[Tensor] = None,
            image_mask_label: Optional[Tensor] = None,
            mask_ratio: Optional[Tensor] = None,
            seg_data: Dict[str, Any] = None,
            visual_seg_data: Dict[str, Any] = None,
            masked_lm_labels: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            char_ids: Optional[Tensor] = None,
            char_seg_data: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            decoder_inputs_embeds: Optional[Tensor] = None,
            decoder_head_mask: Optional[Tensor] = None,
            cross_attn_head_mask: Optional[Tensor] = None,
            use_cache=True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            input_dict: Dict[str, Any] = None,
            **kwargs,
    ) -> Tuple[Tensor, ...]:

        if input_dict is not None:
            return_task_outputs = []
            for task in input_dict:
                return_task_outputs.append(self.forward(**input_dict[task]))
            return return_task_outputs

        if encoder_outputs is None:
            inputs_patches = None
            if image is not None:
                assert visual_seg_data is not None
                target_dtype = self.patch_embed.proj.bias.dtype if self.patch_embed.proj.bias is not None else self.patch_embed.proj.weight.dtype
                if image.dtype != target_dtype:
                    image = image.to(dtype=target_dtype)
                x = self.patch_embed(image)
                num_patches = image.size(2) // 16
                if ids_keep is not None:
                    x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(-1)))
                    pad_tokens = self.pad_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
                    x_padded = torch.cat([x, pad_tokens], dim=1)  # no cls token
                    x_padded = torch.gather(x_padded, dim=1,
                                            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_padded.shape[2]))
                    inputs_patches = x_padded
                else:
                    inputs_patches = x

            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                seg_data=seg_data,
                visual_seg_data=visual_seg_data,
                inputs_patches=inputs_patches,
                num_patches=num_patches,
                special_vis_token=self.special_vis_token,
                ids_keep=ids_keep,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if encoder_outputs is None:
            return None

        if masked_lm_labels is not None and labels is None:
            labels = masked_lm_labels

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        # ugly hack for model to work as an encoder
        if decoder_input_ids is None and masked_lm_labels is None:
            return encoder_outputs

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=encoder_outputs.attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs  # type: ignore

    def get_encoder(self):
        return self

class UdopUnimodelForConditionalGenerationOrder(UdopUnimodelForConditionalGeneration):
    """
    Copied from original T5ForConditionalGeneration class with signature extended with 2D data.
    :param config: a `T5Config` instance
    """

    def __init__(self, config):
        super(UdopUnimodelForConditionalGenerationOrder, self).__init__(config)

        # get max length of decoder part, for T5 decoder lenght depends
        # on the task and it can be modified by passing `_max_decoder_length` to the model/config
        self._max_decoder_length = config.max_decoder_length if hasattr(config, "max_decoder_length") else 256

        self.config.decoder_start_token_id = self.config.pad_token_id

        self.encoder = T52dStackOrder(self.encoder.config, self.shared)
        self.decoder = T52dStackOrder(self.decoder.config, self.shared)

        self.init_weights()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        mae_model_tmp = mae_model(config.mae_version, config.mae_checkpoint, config.image_size, config.vocab_size,
                                  config.max_2d_position_embeddings)

        self.patch_embed = mae_model_tmp.patch_embed
        self.embed_dim = mae_model_tmp.embed_dim
        self.pos_embed = mae_model_tmp.pos_embed
        self.special_vis_token = mae_model_tmp.special_vis_token

class UDOPDecoder(nn.Module):
    def __init__(self, config):
        super(UDOPDecoder,self).__init__()
        udop = UdopUnimodelForConditionalGeneration.from_pretrained(config)
        self.decoder = udop.decoder

class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class UDOPEncoderOrder(nn.Module):
    def __init__(self, config):
        super(UDOPEncoderOrder, self).__init__()
        udop = UdopUnimodelForConditionalGenerationOrder.from_pretrained(
            config,
            ignore_mismatched_sizes=True,
        )

        self.encoder = udop.encoder
        self.patch_embed = udop.patch_embed
        self.patch_embed = self.patch_embed.float()
        self.embed_dim = udop.embed_dim
        self.pos_embed = udop.pos_embed
        self.special_vis_token = udop.special_vis_token
        self.image_processor = AutoTokenizer.from_pretrained(DEFAULT_T5_TOKENIZER_PATH)

        self.element_embed = _get_clone(self.patch_embed)

        self.graph_attention_layer = GraphAttentionLayer()
        self.sublayer = SubLayerConnection()
        self.graph_mask_q = nn.Linear(1024,1024)
        self.graph_mask_k = nn.Linear(1024,1024)
        self.dist_mlp = Mlp(in_features=6,hidden_features=512,out_features=1)
        # import ipdb;ipdb.set_trace()

    def get_dynamic_graph_mask(self, text_embeds, text_dist_graph, graph_shape, thr=0.5, theta=2.5):
        text_q = self.graph_mask_q(text_embeds)
        text_k = self.graph_mask_k(text_embeds)
        attn_mask = -9e8 * torch.ones(text_embeds.shape[0],text_embeds.shape[1],text_embeds.shape[1]).to(dtype=text_embeds.dtype, device=text_embeds.device)
        for bs in range(text_embeds.shape[0]):
            attn_mask[bs,:graph_shape[bs],:graph_shape[bs]] = 0
        text_scores = torch.matmul(text_q, text_k.transpose(-1, -2)) / math.sqrt(1024)
        text_dist_graph = text_dist_graph.to(device=text_embeds.device,dtype=text_embeds.dtype)
        text_scores += self.dist_mlp(text_dist_graph).squeeze(-1)
        # text_eye = -9e8 * torch.eye(text_embeds.shape[1]).to(dtype=text_scores.dtype, device=text_scores.device)

        text_scores_wattn = text_scores + attn_mask
        texts_probs = text_scores_wattn.sigmoid()


        text_mask = -9e8 * torch.ones((text_embeds.shape[0], text_embeds.shape[1],text_embeds.shape[1])).to(dtype=texts_probs.dtype,device=texts_probs.device)

        # topk_score,_ = torch.topk(texts_probs.flatten(0),10)
        # if thr > topk_score[-1]:
        #     thr = topk_score[-1]
        text_mask[texts_probs > thr] = 0
        graph_mask = texts_probs * theta + text_mask

        return graph_mask, text_scores



    @staticmethod
    def get_required_segment_levels() -> Sequence[str]:
        return ["tokens"]

    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, RelativePositionBiasBase):
            factor = self.config.initializer_factor
            d_model = self.config.d_model
            module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def crop_element(self,image,pos,patch_size=16):
        h, w = image.shape[-2:]
        x1, y1, x2, y2 = int(pos[0] * w), int(pos[1] * h), int(pos[2] * w), int(pos[3] * h)
        crop_image = image[:, y1:y2, x1:x2]
        crop_image = F.interpolate(crop_image.unsqueeze(0).to(dtype=torch.float, device=image.device), size=(patch_size,patch_size), mode='bilinear').squeeze(0).to(dtype=image.dtype, device=image.device)
        return crop_image


    def get_element_embeds(self,image,element_data):
        element_pool = []
        element_num = []

        for img, element_pos in zip(image,element_data):
            if element_pos is None:
                element_num.append(None)
                continue
            element_num.append(len(element_pos))
            for pos in element_pos:
                element_pool.append(self.crop_element(img,pos))
        if len(element_pool) > 0:
            element_imgs = torch.stack(element_pool,dim=0).to(dtype=image.dtype, device=image.device)
            element_embeds = self.element_embed.proj(element_imgs)
            element_embeds = element_embeds.flatten(2).transpose(1, 2)
            element_embeds = self.element_embed.norm(element_embeds).squeeze(1)
        res_pool = []
        prev_num = 0
        for num in element_num:
            if num is None:
                res_pool.append(None)
                continue
            res_pool.append(element_embeds[prev_num:prev_num+num,...])
            prev_num += num
        return res_pool

    def get_relative_pos(self,box1, box2):
        box1_ctr_x = (box1[0] + box1[2]) / 2
        box1_ctr_y = (box1[1] + box1[3]) / 2
        box2_ctr_x = (box2[0] + box2[2]) / 2
        box2_ctr_y = (box2[1] + box2[3]) / 2
        w_dist = (box1_ctr_x - box2_ctr_x) / (box1[2] - box1[0])
        h_dist = (box1_ctr_y - box2_ctr_y) / (box1[3] - box1[1])
        return [w_dist, h_dist]

    def get_union_box(self,box1,box2):
        return [min(box1[0],box2[0]),min(box1[1],box2[1]),max(box1[2],box2[2]),max(box1[3],box2[3])]

    def get_dist_graph(self, seg_data):
        bs, text_len, _ = seg_data.shape
        matrix = torch.zeros(bs,text_len,text_len,6).to(dtype=seg_data.dtype,device=seg_data.device)
        for bs in range(seg_data.shape[0]):
            cur_seg_data = seg_data[bs]
            for i in range(seg_data.shape[1]):
                for j in range(seg_data.shape[1]):
                    if (cur_seg_data[i] == cur_seg_data[j]).all():
                        matrix[bs,i,j] = torch.zeros(6).to(dtype=seg_data.dtype,device=seg_data.device)
                        matrix[bs, i, j] = torch.exp(matrix[bs,i,j])
                        continue
                    if cur_seg_data[i].mean() in [0.0,1.0] or cur_seg_data[j].mean() in [0.0,1.0]:
                        continue
                    cur_dist_list = []
                    union_box = self.get_union_box(cur_seg_data[i], cur_seg_data[j])
                    cur_dist_list.extend(self.get_relative_pos(cur_seg_data[i],cur_seg_data[j]))
                    cur_dist_list.extend(self.get_relative_pos(cur_seg_data[i],union_box))
                    cur_dist_list.extend(self.get_relative_pos(cur_seg_data[j],union_box))
                    matrix[bs, i, j] = torch.tensor(cur_dist_list).to(dtype=seg_data.dtype,device=seg_data.device)
                    matrix[bs, i, j] = torch.exp(matrix[bs,i,j])
        return matrix



    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Tensor = None,
            decoder_input_ids: Optional[Tensor] = None,
            decoder_attention_mask: Optional[Tensor] = None,
            encoder_outputs: Optional[Tensor] = None,
            past_key_values: Optional[Tensor] = None,
            image: Optional[Tensor] = None,
            ids_keep: Optional[Tensor] = None,
            ids_restore: Optional[Tensor] = None,
            image_mask_label: Optional[Tensor] = None,
            mask_ratio: Optional[Tensor] = None,
            seg_data: Dict[str, Any] = None,
            visual_seg_data: Dict[str, Any] = None,
            element_data: Dict[str, Any] = None,
            text_dist_graph: Dict[str, Any] = None,
            graph_shape: Dict[str, Any] = None,
            relation_label: Dict[str, Any] = None,
            masked_lm_labels: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            char_ids: Optional[Tensor] = None,
            char_seg_data: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            decoder_inputs_embeds: Optional[Tensor] = None,
            decoder_head_mask: Optional[Tensor] = None,
            cross_attn_head_mask: Optional[Tensor] = None,
            use_cache=True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            input_dict: Dict[str, Any] = None,
            question: Tensor = None,
            **kwargs,
    ) -> Tuple[Tensor, ...]:

        if input_dict is not None:
            return_task_outputs = []
            for task in input_dict:
                return_task_outputs.append(self.forward(**input_dict[task]))
            return return_task_outputs

        if question is not None:
            question_len = question.shape[1]
            question_seg_data = torch.zeros(seg_data.shape[0],question_len,seg_data.shape[2]).to(device=seg_data.device, dtype=seg_data.dtype)
            input_ids = torch.cat([question, input_ids],dim=1).to(device=input_ids.device, dtype=input_ids.dtype)
            seg_data = torch.cat([question_seg_data, seg_data],dim=1).to(device=seg_data.device, dtype=seg_data.dtype)
            question_attention_mask = torch.ones(attention_mask.shape[0], question_len).to(device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([question_attention_mask, attention_mask], dim=1).to(device=attention_mask.device, dtype=attention_mask.dtype)

        if encoder_outputs is None:
            inputs_patches = None
            if image is not None:
                assert visual_seg_data is not None
                x = self.patch_embed(image)
                num_patches = image.size(2) // 16
                if ids_keep is not None:
                    x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(-1)))
                    pad_tokens = self.pad_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
                    x_padded = torch.cat([x, pad_tokens], dim=1)  # no cls token
                    x_padded = torch.gather(x_padded, dim=1,
                                            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_padded.shape[2]))
                    inputs_patches = x_padded
                else:
                    inputs_patches = x


            # Convert encoder inputs in embeddings if needed
            element_patches = self.get_element_embeds(image,element_data)
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                seg_data=seg_data,
                visual_seg_data=visual_seg_data,
                inputs_patches=inputs_patches,
                num_patches=num_patches,
                element_patches=element_patches,
                element_data=element_data,
                special_vis_token=self.special_vis_token,
                ids_keep=ids_keep,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        if question is not None:
            question_output = encoder_outputs.last_hidden_state[:,:question_len,:]
            seq_output = encoder_outputs.last_hidden_state[:,question_len:seg_data.shape[1],:]
            vision_output = encoder_outputs.last_hidden_state[:, seg_data.shape[1]:, :]
        else:
            seq_output = encoder_outputs.last_hidden_state[:,:seg_data.shape[1],:]
            vision_output = encoder_outputs.last_hidden_state[:,seg_data.shape[1]:,:]



        graph_mask, relation_prediction = self.get_dynamic_graph_mask(seq_output,text_dist_graph, graph_shape)
        graph_seq_output =self.sublayer(seq_output,graph_mask,self.graph_attention_layer)
        if question is not None:
            new_feat = torch.cat([question_output, graph_seq_output, vision_output], dim=1)
        else:
            new_feat = torch.cat([graph_seq_output, vision_output], dim=1)
        assert (
                    encoder_outputs.last_hidden_state.shape == new_feat.shape), f'something wrong, shape {encoder_outputs.last_hidden_state.shape} vs {new_feat.shape}.'
        encoder_outputs.last_hidden_state = new_feat
        encoder_outputs.predict_relation = relation_prediction





        return encoder_outputs

    def get_encoder(self):
        return self

class UDOPEncoder(nn.Module):
    def __init__(self, config):
        super(UDOPEncoder,self).__init__()
        udop = UdopUnimodelForConditionalGeneration.from_pretrained(
            config,
            ignore_mismatched_sizes=True,
        )

        self.encoder = udop.encoder
        self.patch_embed = udop.patch_embed
        self.patch_embed = self.patch_embed.float()
        self.embed_dim = udop.embed_dim
        self.pos_embed = udop.pos_embed
        self.special_vis_token = udop.special_vis_token
        self.image_processor = AutoTokenizer.from_pretrained(DEFAULT_T5_TOKENIZER_PATH)

    @staticmethod
    def get_required_segment_levels() -> Sequence[str]:
        return ["tokens"]

    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, RelativePositionBiasBase):
            factor = self.config.initializer_factor
            d_model = self.config.d_model
            module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def forward(
            self,
            input_ids: Tensor = None,
            attention_mask: Tensor = None,
            decoder_input_ids: Optional[Tensor] = None,
            decoder_attention_mask: Optional[Tensor] = None,
            encoder_outputs: Optional[Tensor] = None,
            past_key_values: Optional[Tensor] = None,
            image: Optional[Tensor] = None,
            ids_keep: Optional[Tensor] = None,
            ids_restore: Optional[Tensor] = None,
            image_mask_label: Optional[Tensor] = None,
            mask_ratio: Optional[Tensor] = None,
            seg_data: Dict[str, Any] = None,
            visual_seg_data: Dict[str, Any] = None,
            masked_lm_labels: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            char_ids: Optional[Tensor] = None,
            char_seg_data: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            decoder_inputs_embeds: Optional[Tensor] = None,
            decoder_head_mask: Optional[Tensor] = None,
            cross_attn_head_mask: Optional[Tensor] = None,
            use_cache=True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            input_dict: Dict[str, Any] = None,
            question: Tensor = None,
            **kwargs,
    ) -> Tuple[Tensor, ...]:

        if input_dict is not None:
            return_task_outputs = []
            for task in input_dict:
                return_task_outputs.append(self.forward(**input_dict[task]))
            return return_task_outputs

        if question is not None:
            question_len = question.shape[1]
            question_seg_data = torch.zeros(seg_data.shape[0],question_len,seg_data.shape[2]).to(device=seg_data.device, dtype=seg_data.dtype)
            input_ids = torch.cat([question, input_ids],dim=1).to(device=input_ids.device, dtype=input_ids.dtype)
            seg_data = torch.cat([question_seg_data, seg_data],dim=1).to(device=seg_data.device, dtype=seg_data.dtype)
            question_attention_mask = torch.ones(attention_mask.shape[0], question_len).to(device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([question_attention_mask, attention_mask], dim=1).to(device=attention_mask.device, dtype=attention_mask.dtype)

        if encoder_outputs is None:
            inputs_patches = None
            if image is not None:
                assert visual_seg_data is not None
                x = self.patch_embed(image)
                num_patches = image.size(2) // 16
                if ids_keep is not None:
                    x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(-1)))
                    pad_tokens = self.pad_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
                    x_padded = torch.cat([x, pad_tokens], dim=1)  # no cls token
                    x_padded = torch.gather(x_padded, dim=1,
                                            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_padded.shape[2]))
                    inputs_patches = x_padded
                else:
                    inputs_patches = x

            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                seg_data=seg_data,
                visual_seg_data=visual_seg_data,
                inputs_patches=inputs_patches,
                num_patches=num_patches,
                special_vis_token=self.special_vis_token,
                ids_keep=ids_keep,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        return encoder_outputs

    def get_encoder(self):
        return self



