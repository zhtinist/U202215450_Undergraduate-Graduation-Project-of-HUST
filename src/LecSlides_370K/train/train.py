# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torchvision.transforms as T
import torch
from torchvision.transforms import functional as F
import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def print_trainable_parm(model,prefix):
    for name, module in model.named_modules():
        print_flag = False
        for p in module.parameters():
            if p.requires_grad == True:
                print(f'{prefix}:  {name}')
                print_flag = True
                break

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_bb: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    ocr_free_tower: Optional[str] = field(default=None)
    ocr_base_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    model_map_name: Optional[str] = field(default="llava_phi")
    task: Optional[str] = field(default="lecture_gen")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    ocr_order: str = 'my'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_phi(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
) -> Dict:
    assert has_image is True, f'{sources} do not contain image.'
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids) + 1
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX


        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "phi":
        return preprocess_phi(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

class MLS_Dataset(LazySupervisedDataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(MLS_Dataset, self).__init__(data_path, tokenizer, data_args)
        self.image_root = os.path.dirname(data_path)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if 'image' in sources[0]:
            sample = sources[0]
            image_list = sample['image']
            asr = sample['asr']
            caption = sample['summary']
            image_path = os.path.join(self.image_root, image_list[-1]['image_path'])
            image = Image.open(image_path).convert('RGB')
            processor = self.data_args.image_processor
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # sources = [[{'from': 'human', 'value': '<image>\nCan you give me a summary of this clip according to following teachers transcripts: ' + asr},
            #             {'from': 'gpt', 'value': caption}]]
            # sources = [[{'from': 'human', 'value': '<image>\nCan you give me a summary of this lecture video clip'},
            #             {'from': 'gpt', 'value': caption}]]
            sources = [[{'from': 'human', 'value': '<image>\nThere is a single slide and the speaker speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text (picture) in the given single slide'},
                        {'from': 'gpt', 'value': asr}]]
            sources = preprocess_multimodal(
                copy.deepcopy(sources),
                self.data_args)
        else:
            sample = sources[0]
            asr = sample['asr']
            caption = sample['summary']
            # w asr
            # sources = [[{'from': 'human', 'value': '<image>\nCan you give me a summary of this clip according to following teachers transcripts: ' + asr},
            #             {'from': 'gpt', 'value': caption}]]
            # wo asr
            # sources = [[{'from': 'human', 'value': '<image>\nCan you give me a summary of this lecture video clip'},
            #             {'from': 'gpt', 'value': caption}]]
            # script generate task
            sources = [[{'from': 'human', 'value': '<image>\nThere is a single slide and the speaker speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text (picture) in the given single slide'},
                        {'from': 'gpt', 'value': asr}]]
            sources = copy.deepcopy(sources)
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

class Layout_Pretrain_Dataset(MLS_Dataset):
    def deal_file_path(self, file_path):
        relative_path = file_path.split('/')[-3:]
        return '/'.join(relative_path)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = self.deal_file_path(sample['image_path'])
        ocr_path = self.deal_file_path(sample['ocr_path'])
        layout_conversation = sample['conversations']
        question = layout_conversation[0]['value']
        if '<image>' not in question:
            question = '<image>\n' + question
        answer = layout_conversation[1]['value']
        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert('RGB')
        processor = self.data_args.image_processor
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        sources = [[{'from': 'human', 'value': question},
                    {'from': 'gpt', 'value': answer}]]
        sources = preprocess_multimodal(
            copy.deepcopy(sources),
            self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        data_dict['image'] = image
        return data_dict
class Slideshare_Dataset(MLS_Dataset):
    def get_evidence_idx(self,image_path,evidence):
        img_idx = [int(img.split('/')[-1].split('-')[-2]) for img in image_path]
        idx_pool = []
        for evi in evidence:
            for idx, img in enumerate(img_idx):
                if evi == img:
                    idx_pool.append(idx)
                    break
        if len(idx_pool) != 0:
            return idx_pool
        else:
            return None

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        summary = sample['summary']
        image_pool = []
        for path in image_path:
            cur_path = os.path.join(self.image_root, path)
            image = Image.open(cur_path).convert('RGB')
            image_pool.append(copy.deepcopy(image))
        processor = self.data_args.image_processor
        image = processor.preprocess(image_pool, return_tensors='pt')['pixel_values']
        sources = [[{'from': 'human', 'value': '<image>\nThere are some slides, Can you give me a summary of these slides'},
                    {'from': 'gpt', 'value': summary}]]
        sources = preprocess_multimodal(
            copy.deepcopy(sources),
            self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        data_dict['image'] = image
        # data_dict['evidence'] = evidence_idx
        return data_dict

class Slides_VQA_Dataset(MLS_Dataset):
    def get_evidence_idx(self,image_path,evidence):
        img_idx = [int(img.split('/')[-1].split('-')[-2]) for img in image_path]
        idx_pool = []
        for evi in evidence:
            for idx, img in enumerate(img_idx):
                if evi == img:
                    idx_pool.append(idx)
                    break
        if len(idx_pool) != 0:
            return idx_pool
        else:
            return None

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        question = sample['question']
        if '<image>' not in question:
            question = '<image>\n' + question
        answer = sample['answer']
        evidence = sample['evidence']
        evidence_idx = self.get_evidence_idx(image_path,evidence)
        image_pool = []
        for path in image_path:
            cur_path = os.path.join(self.image_root, path)
            image = Image.open(cur_path).convert('RGB')
            image_pool.append(copy.deepcopy(image))
        processor = self.data_args.image_processor
        image = processor.preprocess(image_pool, return_tensors='pt')['pixel_values']
        sources = [[{'from': 'human', 'value': question},
                    {'from': 'gpt', 'value': answer}]]
        sources = preprocess_multimodal(
            copy.deepcopy(sources),
            self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        data_dict['image'] = image
        # data_dict['evidence'] = evidence_idx
        return data_dict

class Slides_VQA_Dataset_WTG(Slides_VQA_Dataset):
    def get_evidence_idx(self,image_path,evidence):
        img_idx = [int(img.split('/')[-1].split('-')[-2]) for img in image_path]
        idx_pool = []
        for evi in evidence:
            for idx, img in enumerate(img_idx):
                if evi == img:
                    idx_pool.append(idx)
                    break
        if len(idx_pool) != 0:
            return idx_pool
        else:
            return None

    def get_ocr_prompt(self, image_path,evidence):
        ocr_pool = []
        for idx in evidence:
            ocr_path = image_path[idx]
            ocr_path = ocr_path.split('.')[0] + '_ocr.json'
            ocr_data = json.load(open(os.path.join(self.image_root,ocr_path)))
            ocrs = [data[1][0] for data in ocr_data]
            ocr_pool.extend(ocrs)
        return ' '.join(ocr_pool)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        question = sample['question']
        if '<image>' not in question:
            question = '<image>\n' + question
        answer = sample['answer']
        evidence = sample['evidence']
        evidence_idx = self.get_evidence_idx(image_path,evidence)
        ocr_prompt = self.get_ocr_prompt(image_path, evidence_idx)
        image_pool = []
        for path in image_path:
            cur_path = os.path.join(self.image_root, path)
            image = Image.open(cur_path).convert('RGB')
            image_pool.append(copy.deepcopy(image))
        processor = self.data_args.image_processor
        ocr_free_processor = processor['ocr_free']
        image = ocr_free_processor.preprocess(image_pool, return_tensors='pt')['pixel_values']
        sources = [[{'from': 'human', 'value': question},
                    {'from': 'gpt', 'value': answer}]]
        # sources = [[{'from': 'human', 'value':  question + f' The OCR in slides are: {ocr_prompt}'},
        #             {'from': 'gpt', 'value': answer}]]
        sources = preprocess_multimodal(
            copy.deepcopy(sources),
            self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        question_inputs = processor['text_tokenizer']([sample['question']], padding=True, return_tensors="pt")

        # image exist in the data
        data_dict['image'] = image
        data_dict['evidence'] = evidence_idx
        data_dict['text_input_ids'] = question_inputs.input_ids[0]
        return data_dict

class Slides_VQA_Dataset_WOCR(MLS_Dataset):
    def get_evidence_idx(self,image_path,evidence):
        img_idx = [int(img.split('/')[-1].split('-')[-2]) for img in image_path]
        idx_pool = []
        for evi in evidence:
            for idx, img in enumerate(img_idx):
                if evi == img:
                    idx_pool.append(idx)
                    break
        if len(idx_pool) != 0:
            return idx_pool
        else:
            return None

    def get_ocr_prompt(self, image_path,evidence):
        ocr_pool = []
        for idx in evidence:
            ocr_path = image_path[idx]
            ocr_path = ocr_path.split('.')[0] + '_ocr.json'
            ocr_data = json.load(open(os.path.join(self.image_root,ocr_path)))
            ocrs = [data[1][0] for data in ocr_data]
            ocr_pool.extend(ocrs)
        return ' '.join(ocr_pool)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        question = sample['question']
        if '<image>' not in question:
            question = '<image>\n' + question
        answer = sample['answer']
        evidence = sample['evidence']
        evidence_idx = self.get_evidence_idx(image_path,evidence)
        ocr_prompt = self.get_ocr_prompt(image_path,evidence_idx)
        image_pool = []
        for path in image_path:
            cur_path = os.path.join(self.image_root, path)
            image = Image.open(cur_path).convert('RGB')
            image_pool.append(copy.deepcopy(image))
        processor = self.data_args.image_processor
        image = processor.preprocess(image_pool, return_tensors='pt')['pixel_values']
        sources = [[{'from': 'human', 'value':  question + f' The OCR in slides are: {ocr_prompt}'},
                    {'from': 'gpt', 'value': answer}]]
        sources = preprocess_multimodal(
            copy.deepcopy(sources),
            self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        data_dict['image'] = image
        # data_dict['evidence'] = evidence_idx
        return data_dict

class Layout_Pretrain_Dataset_WOCR(Layout_Pretrain_Dataset):
    def deal_ocr_info(self,ocr_info):
        deal_box = []
        deal_text = []
        for info in ocr_info:
            box = info['box']
            deal_box.append(box)
            deal_text.append(info['text'])
        return deal_text, deal_box
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        sample = sources[0]
        image_path = self.deal_file_path(sample['image_path'])
        ocr_path = self.deal_file_path(sample['ocr_path'])
        ocr_info = json.load(open(os.path.join(self.image_root, ocr_path)))
        words, bbox = self.deal_ocr_info(ocr_info)
        prompts_ocr = ' '.join(words)
        layout_conversation = sample['conversations']
        question = layout_conversation[0]['value']
        if '<image>' not in question:
            question = '<image>\n' + question
        answer = layout_conversation[1]['value']
        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert('RGB')
        processor = self.data_args.image_processor
        if self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        sources = [[{'from': 'human', 'value': question + f' The OCR in the document are: {prompts_ocr}'},
                    {'from': 'gpt', 'value': answer}]]
        sources = preprocess_multimodal(
            copy.deepcopy(sources),
            self.data_args)
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        data_dict['image'] = image


        return data_dict


class Mixed_MLS_Dataset(MLS_Dataset):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if 'image' in sources[0]:
            sample = sources[0]
            image_list = sample['image']
            asr = sample['asr']
            caption = sample['summary']
            image_path = os.path.join(self.image_root, image_list[-1]['image_path'])
            image = Image.open(image_path).convert('RGB')
            processor = self.data_args.image_processor
            ocr_free_processor = processor['ocr_free']
            ocr_free_image = ocr_free_processor(image, return_tensors="pt").pixel_values[0]
            ocr_base_processor = processor['ocr_base']
            words = sample['ocr_list']
            bbox = sample['ocr_list_box']
            ocr_base_inputs = ocr_base_processor(image,words,boxes=bbox, return_tensors="pt")
            # sources = [[{'from': 'human', 'value': '<image>\nCan you give me a summary of this clip according to following teachers transcripts: ' + asr},
            #             {'from': 'gpt', 'value': caption}]]
            # sources = [[{'from': 'human', 'value': '<image>\nCan you give me a summary of this lecture video clip'},
            #             {'from': 'gpt', 'value': caption}]]
            sources = [[{'from': 'human', 'value': '<image>\nThere is a single slide and the speaker speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text (picture) in the given single slide'},
                        {'from': 'gpt', 'value': asr}]]
            sources = preprocess_multimodal(
                copy.deepcopy(sources),
                self.data_args)
        else:
            sample = sources[0]
            asr = sample['asr']
            caption = sample['summary']
            # w asr
            # sources = [[{'from': 'human', 'value': '<image>\nCan you give me a summary of this clip according to following teachers transcripts: ' + asr},
            #             {'from': 'gpt', 'value': caption}]]
            # wo asr
            # sources = [[{'from': 'human', 'value': '<image>\nCan you give me a summary of this lecture video clip'},
            #             {'from': 'gpt', 'value': caption}]]
            # script generate task
            sources = [[{'from': 'human', 'value': '<image>\nThere is a single slide and the speaker speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text (picture) in the given single slide'},
                        {'from': 'gpt', 'value': asr}]]
            sources = copy.deepcopy(sources)
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = ocr_free_image
            data_dict['ocr'] = words
            # new_bbox = []
            # for box in bbox:
            #     new = []
            #     for p in box:
            #         if p > 960:
            #             p = 960
            #         elif p < 0:
            #             p = 0
            #         new.append(p)
            #     new_bbox.append(new)

            data_dict['ocr_box'] = bbox

            data_dict['ocr_base_image'] = image

            data_dict['ocr_ids'] = ocr_base_inputs.input_ids
            data_dict['ocr_base_images'] = ocr_base_inputs.pixel_values
            data_dict['ocr_attn'] = ocr_base_inputs.attention_mask
            data_dict['ocr_bbox'] = ocr_base_inputs.bbox


        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor['ocr_free'].crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

class Mixed_Layout_Pretrain_Dataset(MLS_Dataset):

    def deal_file_path(self, file_path):
        relative_path = file_path.split('/')[-3:]
        return '/'.join(relative_path)

    def normalize_bbox(self, bbox, src_size, dst_size):
        """
        Normalize bounding box coordinates.

        Args:
            bbox (List[Union[int, float]]): Bounding box coordinates.
            src_size (Dict[str, Union[int, float]]): Source image size.
            dst_size (Dict[str, Union[int, float]]): Destination image size.

        Returns:
            List[Union[int, float]]: Normalized bounding box coordinates.
        """
        src_w, src_h = src_size["width"], src_size["height"]
        dst_w, dst_h = dst_size["width"], dst_size["height"]
        x1, y1, x2, y2 = bbox
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        x1 = int(x_min / src_w * dst_w)
        y1 = int(y_min / src_h * dst_h)
        x2 = int(x_max / src_w * dst_w)
        y2 = int(y_max / src_h * dst_h)
        ocr_box = [x1, y1, x2, y2]
        filter_box = []
        for p in ocr_box:
            if p < 0:
                p = 0
            elif p > 1000:
                p = 1000
            filter_box.append(p)
        # if any([(x < 0 or x > 1000) for x in ocr_box]):
        #     print(f'wrong box: {ocr_box}, {bbox}, {src_size}')
        #     return False

        return filter_box

    def deal_ocr_info(self,ocr_info,im_w,im_h):
        deal_box = []
        deal_text = []
        for info in ocr_info:
            box = self.normalize_bbox(info['box'],src_size={"width": im_w, "height": im_h},dst_size={"width": 1000, "height": 1000})
            deal_box.append(box)
            deal_text.append(info['text'])
        return deal_text, deal_box

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        sample = sources[0]
        image_path = self.deal_file_path(sample['image_path'])
        ocr_path = self.deal_file_path(sample['ocr_path'])
        layout_conversation = sample['conversations']
        question = layout_conversation[0]['value']
        if '<image>' not in question:
            question = '<image>\n' + question
        answer = layout_conversation[1]['value']

        ocr_info = json.load(open(os.path.join(self.image_root,ocr_path)))
        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert('RGB')
        im_w, im_h = image.size
        processor = self.data_args.image_processor
        ocr_free_processor = processor['ocr_free']
        ocr_free_image = ocr_free_processor(image, return_tensors="pt").pixel_values[0]
        words, bbox = self.deal_ocr_info(ocr_info,im_w,im_h)
        sources = [[{'from': 'human', 'value': question},
                    {'from': 'gpt', 'value': answer}]]
        sources = preprocess_multimodal(
            copy.deepcopy(sources),
            self.data_args)
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        data_dict['image'] = ocr_free_image
        data_dict['ocr'] = words

        data_dict['ocr_box'] = bbox

        data_dict['ocr_base_image'] = image

        return data_dict
class Normalize(object):
    def __init__(self, mean, std, format='rgb'):
        self.mean = mean
        self.std = std
        self.format = format.lower()

    def __call__(self, image):
        if 'bgr' in self.format:
            image = image[[2, 1, 0]]
        if '255' in self.format:
            image = image * 255
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image
class UDOP_Layout_Pretrain_Dataset(Mixed_Layout_Pretrain_Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(UDOP_Layout_Pretrain_Dataset, self).__init__(data_path, tokenizer, data_args)
        self.patch_pos = self.get_patch_position()

    def deal_file_path(self, file_path):
        relative_path = file_path.split('/')[-3:]
        return '/'.join(relative_path)

    def normalize_bbox(self, bbox, src_size, dst_size):
        """
        Normalize bounding box coordinates.

        Args:
            bbox (List[Union[int, float]]): Bounding box coordinates.
            src_size (Dict[str, Union[int, float]]): Source image size.
            dst_size (Dict[str, Union[int, float]]): Destination image size.

        Returns:
            List[Union[int, float]]: Normalized bounding box coordinates.
        """
        src_w, src_h = src_size["width"], src_size["height"]
        dst_w, dst_h = dst_size["width"], dst_size["height"]
        x1, y1, x2, y2 = bbox
        x_min = min(x1, x2)
        x_max = max(x1, x2)
        y_min = min(y1, y2)
        y_max = max(y1, y2)

        x1 = float(x_min / src_w * dst_w)
        y1 = float(y_min / src_h * dst_h)
        x2 = float(x_max / src_w * dst_w)
        y2 = float(y_max / src_h * dst_h)
        ocr_box = [x1, y1, x2, y2]
        filter_box = []
        for p in ocr_box:
            if p < 0:
                p = 0
            elif p >= 1:
                p = 1
            filter_box.append(p)

        return filter_box

    def deal_ocr_info(self,ocr_info,im_w,im_h):
        deal_box = []
        deal_text = []
        for info in ocr_info:
            box = self.normalize_bbox(info['box'],src_size={"width": im_w, "height": im_h},dst_size={"width": 1.0, "height": 1.0})
            deal_box.append(box)
            deal_text.append(info['text'])
        return deal_text, deal_box

    def UDOP_image_transformers(self,image,image_size=224):
        trans = T.Compose([
            T.Resize([image_size, image_size]),
            T.ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        image = trans(image)  # copy to make it writeable
        return image

    def tokenize_ocr(self,words,boxes,processor,max_len=512):
        return_tokens = []
        return_boxes = []
        for w, b in zip(words,boxes):
            tokens = processor(w,add_special_tokens=False).input_ids
            return_tokens.extend(tokens)
            return_boxes.extend([b] * len(tokens))
        if len(return_tokens) == 0:
            return_tokens = [0]
            return_boxes = [[0,0,0,0]]
        return_tokens = torch.tensor(return_tokens).long()
        return_boxes = torch.tensor(return_boxes).float()
        if len(return_tokens) > max_len:
            return_tokens = return_tokens[:max_len]
            return_boxes = return_boxes[:max_len]
        return return_tokens, return_boxes

    def get_patch_position(self, num_patches=14):
        patch_pos = []
        for j in range(0,num_patches):
            for i in range(0,num_patches):
                patch_pos.append([i, j, i + 1, j + 1])
        patch_pos = torch.tensor(patch_pos).float()
        patch_pos = patch_pos / num_patches

        return patch_pos

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        sample = sources[0]
        image_path = self.deal_file_path(sample['image_path'])
        ocr_path = self.deal_file_path(sample['ocr_path'])
        layout_conversation = sample['conversations']
        question = layout_conversation[0]['value']
        if '<image>' not in question:
            question = '<image>\n' + question
        answer = layout_conversation[1]['value']

        ocr_info = json.load(open(os.path.join(self.image_root,ocr_path)))
        image_path = os.path.join(self.image_root, image_path)
        image = Image.open(image_path).convert('RGB')
        im_w, im_h = image.size
        processor = self.data_args.image_processor
        image_tensor = self.UDOP_image_transformers(image)
        words, bbox = self.deal_ocr_info(ocr_info,im_w,im_h)
        ocr_tokens, ocr_box = self.tokenize_ocr(words,bbox,processor)
        sources = [[{'from': 'human', 'value': question},
                    {'from': 'gpt', 'value': answer}]]
        sources = preprocess_multimodal(
            copy.deepcopy(sources),
            self.data_args)
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        data_dict['image'] = image_tensor
        data_dict['ocr'] = ocr_tokens
        data_dict['ocr_box'] = ocr_box
        data_dict['patch_pos'] = self.patch_pos
        return data_dict

class Slides_VQA_UDOP_Dataset(UDOP_Layout_Pretrain_Dataset):
    def get_evidence_idx(self,image_path,evidence):
        img_idx = [int(img.split('/')[-1].split('-')[-2]) for img in image_path]
        idx_pool = []
        for evi in evidence:
            for idx, img in enumerate(img_idx):
                if evi == img:
                    idx_pool.append(idx)
                    break
        if len(idx_pool) != 0:
            return idx_pool
        else:
            return None

    def deal_bbox(self, bbox, im_w, im_h):
        dealed_box = []
        for box in bbox:
            x_min = min([point[0] for point in box])
            x_max = max([point[0] for point in box])
            y_min = min([point[1] for point in box])
            y_max = max([point[1] for point in box])
            cur_box = [x_min,y_min,x_max,y_max]
            cur_box = self.normalize_bbox(cur_box, src_size={"width": im_w, "height": im_h},
                                dst_size={"width": 1.0, "height": 1.0})
            dealed_box.append(cur_box)

        return dealed_box


    def deal_ocr_info(self, image_path, im_w, im_h):
        ocr_path = image_path.split('.')[0] + '_ocr.json'
        if os.path.exists(os.path.join(self.image_root, ocr_path)):
            ocr_data = json.load(open(os.path.join(self.image_root, ocr_path)))
            if ocr_data is not None:
                ocrs = [data[1][0] for data in ocr_data]
                bbox = [data[0] for data in ocr_data]
                dealed_bbox = self.deal_bbox(bbox,im_w,im_h)
                return ocrs, dealed_bbox
        return 'Slide', [[0,0,0,0]]


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        question = sample['question']
        if '<image>' not in question:
            question = '<image>\n' + question
        answer = sample['answer']
        evidence = sample['evidence']
        evidence_idx = self.get_evidence_idx(image_path,evidence)
        processor = self.data_args.image_processor
        image_pool = []
        ocr_tokens_pool = []
        ocr_box_pool = []
        for path in image_path:
            cur_path = os.path.join(self.image_root, path)
            image = Image.open(cur_path).convert('RGB')
            image_tensor = self.UDOP_image_transformers(image)
            image_pool.append(image_tensor)
            im_w, im_h = image.size
            words, bbox = self.deal_ocr_info(path,im_w,im_h)
            ocr_tokens, ocr_box = self.tokenize_ocr(words, bbox, processor)
            ocr_tokens_pool.append(ocr_tokens)
            ocr_box_pool.append(ocr_box)


        sources = [[{'from': 'human', 'value': question},
                    {'from': 'gpt', 'value': answer}]]
        sources = preprocess_multimodal(
            copy.deepcopy(sources),
            self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data

        # padding
        ocrs = torch.nn.utils.rnn.pad_sequence(
            ocr_tokens_pool,
            batch_first=True,
            padding_value=processor.pad_token_id)

        max_len = max(x.shape[0] for x in ocr_box_pool)
        align_ocr_boxes = []
        for box in ocr_box_pool:
            cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
                                                       dtype=box.dtype)), dim=0)
            align_ocr_boxes.append(cur_box)
        align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        data_dict['seg_data'] = align_ocr_boxes
        data_dict['ocrs'] = ocrs
        data_dict['visual_seg_data'] = self.patch_pos.repeat(align_ocr_boxes.shape[0],1,1)
        data_dict['images'] = torch.stack(image_pool)
        # data_dict['evidence'] = evidence_idx
        return data_dict

class Slideshare_UDOP_Dataset(Slides_VQA_UDOP_Dataset):
    def deal_ocr_info(self, ocr_path, im_w, im_h):
        if os.path.exists(ocr_path):
            ocr_data = json.load(open(ocr_path))
            if ocr_data is not None:
                ocrs = [data[1][0] for data in ocr_data]
                bbox = [data[0] for data in ocr_data]
                dealed_bbox = self.deal_bbox(bbox,im_w,im_h)
                return ocrs, dealed_bbox
        return 'Slide', [[0,0,0,0]]
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        ocr_path = sample['ocr']
        summary = sample['summary']


        processor = self.data_args.image_processor
        image_pool = []
        ocr_tokens_pool = []
        ocr_box_pool = []
        for path, ocr_p in zip(image_path, ocr_path):
            cur_path = os.path.join(path)
            image = Image.open(cur_path).convert('RGB')
            image_tensor = self.UDOP_image_transformers(image)
            image_pool.append(image_tensor)
            im_w, im_h = image.size
            words, bbox = self.deal_ocr_info(ocr_p,im_w,im_h)
            ocr_tokens, ocr_box = self.tokenize_ocr(words, bbox, processor)
            ocr_tokens_pool.append(ocr_tokens)
            ocr_box_pool.append(ocr_box)


        sources = [[{'from': 'human', 'value': '<image>\nThere are some slides, Can you give me a summary of these slides'},
                    {'from': 'gpt', 'value': summary}]]
        sources = preprocess_multimodal(
            copy.deepcopy(sources),
            self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data

        # padding
        ocrs = torch.nn.utils.rnn.pad_sequence(
            ocr_tokens_pool,
            batch_first=True,
            padding_value=processor.pad_token_id)

        max_len = max(x.shape[0] for x in ocr_box_pool)
        align_ocr_boxes = []
        for box in ocr_box_pool:
            cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
                                                       dtype=box.dtype)), dim=0)
            align_ocr_boxes.append(cur_box)
        align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        data_dict['seg_data'] = align_ocr_boxes
        data_dict['ocrs'] = ocrs
        data_dict['visual_seg_data'] = self.patch_pos.repeat(align_ocr_boxes.shape[0],1,1)
        data_dict['images'] = torch.stack(image_pool)
        # data_dict['evidence'] = evidence_idx
        return data_dict

class Slideshare_UDOP_Dataset_For_T5(Slideshare_UDOP_Dataset):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        ocr_path = sample['ocr']
        summary = sample['summary']


        processor = self.data_args.image_processor
        image_pool = []
        ocr_tokens_pool = []
        ocr_box_pool = []
        for path, ocr_p in zip(image_path, ocr_path):
            cur_path = os.path.join(path)
            image = Image.open(cur_path).convert('RGB')
            image_tensor = self.UDOP_image_transformers(image)
            image_pool.append(image_tensor)
            im_w, im_h = image.size
            words, bbox = self.deal_ocr_info(ocr_p,im_w,im_h)
            ocr_tokens, ocr_box = self.tokenize_ocr(words, bbox, processor)
            ocr_tokens_pool.append(ocr_tokens)
            ocr_box_pool.append(ocr_box)

        data_dict = {}
        inputs = 'There are some texts in slides, Can you give me a summary of these slides.'
        labels = summary
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(labels, return_tensors="pt").input_ids[0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        # image exist in the data

        # padding
        ocrs = torch.nn.utils.rnn.pad_sequence(
            ocr_tokens_pool,
            batch_first=True,
            padding_value=processor.pad_token_id)

        max_len = max(x.shape[0] for x in ocr_box_pool)
        align_ocr_boxes = []
        for box in ocr_box_pool:
            cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
                                                       dtype=box.dtype)), dim=0)
            align_ocr_boxes.append(cur_box)
        align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        data_dict['seg_data'] = align_ocr_boxes
        data_dict['ocrs'] = ocrs
        data_dict['visual_seg_data'] = self.patch_pos.repeat(align_ocr_boxes.shape[0],1,1)
        data_dict['images'] = torch.stack(image_pool)
        # data_dict['evidence'] = evidence_idx
        return data_dict

class MYQA_UDOP_Dataset_For_T5(Slideshare_UDOP_Dataset):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        ocr_path = sample['ocr']
        qa_pair = sample['qa_pair']
        summary = qa_pair['answer']
        question = qa_pair['question']


        processor = self.data_args.image_processor
        image_pool = []
        ocr_tokens_pool = []
        ocr_box_pool = []
        for path, ocr_p in zip(image_path, ocr_path):
            cur_path = os.path.join(path)
            image = Image.open(cur_path).convert('RGB')
            image_tensor = self.UDOP_image_transformers(image)
            image_pool.append(image_tensor)
            im_w, im_h = image.size
            words, bbox = self.deal_ocr_info(ocr_p,im_w,im_h)
            ocr_tokens, ocr_box = self.tokenize_ocr(words, bbox, processor)
            ocr_tokens_pool.append(ocr_tokens)
            ocr_box_pool.append(ocr_box)

        data_dict = {}
        inputs = 'There are some texts in slides, Can you give me a summary of these slides.'
        labels = summary
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(labels, return_tensors="pt").input_ids[0]
        question = self.tokenizer(question, return_tensors="pt").input_ids[0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        # image exist in the data

        # padding
        ocrs = torch.nn.utils.rnn.pad_sequence(
            ocr_tokens_pool,
            batch_first=True,
            padding_value=processor.pad_token_id)

        max_len = max(x.shape[0] for x in ocr_box_pool)
        align_ocr_boxes = []
        for box in ocr_box_pool:
            cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
                                                       dtype=box.dtype)), dim=0)
            align_ocr_boxes.append(cur_box)
        align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        data_dict['seg_data'] = align_ocr_boxes
        data_dict['ocrs'] = ocrs
        data_dict['visual_seg_data'] = self.patch_pos.repeat(align_ocr_boxes.shape[0],1,1)
        data_dict['images'] = torch.stack(image_pool)
        data_dict['question'] = question.repeat(ocrs.shape[0], 1)
        # data_dict['evidence'] = evidence_idx
        return data_dict

class Slideshare_UDOP_Dataset_For_T5_Single(Slideshare_UDOP_Dataset):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        ocr_path = sample['ocr']
        summary = sample['description']


        processor = self.data_args.image_processor
        cur_path = os.path.join(image_path)
        image = Image.open(cur_path).convert('RGB')
        image_tensor = self.UDOP_image_transformers(image)
        im_w, im_h = image.size
        words, bbox = self.deal_ocr_info(ocr_path,im_w,im_h)
        ocr_tokens, ocr_box = self.tokenize_ocr(words, bbox, processor)

        data_dict = {}
        inputs = 'There are some texts in slides, Can you give me a summary of these slides.'
        labels = summary
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(labels, return_tensors="pt").input_ids[0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        data_dict['ocr_box'] = ocr_box
        data_dict['ocr'] = ocr_tokens
        data_dict['patch_pos'] = self.patch_pos
        data_dict['image'] = image_tensor

        return data_dict

        # image exist in the data

        # padding
        # ocrs = torch.nn.utils.rnn.pad_sequence(
        #     ocr_tokens_pool,
        #     batch_first=True,
        #     padding_value=processor.pad_token_id)
        #
        # max_len = max(x.shape[0] for x in ocr_box_pool)
        # align_ocr_boxes = []
        # for box in ocr_box_pool:
        #     cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
        #                                                dtype=box.dtype)), dim=0)
        #     align_ocr_boxes.append(cur_box)
        # align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        # data_dict['seg_data'] = align_ocr_boxes
        # data_dict['ocrs'] = ocrs
        # data_dict['visual_seg_data'] = self.patch_pos.repeat(align_ocr_boxes.shape[0],1,1)
        # data_dict['images'] = torch.stack(image_pool)
        # # data_dict['evidence'] = evidence_idx
        # return data_dict
def get_relative_pos(box1, box2):
    box1_ctr_x = (box1[0] + box1[2]) / 2
    box1_ctr_y = (box1[1] + box1[3]) / 2
    box2_ctr_x = (box2[0] + box2[2]) / 2
    box2_ctr_y = (box2[1] + box2[3]) / 2
    w_dist = torch.abs((box1_ctr_x - box2_ctr_x) / (box1[2] - box1[0]))
    h_dist = torch.abs((box1_ctr_y - box2_ctr_y) / (box1[3] - box1[1]))
    return [w_dist, h_dist]

def get_union_box(box1,box2):
    return [min(box1[0],box2[0]),min(box1[1],box2[1]),max(box1[2],box2[2]),max(box1[3],box2[3])]

def get_dist_graph(seg_data, max_len):
    text_len= len(seg_data)
    if text_len > max_len:
        text_len = max_len
    matrix = torch.zeros(text_len,text_len,6)
    cur_seg_data = seg_data
    for i in range(text_len):
        for j in range(text_len):
            if (cur_seg_data[i] == cur_seg_data[j]).all():
                matrix[i,j] = torch.zeros(6)
                matrix[i, j] = torch.exp(-matrix[i,j])
                continue
            if cur_seg_data[i].mean() in [0.0,1.0] or cur_seg_data[j].mean() in [0.0,1.0]:
                continue
            cur_dist_list = []
            union_box = get_union_box(cur_seg_data[i], cur_seg_data[j])
            cur_dist_list.extend(get_relative_pos(cur_seg_data[i],cur_seg_data[j]))
            cur_dist_list.extend(get_relative_pos(cur_seg_data[i],union_box))
            cur_dist_list.extend(get_relative_pos(cur_seg_data[j],union_box))
            matrix[i, j] = torch.tensor(cur_dist_list)
            matrix[i, j] = matrix[i,j]
            matrix[i, j] = torch.exp(-matrix[i,j])
    return matrix

def get_relation_graph(seg_data, max_len, step):
    text_len= len(seg_data)
    if text_len > max_len:
        text_len = max_len
    if len(step) == 0:
        matrix = torch.ones(text_len,text_len)
        return matrix
    matrix = torch.zeros(text_len,text_len)
    prev_step = 0
    for step_ in step:
        matrix[prev_step:step_, prev_step:step_] = 1
        prev_step = step_
    matrix[prev_step:, prev_step:] = 1
    return matrix


class Slideshare_UDOP_Dataset_For_T5_Single_Order(Slideshare_UDOP_Dataset):
    def tokenize_ocr(self,words,boxes,processor,max_len=384):
        return_tokens = []
        return_boxes = []
        return_index = []
        for idx, (w, b) in enumerate(zip(words,boxes)):
            tokens = processor(w,add_special_tokens=False).input_ids
            return_tokens.extend(tokens)
            return_boxes.extend([b] * len(tokens))
            return_index.extend([idx] * len(tokens))
        if len(return_tokens) == 0:
            return_tokens = [0]
            return_boxes = [[0,0,0,0]]
            return_index = [0]
        return_tokens = torch.tensor(return_tokens).long()
        return_boxes = torch.tensor(return_boxes).float()
        if len(return_tokens) > max_len:
            return_tokens = return_tokens[:max_len]
            return_boxes = return_boxes[:max_len]
            return_index = return_index[:max_len]
        return return_tokens, return_boxes, return_index
    def deal_element_bbox(self, bbox, im_w, im_h):
        dealed_box = []
        for box in bbox:
            cur_box = self.normalize_bbox(box, src_size={"width": im_w, "height": im_h},
                                dst_size={"width": 1.0, "height": 1.0})
            dealed_box.append(cur_box)

        return dealed_box
    def get_element_pos(self, box_info, thr=0.4):
        # filter score
        image = []

        for i in range(len(box_info['labels'])):
            if box_info['scores'][i] > thr:
                if box_info['labels'][i] in [3, 4, 6, 8]:
                    image.append(box_info['bboxes'][i])
        return image

    def deal_ocr_info(self, ocr_path, im_w, im_h):
        if os.path.exists(ocr_path):
            ocr_data = json.load(open(ocr_path))
            ocr_data = ocr_data['ocr_info']
            if ocr_data is not None:
                ocrs = [data[1][0] for data in ocr_data]
                bbox = [data[0] for data in ocr_data]
                dealed_bbox = self.deal_bbox(bbox,im_w,im_h)
                return ocrs, dealed_bbox
        return 'Slide', [[0,0,0,0]]
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        ocr_path = sample['ocr']
        if self.data_args.ocr_order == 'my':
            ocr_path = ocr_path.replace('_ocr.json', '_bypos_v2.json')
        elif self.data_args.ocr_order == 'k3':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_kmean3.json')
        elif self.data_args.ocr_order == 'k4':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_kmean4.json')
        elif self.data_args.ocr_order == 'hac':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_hac.json')
        elif self.data_args.ocr_order == 'r2':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_r2.json')
        elif self.data_args.ocr_order == 'r8':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_r8.json')
        else:
            raise NotImplementedError

        # add layout loss
        ocr_step = json.load(open(ocr_path))['step']


        summary = sample['description']

        processor = self.data_args.image_processor
        cur_path = os.path.join(image_path)
        image = Image.open(cur_path).convert('RGB')
        image_tensor = self.UDOP_image_transformers(image)
        im_w, im_h = image.size
        words, bbox = self.deal_ocr_info(ocr_path,im_w,im_h)
        ocr_tokens, ocr_box, ocr_index = self.tokenize_ocr(words, bbox, processor)
        max_index = max(ocr_index)

        cal_bbox = torch.tensor(bbox)
        ocr_matrix = get_dist_graph(cal_bbox, max_index + 1)
        res_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0], 6)
        for i in range(ocr_tokens.shape[0]):
            for j in range(ocr_tokens.shape[0]):
                res_matrix[i, j] = ocr_matrix[ocr_index[i], ocr_index[j]]

        relation_matrix = get_relation_graph(cal_bbox, max_index + 1, step=ocr_step)
        rel_token_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0])
        for i in range(ocr_tokens.shape[0]):
            for j in range(ocr_tokens.shape[0]):
                rel_token_matrix[i, j] = relation_matrix[ocr_index[i], ocr_index[j]]


        # prepare element info
        bbox_path = image_path.replace('/images/', '/bbox/')
        bbox_root = os.path.dirname(bbox_path)
        bbox_file = os.path.basename(bbox_path).split('.')[0] + '.json'
        bbox_path = os.path.join(bbox_root, 'preds', bbox_file)
        bbox_info = json.load(open(bbox_path))
        element_pos = self.get_element_pos(bbox_info)
        max_element_len = 3
        if len(element_pos) > 0:
            element_pos = self.deal_element_bbox(element_pos,im_w,im_h)
            element_pos = element_pos[:max_element_len]
            # element_pos = None
        else:
            element_pos = [[0,0,0.02,0.02]]

        data_dict = {}
        inputs = 'There are some texts in slides, Can you give me a summary of these slides.'
        labels = summary
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(labels, return_tensors="pt").input_ids[0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        data_dict['ocr_box'] = ocr_box
        data_dict['ocr'] = ocr_tokens
        data_dict['patch_pos'] = self.patch_pos
        data_dict['image'] = image_tensor
        data_dict['element_data'] = element_pos
        data_dict['text_dist_graph'] = res_matrix
        data_dict['relation_label'] = rel_token_matrix

        return data_dict

class Slideshare_UDOP_Dataset_For_T5_Order(Slideshare_UDOP_Dataset_For_T5_Single_Order):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        ocr_path = sample['ocr']



        summary = sample['summary']

        image_pool = []
        ocr_tokens_pool = []
        ocr_box_pool = []
        element_pos_pool = []
        res_matrix_pool = []
        graph_shape_pool = []
        rel_token_matrix_pool = []

        processor = self.data_args.image_processor
        for path, ocr_p in zip(image_path, ocr_path):
            cur_path = os.path.join(path)
            if self.data_args.ocr_order == 'my':
                ocr_p = ocr_p.replace('_ocr.json', '_bypos_v2.json')
            elif self.data_args.ocr_order == 'k3':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_kmean3.json')
            elif self.data_args.ocr_order == 'k4':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_kmean4.json')
            elif self.data_args.ocr_order == 'hac':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_hac.json')
            elif self.data_args.ocr_order == 'r2':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_r2.json')
            elif self.data_args.ocr_order == 'r8':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_r8.json')
            else:
                raise NotImplementedError
            ocr_step = json.load(open(ocr_p))['step']

            image = Image.open(cur_path).convert('RGB')
            image_tensor = self.UDOP_image_transformers(image)
            im_w, im_h = image.size
            words, bbox = self.deal_ocr_info(ocr_p,im_w,im_h)
            ocr_tokens, ocr_box, ocr_index = self.tokenize_ocr(words, bbox, processor)
            max_index = max(ocr_index)


            cal_bbox = torch.tensor(bbox)
            ocr_matrix = get_dist_graph(cal_bbox, max_index + 1)
            res_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0], 6)
            for i in range(ocr_tokens.shape[0]):
                for j in range(ocr_tokens.shape[0]):
                    res_matrix[i, j] = ocr_matrix[ocr_index[i], ocr_index[j]]

            relation_matrix = get_relation_graph(cal_bbox, max_index + 1, step=ocr_step)
            rel_token_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0])
            for i in range(ocr_tokens.shape[0]):
                for j in range(ocr_tokens.shape[0]):
                    rel_token_matrix[i, j] = relation_matrix[ocr_index[i], ocr_index[j]]


            # prepare element info
            bbox_path = path.replace('/images/', '/bbox/')
            bbox_root = os.path.dirname(bbox_path)
            bbox_file = os.path.basename(bbox_path).split('.')[0] + '.json'
            bbox_path = os.path.join(bbox_root, 'preds', bbox_file)
            bbox_info = json.load(open(bbox_path))
            element_pos = self.get_element_pos(bbox_info)
            max_element_len = 3
            if len(element_pos) > 0:
                element_pos = self.deal_element_bbox(element_pos,im_w,im_h)
                element_pos = element_pos[:max_element_len]
                # element_pos = None
            else:
                element_pos = [[0,0,0.02,0.02]]

            image_pool.append(image_tensor)
            ocr_tokens_pool.append(ocr_tokens)
            ocr_box_pool.append(ocr_box)
            res_matrix_pool.append(res_matrix)
            graph_shape_pool.append(res_matrix.shape[0])
            rel_token_matrix_pool.append(rel_token_matrix)
            element_pos_pool.append(element_pos)

        data_dict = {}
        inputs = 'There are some texts in slides, Can you give me a summary of these slides.'
        labels = summary
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(labels, return_tensors="pt").input_ids[0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        # padding
        ocrs = torch.nn.utils.rnn.pad_sequence(
            ocr_tokens_pool,
            batch_first=True,
            padding_value=processor.pad_token_id)

        max_len = max(x.shape[0] for x in ocr_box_pool)
        align_ocr_boxes = []
        for box in ocr_box_pool:
            cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
                                                       dtype=box.dtype)), dim=0)
            align_ocr_boxes.append(cur_box)
        align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        data_dict['seg_data'] = align_ocr_boxes
        data_dict['ocrs'] = ocrs
        data_dict['visual_seg_data'] = self.patch_pos.repeat(align_ocr_boxes.shape[0],1,1)
        data_dict['images'] = torch.stack(image_pool)
        data_dict['element_data'] = element_pos_pool
        data_dict['graph_shape'] = graph_shape_pool
        max_shape_list = max(graph_shape_pool)
        text_graph_tensor = torch.zeros(len(graph_shape_pool), max_shape_list, max_shape_list, res_matrix_pool[0].shape[-1])
        for bs in range(text_graph_tensor.shape[0]):
            text_graph_tensor[bs, :graph_shape_pool[bs], :graph_shape_pool[bs]] = res_matrix_pool[bs]
        data_dict['text_dist_graph'] = text_graph_tensor

        relation_label_tensor = torch.zeros(len(graph_shape_pool), max_shape_list, max_shape_list)
        for bs in range(relation_label_tensor.shape[0]):
            relation_label_tensor[bs, :graph_shape_pool[bs], :graph_shape_pool[bs]] = rel_token_matrix_pool[bs]

        data_dict['relation_label'] = relation_label_tensor


        # data_dict['ocr_box'] = ocr_box
        # data_dict['ocr'] = ocr_tokens
        # data_dict['patch_pos'] = self.patch_pos
        # data_dict['image'] = image_tensor
        # data_dict['element_data'] = element_pos
        # data_dict['text_dist_graph'] = res_matrix
        # data_dict['relation_label'] = rel_token_matrix

        return data_dict

class SlideVQA_UDOP_Dataset_For_T5_Single_Order(Slideshare_UDOP_Dataset_For_T5_Order):

    def get_evidence_idx(self,image_path,evidence):
        img_idx = [int(img.split('/')[-1].split('-')[-2]) for img in image_path]
        idx_pool = []
        for evi in evidence:
            for idx, img in enumerate(img_idx):
                if evi == img:
                    idx_pool.append(idx)
                    break
        if len(idx_pool) != 0:
            return idx_pool
        else:
            return None
    def get_ocr_prompt(self, image_path,evidence):
        ocr_pool = []
        for idx in evidence:
            ocr_path = image_path[idx]
            ocr_path = ocr_path.split('.')[0] + '_ocr.json'
            ocr_data = json.load(open(os.path.join(self.image_root,ocr_path)))
            ocrs = [data[1][0] for data in ocr_data]
            ocr_pool.extend(ocrs)
        return ' '.join(ocr_pool)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_paths = sample['image']
        question = sample['question']
        evidence = sample['evidence']
        evidence_idx = self.get_evidence_idx(image_paths,evidence)[0]
        image_path = image_paths[evidence_idx]
        image_path = os.path.join(self.image_root,image_path)
        ocr_path = image_path.split('.')[0] + '_ocr.json'
        if self.data_args.ocr_order == 'my':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_v2.json')
        elif self.data_args.ocr_order == 'k3':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_kmean3.json')
        elif self.data_args.ocr_order == 'k4':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_kmean4.json')
        elif self.data_args.ocr_order == 'hac':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_hac.json')
        elif self.data_args.ocr_order == 'r2':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_r2.json')
        elif self.data_args.ocr_order == 'r8':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_r8.json')
        else:
            raise NotImplementedError

        # add layout loss
        ocr_step = json.load(open(ocr_path))['step']

        summary = sample['answer']

        processor = self.data_args.image_processor
        cur_path = os.path.join(image_path)
        image = Image.open(cur_path).convert('RGB')
        image_tensor = self.UDOP_image_transformers(image)
        im_w, im_h = image.size
        words, bbox = self.deal_ocr_info(ocr_path, im_w, im_h)
        ocr_tokens, ocr_box, ocr_index = self.tokenize_ocr(words, bbox, processor)
        max_index = max(ocr_index)

        cal_bbox = torch.tensor(bbox)
        ocr_matrix = get_dist_graph(cal_bbox, max_index + 1)
        res_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0], 6)
        for i in range(ocr_tokens.shape[0]):
            for j in range(ocr_tokens.shape[0]):
                res_matrix[i, j] = ocr_matrix[ocr_index[i], ocr_index[j]]

        relation_matrix = get_relation_graph(cal_bbox, max_index + 1, step=ocr_step)
        rel_token_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0])
        for i in range(ocr_tokens.shape[0]):
            for j in range(ocr_tokens.shape[0]):
                rel_token_matrix[i, j] = relation_matrix[ocr_index[i], ocr_index[j]]

        # prepare element info
        bbox_path = image_path.replace('/images/', '/bbox/')
        bbox_root = os.path.dirname(bbox_path)
        bbox_file = os.path.basename(bbox_path).split('.')[0] + '.json'
        bbox_path = os.path.join(bbox_root, 'preds', bbox_file)
        bbox_info = json.load(open(bbox_path))
        element_pos = self.get_element_pos(bbox_info)
        max_element_len = 3
        if len(element_pos) > 0:
            element_pos = self.deal_element_bbox(element_pos, im_w, im_h)
            element_pos = element_pos[:max_element_len]
            # element_pos = None
        else:
            element_pos = [[0.01, 0.01, 0.02, 0.02]]

        data_dict = {}
        inputs = 'There are some texts in slides, Can you give me a summary of these slides.'
        labels = summary
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(labels, return_tensors="pt").input_ids[0]
        question = self.tokenizer(question, return_tensors="pt").input_ids[0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        data_dict['ocr_box'] = ocr_box
        data_dict['ocr'] = ocr_tokens
        data_dict['patch_pos'] = self.patch_pos
        data_dict['image'] = image_tensor
        data_dict['element_data'] = element_pos
        data_dict['text_dist_graph'] = res_matrix
        data_dict['relation_label'] = rel_token_matrix
        data_dict['question'] = question

        return data_dict


class MYQA_UDOP_Dataset_For_T5_Single_Order(Slideshare_UDOP_Dataset_For_T5_Order):

    def get_evidence_idx(self,image_path,evidence):
        img_idx = [int(img.split('/')[-1].split('-')[-2]) for img in image_path]
        idx_pool = []
        for evi in evidence:
            for idx, img in enumerate(img_idx):
                if evi == img:
                    idx_pool.append(idx)
                    break
        if len(idx_pool) != 0:
            return idx_pool
        else:
            return None
    def get_ocr_prompt(self, image_path,evidence):
        ocr_pool = []
        for idx in evidence:
            ocr_path = image_path[idx]
            ocr_path = ocr_path.split('.')[0] + '_ocr.json'
            ocr_data = json.load(open(os.path.join(self.image_root,ocr_path)))
            ocrs = [data[1][0] for data in ocr_data]
            ocr_pool.extend(ocrs)
        return ' '.join(ocr_pool)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        question = sample['qa_pair']['question']
        # image_path = os.path.join(self.image_root,image_path)
        ocr_path = sample['ocr']
        if self.data_args.ocr_order == 'my':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_v2.json')
        elif self.data_args.ocr_order == 'k3':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_kmean3.json')
        elif self.data_args.ocr_order == 'k4':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_kmean4.json')
        elif self.data_args.ocr_order == 'hac':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_hac.json')
        elif self.data_args.ocr_order == 'r2':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_r2.json')
        elif self.data_args.ocr_order == 'r8':
            ocr_path = ocr_path.replace('_ocr.json', '_ocr_cluster_r8.json')
        else:
            raise NotImplementedError

        # add layout loss
        ocr_step = json.load(open(ocr_path))['step']

        summary = sample['qa_pair']['answer']

        processor = self.data_args.image_processor
        cur_path = os.path.join(image_path)
        image = Image.open(cur_path).convert('RGB')
        image_tensor = self.UDOP_image_transformers(image)
        im_w, im_h = image.size
        words, bbox = self.deal_ocr_info(ocr_path, im_w, im_h)
        ocr_tokens, ocr_box, ocr_index = self.tokenize_ocr(words, bbox, processor)
        max_index = max(ocr_index)

        cal_bbox = torch.tensor(bbox)
        ocr_matrix = get_dist_graph(cal_bbox, max_index + 1)
        res_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0], 6)
        for i in range(ocr_tokens.shape[0]):
            for j in range(ocr_tokens.shape[0]):
                res_matrix[i, j] = ocr_matrix[ocr_index[i], ocr_index[j]]

        relation_matrix = get_relation_graph(cal_bbox, max_index + 1, step=ocr_step)
        rel_token_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0])
        for i in range(ocr_tokens.shape[0]):
            for j in range(ocr_tokens.shape[0]):
                rel_token_matrix[i, j] = relation_matrix[ocr_index[i], ocr_index[j]]

        # prepare element info
        bbox_path = image_path.replace('/images/', '/bbox/')
        bbox_root = os.path.dirname(bbox_path)
        bbox_file = os.path.basename(bbox_path).split('.')[0] + '.json'
        bbox_path = os.path.join(bbox_root, 'preds', bbox_file)
        bbox_info = json.load(open(bbox_path))
        element_pos = self.get_element_pos(bbox_info)
        max_element_len = 3
        if len(element_pos) > 0:
            element_pos = self.deal_element_bbox(element_pos, im_w, im_h)
            element_pos = element_pos[:max_element_len]
            # element_pos = None
        else:
            element_pos = [[0.01, 0.01, 0.02, 0.02]]

        data_dict = {}
        inputs = 'There are some texts in slides, Can you give me a summary of these slides.'
        labels = summary
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(labels, return_tensors="pt").input_ids[0]
        question = self.tokenizer(question, return_tensors="pt").input_ids[0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        data_dict['ocr_box'] = ocr_box
        data_dict['ocr'] = ocr_tokens
        data_dict['patch_pos'] = self.patch_pos
        data_dict['image'] = image_tensor
        data_dict['element_data'] = element_pos
        data_dict['text_dist_graph'] = res_matrix
        data_dict['relation_label'] = rel_token_matrix
        data_dict['question'] = question

        return data_dict

class MYQA_UDOP_Dataset_For_T5_Order(SlideVQA_UDOP_Dataset_For_T5_Single_Order):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        question = sample['qa_pair']['question']
        # image_path = os.path.join(self.image_root,image_path)
        ocr_path = sample['ocr']

        # MAX_IMAGE = 10
        # image_path = image_path[:MAX_IMAGE]

        summary = sample['qa_pair']['answer']

        image_pool = []
        ocr_tokens_pool = []
        ocr_box_pool = []
        element_pos_pool = []
        res_matrix_pool = []
        graph_shape_pool = []
        rel_token_matrix_pool = []

        processor = self.data_args.image_processor
        for path, ocr_p in zip(image_path, ocr_path):
            cur_path = os.path.join(path)
            if self.data_args.ocr_order == 'my':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_v2.json')
            elif self.data_args.ocr_order == 'k3':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_kmean3.json')
            elif self.data_args.ocr_order == 'k4':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_kmean4.json')
            elif self.data_args.ocr_order == 'hac':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_hac.json')
            elif self.data_args.ocr_order == 'r2':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_r2.json')
            elif self.data_args.ocr_order == 'r8':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_r8.json')
            else:
                raise NotImplementedError
            ocr_step = json.load(open(ocr_p))['step']

            image = Image.open(cur_path).convert('RGB')
            image_tensor = self.UDOP_image_transformers(image)
            im_w, im_h = image.size
            words, bbox = self.deal_ocr_info(ocr_p,im_w,im_h)
            ocr_tokens, ocr_box, ocr_index = self.tokenize_ocr(words, bbox, processor)
            max_index = max(ocr_index)


            cal_bbox = torch.tensor(bbox)
            ocr_matrix = get_dist_graph(cal_bbox, max_index + 1)
            res_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0], 6)
            for i in range(ocr_tokens.shape[0]):
                for j in range(ocr_tokens.shape[0]):
                    res_matrix[i, j] = ocr_matrix[ocr_index[i], ocr_index[j]]

            relation_matrix = get_relation_graph(cal_bbox, max_index + 1, step=ocr_step)
            rel_token_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0])
            for i in range(ocr_tokens.shape[0]):
                for j in range(ocr_tokens.shape[0]):
                    rel_token_matrix[i, j] = relation_matrix[ocr_index[i], ocr_index[j]]


            # prepare element info
            bbox_path = cur_path.replace('/images/', '/bbox/')
            bbox_root = os.path.dirname(bbox_path)
            bbox_file = os.path.basename(bbox_path).split('.')[0] + '.json'
            bbox_path = os.path.join(bbox_root, 'preds', bbox_file)
            bbox_info = json.load(open(bbox_path))
            element_pos = self.get_element_pos(bbox_info)
            max_element_len = 3
            if len(element_pos) > 0:
                element_pos = self.deal_element_bbox(element_pos,im_w,im_h)
                element_pos = element_pos[:max_element_len]
                # element_pos = None
            else:
                element_pos = [[0,0,0.02,0.02]]

            image_pool.append(image_tensor)
            ocr_tokens_pool.append(ocr_tokens)
            ocr_box_pool.append(ocr_box)
            res_matrix_pool.append(res_matrix)
            graph_shape_pool.append(res_matrix.shape[0])
            rel_token_matrix_pool.append(rel_token_matrix)
            element_pos_pool.append(element_pos)

        data_dict = {}
        inputs = 'There are some texts in slides, Can you give me a summary of these slides.'
        labels = summary
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(labels, return_tensors="pt").input_ids[0]
        question = self.tokenizer(question, return_tensors="pt").input_ids[0]

        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        # padding
        ocrs = torch.nn.utils.rnn.pad_sequence(
            ocr_tokens_pool,
            batch_first=True,
            padding_value=processor.pad_token_id)

        max_len = max(x.shape[0] for x in ocr_box_pool)
        align_ocr_boxes = []
        for box in ocr_box_pool:
            cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
                                                       dtype=box.dtype)), dim=0)
            align_ocr_boxes.append(cur_box)
        align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        data_dict['seg_data'] = align_ocr_boxes
        data_dict['ocrs'] = ocrs
        data_dict['visual_seg_data'] = self.patch_pos.repeat(align_ocr_boxes.shape[0],1,1)
        data_dict['images'] = torch.stack(image_pool)
        data_dict['element_data'] = element_pos_pool
        data_dict['graph_shape'] = graph_shape_pool
        max_shape_list = max(graph_shape_pool)
        text_graph_tensor = torch.zeros(len(graph_shape_pool), max_shape_list, max_shape_list, res_matrix_pool[0].shape[-1])
        for bs in range(text_graph_tensor.shape[0]):
            text_graph_tensor[bs, :graph_shape_pool[bs], :graph_shape_pool[bs]] = res_matrix_pool[bs]
        data_dict['text_dist_graph'] = text_graph_tensor

        relation_label_tensor = torch.zeros(len(graph_shape_pool), max_shape_list, max_shape_list)
        for bs in range(relation_label_tensor.shape[0]):
            relation_label_tensor[bs, :graph_shape_pool[bs], :graph_shape_pool[bs]] = rel_token_matrix_pool[bs]

        data_dict['relation_label'] = relation_label_tensor
        data_dict['question'] = question.repeat(ocrs.shape[0],1)


        # data_dict['ocr_box'] = ocr_box
        # data_dict['ocr'] = ocr_tokens
        # data_dict['patch_pos'] = self.patch_pos
        # data_dict['image'] = image_tensor
        # data_dict['element_data'] = element_pos
        # data_dict['text_dist_graph'] = res_matrix
        # data_dict['relation_label'] = rel_token_matrix

        return data_dict

class SlideVQA_UDOP_Dataset_For_T5_Order(SlideVQA_UDOP_Dataset_For_T5_Single_Order):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        question = sample['question']
        evidence = sample['evidence']
        evidence_idx = self.get_evidence_idx(image_path, evidence)
        evidence_image_pool = []
        for idx in evidence_idx:
            evidence_image_pool.append(image_path[idx])
        image_path = evidence_image_pool + image_path

        MAX_IMAGE = 10
        image_path = image_path[:MAX_IMAGE]

        summary = sample['answer']

        image_pool = []
        ocr_tokens_pool = []
        ocr_box_pool = []
        element_pos_pool = []
        res_matrix_pool = []
        graph_shape_pool = []
        rel_token_matrix_pool = []

        processor = self.data_args.image_processor
        for path in image_path:
            cur_path = os.path.join(self.image_root,path)
            ocr_p = cur_path.split('.')[0] + '_ocr.json'
            if self.data_args.ocr_order == 'my':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_v2.json')
            elif self.data_args.ocr_order == 'k3':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_kmean3.json')
            elif self.data_args.ocr_order == 'k4':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_kmean4.json')
            elif self.data_args.ocr_order == 'hac':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_hac.json')
            elif self.data_args.ocr_order == 'r2':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_r2.json')
            elif self.data_args.ocr_order == 'r8':
                ocr_p = ocr_p.replace('_ocr.json', '_ocr_cluster_r8.json')
            else:
                raise NotImplementedError
            ocr_step = json.load(open(ocr_p))['step']

            image = Image.open(cur_path).convert('RGB')
            image_tensor = self.UDOP_image_transformers(image)
            im_w, im_h = image.size
            words, bbox = self.deal_ocr_info(ocr_p,im_w,im_h)
            ocr_tokens, ocr_box, ocr_index = self.tokenize_ocr(words, bbox, processor)
            max_index = max(ocr_index)


            cal_bbox = torch.tensor(bbox)
            ocr_matrix = get_dist_graph(cal_bbox, max_index + 1)
            res_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0], 6)
            for i in range(ocr_tokens.shape[0]):
                for j in range(ocr_tokens.shape[0]):
                    res_matrix[i, j] = ocr_matrix[ocr_index[i], ocr_index[j]]

            relation_matrix = get_relation_graph(cal_bbox, max_index + 1, step=ocr_step)
            rel_token_matrix = torch.zeros(ocr_tokens.shape[0], ocr_tokens.shape[0])
            for i in range(ocr_tokens.shape[0]):
                for j in range(ocr_tokens.shape[0]):
                    rel_token_matrix[i, j] = relation_matrix[ocr_index[i], ocr_index[j]]


            # prepare element info
            bbox_path = cur_path.replace('/images/', '/bbox/')
            bbox_root = os.path.dirname(bbox_path)
            bbox_file = os.path.basename(bbox_path).split('.')[0] + '.json'
            bbox_path = os.path.join(bbox_root, 'preds', bbox_file)
            bbox_info = json.load(open(bbox_path))
            element_pos = self.get_element_pos(bbox_info)
            max_element_len = 3
            if len(element_pos) > 0:
                element_pos = self.deal_element_bbox(element_pos,im_w,im_h)
                element_pos = element_pos[:max_element_len]
                # element_pos = None
            else:
                element_pos = [[0,0,0.02,0.02]]

            image_pool.append(image_tensor)
            ocr_tokens_pool.append(ocr_tokens)
            ocr_box_pool.append(ocr_box)
            res_matrix_pool.append(res_matrix)
            graph_shape_pool.append(res_matrix.shape[0])
            rel_token_matrix_pool.append(rel_token_matrix)
            element_pos_pool.append(element_pos)

        data_dict = {}
        inputs = 'There are some texts in slides, Can you give me a summary of these slides.'
        labels = summary
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(labels, return_tensors="pt").input_ids[0]
        question = self.tokenizer(question, return_tensors="pt").input_ids[0]

        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        # padding
        ocrs = torch.nn.utils.rnn.pad_sequence(
            ocr_tokens_pool,
            batch_first=True,
            padding_value=processor.pad_token_id)

        max_len = max(x.shape[0] for x in ocr_box_pool)
        align_ocr_boxes = []
        for box in ocr_box_pool:
            cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
                                                       dtype=box.dtype)), dim=0)
            align_ocr_boxes.append(cur_box)
        align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        data_dict['seg_data'] = align_ocr_boxes
        data_dict['ocrs'] = ocrs
        data_dict['visual_seg_data'] = self.patch_pos.repeat(align_ocr_boxes.shape[0],1,1)
        data_dict['images'] = torch.stack(image_pool)
        data_dict['element_data'] = element_pos_pool
        data_dict['graph_shape'] = graph_shape_pool
        max_shape_list = max(graph_shape_pool)
        text_graph_tensor = torch.zeros(len(graph_shape_pool), max_shape_list, max_shape_list, res_matrix_pool[0].shape[-1])
        for bs in range(text_graph_tensor.shape[0]):
            text_graph_tensor[bs, :graph_shape_pool[bs], :graph_shape_pool[bs]] = res_matrix_pool[bs]
        data_dict['text_dist_graph'] = text_graph_tensor

        relation_label_tensor = torch.zeros(len(graph_shape_pool), max_shape_list, max_shape_list)
        for bs in range(relation_label_tensor.shape[0]):
            relation_label_tensor[bs, :graph_shape_pool[bs], :graph_shape_pool[bs]] = rel_token_matrix_pool[bs]

        data_dict['relation_label'] = relation_label_tensor
        data_dict['question'] = question.repeat(ocrs.shape[0],1)


        # data_dict['ocr_box'] = ocr_box
        # data_dict['ocr'] = ocr_tokens
        # data_dict['patch_pos'] = self.patch_pos
        # data_dict['image'] = image_tensor
        # data_dict['element_data'] = element_pos
        # data_dict['text_dist_graph'] = res_matrix
        # data_dict['relation_label'] = rel_token_matrix

        return data_dict

class Slideshare_LayoutLM_Dataset_For_T5(Slideshare_UDOP_Dataset_For_T5):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        ocr_path = sample['ocr']
        summary = sample['summary']


        processor = self.data_args.image_processor
        image_pool = []
        ocr_tokens_pool = []
        ocr_box_pool = []
        for path, ocr_p in zip(image_path, ocr_path):
            cur_path = os.path.join(path)
            image = Image.open(cur_path).convert('RGB')

            im_w, im_h = image.size
            words, bbox = self.deal_ocr_info(ocr_p,im_w,im_h)
            new_box = []
            for box in bbox:
                cur_box = []
                for point in box:
                    cur_box.append(int(point * 1000))
                new_box.append(cur_box)
            image_pool.append(image)
            ocr_tokens_pool.append(words)
            ocr_box_pool.append(new_box)


        data_dict = {}
        ocr_base_inputs = processor(image_pool, ocr_tokens_pool, boxes=ocr_box_pool, return_tensors="pt", padding=True,
                                                  truncation=True, max_length=512)
        data_dict['ocr_ids'] = ocr_base_inputs.input_ids
        data_dict['images'] = ocr_base_inputs.pixel_values
        data_dict['ocr_attn'] = ocr_base_inputs.attention_mask
        data_dict['ocr_bbox'] = ocr_base_inputs.bbox
        inputs = 'There are some texts in slides, Can you give me a summary of these slides.'
        labels = summary
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(labels, return_tensors="pt").input_ids[0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        # image exist in the data

        # padding
        # ocrs = torch.nn.utils.rnn.pad_sequence(
        #     ocr_tokens_pool,
        #     batch_first=True,
        #     padding_value=processor.pad_token_id)
        #
        # max_len = max(x.shape[0] for x in ocr_box_pool)
        # align_ocr_boxes = []
        # for box in ocr_box_pool:
        #     cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
        #                                                dtype=box.dtype)), dim=0)
        #     align_ocr_boxes.append(cur_box)
        # align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        # data_dict['seg_data'] = align_ocr_boxes
        # data_dict['ocrs'] = ocrs
        # data_dict['visual_seg_data'] = self.patch_pos.repeat(align_ocr_boxes.shape[0],1,1)
        # data_dict['images'] = torch.stack(image_pool)
        # data_dict['evidence'] = evidence_idx
        return data_dict

class MYQA_LayoutLM_Dataset_For_T5(Slideshare_UDOP_Dataset_For_T5):
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        ocr_path = sample['ocr']
        qa_pair = sample['qa_pair']
        summary = qa_pair['answer']
        question = qa_pair['question']


        processor = self.data_args.image_processor
        image_pool = []
        ocr_tokens_pool = []
        ocr_box_pool = []
        for path, ocr_p in zip(image_path, ocr_path):
            cur_path = os.path.join(path)
            image = Image.open(cur_path).convert('RGB')

            im_w, im_h = image.size
            words, bbox = self.deal_ocr_info(ocr_p,im_w,im_h)
            new_box = []
            for box in bbox:
                cur_box = []
                for point in box:
                    cur_box.append(int(point * 1000))
                new_box.append(cur_box)
            words = [question] + words
            new_box = [[0,0,0,0]] + new_box
            image_pool.append(image)
            ocr_tokens_pool.append(words)
            ocr_box_pool.append(new_box)


        data_dict = {}
        ocr_base_inputs = processor(image_pool, ocr_tokens_pool, boxes=ocr_box_pool, return_tensors="pt", padding=True,
                                                  truncation=True, max_length=512)
        data_dict['ocr_ids'] = ocr_base_inputs.input_ids
        data_dict['images'] = ocr_base_inputs.pixel_values
        data_dict['ocr_attn'] = ocr_base_inputs.attention_mask
        data_dict['ocr_bbox'] = ocr_base_inputs.bbox
        inputs = 'There are some texts in slides, Can you give me a summary of these slides.'
        labels = summary
        input_ids = self.tokenizer(inputs, return_tensors="pt").input_ids[0]
        labels = self.tokenizer(labels, return_tensors="pt").input_ids[0]
        data_dict['input_ids'] = input_ids
        data_dict['labels'] = labels

        # image exist in the data

        # padding
        # ocrs = torch.nn.utils.rnn.pad_sequence(
        #     ocr_tokens_pool,
        #     batch_first=True,
        #     padding_value=processor.pad_token_id)
        #
        # max_len = max(x.shape[0] for x in ocr_box_pool)
        # align_ocr_boxes = []
        # for box in ocr_box_pool:
        #     cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
        #                                                dtype=box.dtype)), dim=0)
        #     align_ocr_boxes.append(cur_box)
        # align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        # data_dict['seg_data'] = align_ocr_boxes
        # data_dict['ocrs'] = ocrs
        # data_dict['visual_seg_data'] = self.patch_pos.repeat(align_ocr_boxes.shape[0],1,1)
        # data_dict['images'] = torch.stack(image_pool)
        # data_dict['evidence'] = evidence_idx
        return data_dict

class Slides_VQA_UDOP_Dataset_WOCR(Slides_VQA_UDOP_Dataset):
    def get_ocr_prompt(self, image_path,evidence):
        ocr_pool = []
        for idx in evidence:
            ocr_path = image_path[idx]
            ocr_path = ocr_path.split('.')[0] + '_ocr.json'
            ocr_data = json.load(open(os.path.join(self.image_root,ocr_path)))
            ocrs = [data[1][0] for data in ocr_data]
            ocr_pool.extend(ocrs)
        return ' '.join(ocr_pool)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample = sources[0]
        image_path = sample['image']
        question = sample['question']
        if '<image>' not in question:
            question = '<image>\n' + question
        answer = sample['answer']
        evidence = sample['evidence']
        evidence_idx = self.get_evidence_idx(image_path,evidence)
        ocr_prompt = self.get_ocr_prompt(image_path, evidence_idx)
        processor = self.data_args.image_processor
        image_pool = []
        ocr_tokens_pool = []
        ocr_box_pool = []
        for path in image_path:
            cur_path = os.path.join(self.image_root, path)
            image = Image.open(cur_path).convert('RGB')
            image_tensor = self.UDOP_image_transformers(image)
            image_pool.append(image_tensor)
            im_w, im_h = image.size
            words, bbox = self.deal_ocr_info(path,im_w,im_h)
            ocr_tokens, ocr_box = self.tokenize_ocr(words, bbox, processor, max_len=512)
            ocr_tokens_pool.append(ocr_tokens)
            ocr_box_pool.append(ocr_box)


        sources = [[{'from': 'human', 'value':  question + f' The OCR in slides are: {ocr_prompt}'},
                    {'from': 'gpt', 'value': answer}]]
        sources = preprocess_multimodal(
            copy.deepcopy(sources),
            self.data_args)

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data

        # padding
        ocrs = torch.nn.utils.rnn.pad_sequence(
            ocr_tokens_pool,
            batch_first=True,
            padding_value=processor.pad_token_id)

        max_len = max(x.shape[0] for x in ocr_box_pool)
        align_ocr_boxes = []
        for box in ocr_box_pool:
            cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
                                                       dtype=box.dtype)), dim=0)
            align_ocr_boxes.append(cur_box)
        align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        data_dict['seg_data'] = align_ocr_boxes
        data_dict['ocrs'] = ocrs
        data_dict['visual_seg_data'] = self.patch_pos.repeat(align_ocr_boxes.shape[0],1,1)
        data_dict['images'] = torch.stack(image_pool)
        # data_dict['evidence'] = evidence_idx
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

@dataclass
class DataCollatorForSupervisedDatasetUDOP(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __init__(self, tokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        ocrs = [instance['ocr'] for instance in instances]
        ocrs = torch.nn.utils.rnn.pad_sequence(
            ocrs,
            batch_first=True,
            padding_value=self.image_processor.pad_token_id)
        batch['ocrs'] = ocrs

        ocr_boxes = [instance['ocr_box'] for instance in instances]
        max_len = max(x.shape[0] for x in ocr_boxes)
        align_ocr_boxes = []
        for box in ocr_boxes:
            cur_box = torch.cat((box,torch.zeros((max_len - box.shape[0], box.shape[1]),
                                                       dtype=box.dtype)), dim=0)
            align_ocr_boxes.append(cur_box)
        align_ocr_boxes = torch.stack(align_ocr_boxes,dim=0)
        batch['seg_data'] = align_ocr_boxes

        patch_pos = [instance['patch_pos'] for instance in instances]
        batch['visual_seg_data'] = torch.stack(patch_pos,dim=0)


        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        if 'element_data' in instances[0]:
            element_data = [instance['element_data'] for instance in instances]
            batch['element_data'] = element_data

        if 'text_dist_graph' in instances[0]:
            text_dist_graph = [instance['text_dist_graph'] for instance in instances]
            shape_list = [graph.shape[0] for graph in text_dist_graph]
            max_shape_list = max(shape_list)
            text_graph_tensor = torch.zeros(len(shape_list),max_shape_list,max_shape_list,text_dist_graph[0].shape[-1])
            for bs in range(text_graph_tensor.shape[0]):
                text_graph_tensor[bs,:shape_list[bs],:shape_list[bs]] = text_dist_graph[bs]

            batch['text_dist_graph'] = text_graph_tensor
            batch['graph_shape'] = shape_list

        if 'relation_label' in instances[0]:
            relation_label = [instance['relation_label'] for instance in instances]
            shape_list = [graph.shape[0] for graph in relation_label]
            max_shape_list = max(shape_list)
            relation_label_tensor = torch.zeros(len(shape_list),max_shape_list,max_shape_list)
            for bs in range(relation_label_tensor.shape[0]):
                relation_label_tensor[bs,:shape_list[bs],:shape_list[bs]] = relation_label[bs]

            batch['relation_label'] = relation_label_tensor

        if 'question' in instances[0]:
            question = [instance['question'] for instance in instances]
            question = torch.nn.utils.rnn.pad_sequence(
                question,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)

            batch['question'] = question



        return batch

@dataclass
class DataCollatorForSupervisedDatasetUDOPSlideVQA(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __init__(self, tokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        batch['ocrs'] = [instance['ocrs'] for instance in instances]

        batch['seg_data'] = [instance['seg_data'] for instance in instances]
        batch['visual_seg_data'] = [instance['visual_seg_data'] for instance in instances]
        batch['images'] = [instance['images'] for instance in instances]
        if 'element_data' in instances[0]:
            batch['element_data'] = [instance['element_data'] for instance in instances]
            batch['graph_shape'] = [instance['graph_shape'] for instance in instances]
            batch['text_dist_graph'] = [instance['text_dist_graph'] for instance in instances]
            batch['relation_label'] = [instance['relation_label'] for instance in instances]
        if 'question' in instances[0]:
            batch['question'] = [instance['question'] for instance in instances]

        return batch

@dataclass
class DataCollatorForSupervisedDatasetLayoutLM(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __init__(self, tokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        batch['ocr_ids'] = [instance['ocr_ids'] for instance in instances]

        batch['ocr_attn'] = [instance['ocr_attn'] for instance in instances]
        batch['ocr_bbox'] = [instance['ocr_bbox'] for instance in instances]
        batch['images'] = [instance['images'] for instance in instances]

        return batch

@dataclass
class DataCollatorForSupervisedDatasetSlideVQA(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __init__(self, tokenizer, image_processor=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'evidence' in instances[0]:
            evidence = [instance['evidence'] for instance in instances]
            batch['evidence'] = evidence

        if 'text_input_ids' in instances[0]:
            text_input_ids = [instance['text_input_ids'] for instance in instances]
            text_input_ids = torch.nn.utils.rnn.pad_sequence(
                text_input_ids,
                batch_first=True,
                padding_value=-100)
            batch['text_input_ids'] = text_input_ids



        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # if all(x is not None and x.shape == images[0].shape for x in images):
            #     batch['images'] = torch.stack(images)
            # else:
            batch['images'] = images

        return batch

@dataclass
class DataCollatorForSupervisedDatasetWOCR(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(self, tokenizer, ocr_base_processor):
        self.tokenizer = tokenizer
        self.ocr_base_processor = ocr_base_processor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        if 'ocr_base_image' in instances[0]:
            ocr_base_image = [instance['ocr_base_image'] for instance in instances]
            ocr = [instance['ocr'] for instance in instances]
            ocr_box = [instance['ocr_box'] for instance in instances]
            ocr_base_inputs = self.ocr_base_processor(ocr_base_image, ocr, boxes=ocr_box, return_tensors="pt", padding=True, truncation=True, max_length=512)
            batch['ocr_ids'] = ocr_base_inputs.input_ids
            batch['ocr_base_images'] = ocr_base_inputs.pixel_values
            batch['ocr_attn'] = ocr_base_inputs.attention_mask
            batch['ocr_bbox'] = ocr_base_inputs.bbox


        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_slidevqa_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slides_VQA_Dataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDatasetSlideVQA(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_slideshare_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slideshare_Dataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDatasetSlideVQA(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
def make_slidevqa_data_module_wattn(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slides_VQA_Dataset_WTG(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDatasetSlideVQA(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def make_slidevqa_data_module_wocr(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slides_VQA_Dataset_WOCR(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDatasetSlideVQA(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_mls_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MLS_Dataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
def make_layout_pretrain_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Layout_Pretrain_Dataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_layout_pretrain_data_module_wocr(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Layout_Pretrain_Dataset_WOCR(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_mix_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Mixed_MLS_Dataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetWOCR(tokenizer=tokenizer,ocr_base_processor=data_args.image_processor['ocr_base'])
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_mix_data_layout_pretrain_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Mixed_Layout_Pretrain_Dataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetWOCR(tokenizer=tokenizer,ocr_base_processor=data_args.image_processor['ocr_base'])
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_udop_data_layout_pretrain_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = UDOP_Layout_Pretrain_Dataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOP(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_udop_data_slidevqa_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slides_VQA_UDOP_Dataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOPSlideVQA(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_udop_data_slideshare_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slideshare_UDOP_Dataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOPSlideVQA(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_udop_data_slideshare_module_for_t5(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slideshare_UDOP_Dataset_For_T5(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOPSlideVQA(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_udop_data_myqa_module_for_t5(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MYQA_UDOP_Dataset_For_T5(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOPSlideVQA(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_udop_data_slideshare_module_for_t5_single(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slideshare_UDOP_Dataset_For_T5_Single(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOP(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_udop_data_slideshare_module_for_t5_single_order(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slideshare_UDOP_Dataset_For_T5_Single_Order(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOP(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_udop_data_slideshare_module_for_t5_order(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slideshare_UDOP_Dataset_For_T5_Order(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOPSlideVQA(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def make_udop_data_slidevqa_module_for_t5_single_order(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SlideVQA_UDOP_Dataset_For_T5_Single_Order(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOP(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_udop_data_myqa_module_for_t5_single_order(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MYQA_UDOP_Dataset_For_T5_Single_Order(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOP(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
def make_udop_data_myqa_module_for_t5_order(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MYQA_UDOP_Dataset_For_T5_Order(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOPSlideVQA(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
def make_udop_data_slidevqa_module_for_t5_order(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SlideVQA_UDOP_Dataset_For_T5_Order(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOPSlideVQA(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_layoutlm_data_slideshare_module_for_t5(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slideshare_LayoutLM_Dataset_For_T5(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetLayoutLM(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
def make_layoutlm_data_myqa_module_for_t5(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MYQA_LayoutLM_Dataset_For_T5(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetLayoutLM(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_udop_data_slidevqa_module_wocr(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = Slides_VQA_UDOP_Dataset_WOCR(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    # data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDatasetUDOPSlideVQA(tokenizer=tokenizer,image_processor=data_args.image_processor)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def train(attn_implementation=None):
    global local_rank
    model_map = {
        'llava_phi':LlavaPhiForCausalLM,
        'mix': MixedLlavaPhiForCausalLM,
        'mix_decouple': MixedLlavaPhiForCausalLMDecoupleLayout,
        'merge_phi': MergeLlavaPhiForCausalLM,
        'merge_phi_wattn': MergeLlavaPhiForCausalLMWTG,
        'udop_phi': UDOPLlavaPhiForCausalLM,
        'udop_phi_merge': UDOPLlavaPhiForCausalLMMerging,
        'udop_t5': UDOP2T5,
        'udop_t5_single': UDOP2T5Single,
        'udop_t5_single_order': UDOP2T5SingleOrder,
        'udop_t5_order': UDOP2T5SingleOrderMerge,
        'layoutlm_t5': LayoutLM2T5,
    }

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    model_map_name = getattr(model_args,'model_map_name','llava_phi')

    model = model_map[model_map_name].from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args
    )

    if model_map_name == 'mix_decouple' and model.is_layoutlm is False:
        model.init_layoutlm(model_args.ocr_base_tower)
    elif model_map_name == 'mix_decouple' and model.is_layoutlm:
        print('skip init layoutlm in train.py')

    if model_map_name == 'merge_phi_wattn' and model.is_text_encoder is False:
        model.init_text_encoder(model_args.vision_tower)
    elif model_map_name == 'merge_phi_wattn' and model.is_text_encoder:
        print('skip init text encoder in train.py')
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif 't5' in model_map_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            '/home/emzhang/data/t5-large',
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "llava_phi":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    elif model_args.version != "t5":
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.version != "t5" and (model_args.vision_tower is not None or model_args.ocr_free_tower is not None or model_args.ocr_base_tower is not None):
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        elif model_args.tune_bb:
            # model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            for p in model.get_model().vision_tower.parameters():
                p.requires_grad = True
        else:
            model.model.vision_tower.requires_grad_(False)

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False


        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    else:
        model.initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        vision_tower = model.vision_tower
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.encoder.requires_grad_(False)
        if not model_args.tune_bb:
            model.vision_tower.requires_grad_(False)



    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if model_args.task == 'lecture_gen':
        if model_map_name == 'llava_phi':
            data_module = make_mls_data_module(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name in ['mix','mix_decouple']:
            data_module = make_mix_data_module(tokenizer=tokenizer,
                                                      data_args=data_args)
        else:
            raise NotImplementedError
    elif model_args.task == 'layout_pretrain':
        if model_map_name == 'llava_phi':
            data_module = make_layout_pretrain_data_module(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name in ['mix','mix_decouple']:
            data_module = make_mix_data_layout_pretrain_module(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name in ['udop_phi']:
            data_module = make_udop_data_layout_pretrain_module(tokenizer=tokenizer,
                                                               data_args=data_args)

        else:
            raise NotImplementedError
    elif model_args.task == 'layout_pretrain_wocr':
        if model_map_name == 'llava_phi':
            data_module = make_layout_pretrain_data_module_wocr(tokenizer=tokenizer,
                                                      data_args=data_args)
    elif model_args.task == 'slide_vqa':
        if model_map_name == 'merge_phi':
            data_module = make_slidevqa_data_module(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name == 'merge_phi_wattn':
            data_module = make_slidevqa_data_module_wattn(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name == 'udop_phi_merge':
            data_module = make_udop_data_slidevqa_module(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name == 'udop_t5_single_order':
            data_module = make_udop_data_slidevqa_module_for_t5_single_order(tokenizer=tokenizer,
                                                      data_args=data_args)


        elif model_map_name == 'udop_t5_order':
            data_module = make_udop_data_slidevqa_module_for_t5_order(tokenizer=tokenizer,
                                                      data_args=data_args)
        else:
            raise NotImplementedError
    elif model_args.task == 'slide_vqa_wocr':
        if model_map_name == 'merge_phi':
            data_module = make_slidevqa_data_module_wocr(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name == 'udop_phi_merge':
            data_module = make_udop_data_slidevqa_module_wocr(tokenizer=tokenizer,
                                                      data_args=data_args)
        else:
            raise NotImplementedError
    elif model_args.task == 'my_summary':
        if model_map_name == 'merge_phi':
            data_module = make_slideshare_data_module(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name == 'udop_phi_merge':
            data_module = make_udop_data_slideshare_module(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name == 'udop_t5':
            data_module = make_udop_data_slideshare_module_for_t5(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name == 'udop_t5_single':
            data_module = make_udop_data_slideshare_module_for_t5_single(tokenizer=tokenizer,
                                                                  data_args=data_args)
        elif model_map_name == 'udop_t5_single_order':
            data_module = make_udop_data_slideshare_module_for_t5_single_order(tokenizer=tokenizer,
                                                                  data_args=data_args)
        elif model_map_name == 'udop_t5_order':
            data_module = make_udop_data_slideshare_module_for_t5_order(tokenizer=tokenizer,
                                                                  data_args=data_args)

        elif model_map_name == 'layoutlm_t5':
            data_module = make_layoutlm_data_slideshare_module_for_t5(tokenizer=tokenizer,
                                                      data_args=data_args)
        else:
            raise NotImplementedError
    elif model_args.task == 'my_qa':
        if model_map_name == 'udop_t5':
            data_module = make_udop_data_myqa_module_for_t5(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name == 'layoutlm_t5':
            data_module = make_layoutlm_data_myqa_module_for_t5(tokenizer=tokenizer,
                                                      data_args=data_args)
        elif model_map_name == 'udop_t5_single_order':
            data_module = make_udop_data_myqa_module_for_t5_single_order(tokenizer=tokenizer,
                                                      data_args=data_args)


        elif model_map_name == 'udop_t5_order':
            data_module = make_udop_data_myqa_module_for_t5_order(tokenizer=tokenizer,
                                                      data_args=data_args)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


    torch.backends.cudnn.enable = True


    print_trainable_parm(model, prefix='final')
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # safe_save_model_for_hf_trainer(trainer=trainer,
    #                                output_dir=training_args.output_dir)
    # return

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
