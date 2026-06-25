import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class MixDocProjector(nn.Module):
    def __init__(self,config):
        super().__init__()
        ocr_free_size = config.ocr_free_size
        self.ocr_free_projector = nn.Linear(ocr_free_size,config.hidden_size)
        self.ocr_base_projector = nn.Linear(config.ocr_base_size,config.hidden_size)

    def forward(self,features):
        ocr_free_features = features['ocr_free_feature']
        ocr_base_feature = features['ocr_base_feature']
        ocr_free_x = self.ocr_free_projector(ocr_free_features)
        ocr_base_x = self.ocr_base_projector(ocr_base_feature)
        output = torch.concat((ocr_free_x,ocr_base_x),dim=1)


        return output


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == 'mix':
        return MixDocProjector(config)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
