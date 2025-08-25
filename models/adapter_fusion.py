import torch
import torch.nn as nn

from copy import deepcopy
from enum import Enum
from typing import Union, List, Optional, Dict, Callable, Any
from diffusers import T2IAdapter
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .blocks import LayerNorm, ResidualAttentionBlock, ResnetBlock
from .utils import zero_module

# -- Helpers -- #
class AvailableAdapters(Enum):
    sketch = 0
    keypose = 1
    seg = 2
    depth = 3
    canny = 4
    style = 5
    color = 6
    openpose = 7

# -- Adapter & Fuser Models -- #
class T2IAdapterXL(ModelMixin, ConfigMixin):
    """
    The T2IAdapterXL class is the Adapter adapted for Stable Diffusion XL.
    Model taken from https://github.com/TencentARC/T2I-Adapter and modified to work with Pipelines
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        kernel_size: int=1,
        downscale_factor: int = 16,
    ):
        super().__init__()
        in_channels = in_channels * downscale_factor**2
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, 1, 1)
        self.body = nn.ModuleList()
        for idx in range(len(channels)):
            for jdx in range(num_res_blocks):
                if (idx == 1) and (jdx == 0):
                    self.body.append(ResnetBlock(channels[idx-1], channels[idx], False, ksize=kernel_size))
                elif (idx == 2) and (jdx == 0):
                    self.body.append(ResnetBlock(channels[idx-1], channels[idx], True, ksize=kernel_size))
                else:
                    self.body.append(ResnetBlock(channels[idx], channels[idx], False, ksize=kernel_size))
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.total_downscale_factor = downscale_factor * 2 ** (len(channels) - 1)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.conv_in(x)
        features = []
        for idx in range(len(self.channels)):
            for jdx in range(self.num_res_blocks):
                kdx = idx * self.num_res_blocks + jdx
                x = self.body[kdx](x)
            features.append(x)
        return features 


class Fuser(ModelMixin, ConfigMixin):
    """
    The Fuser class to fuse different Adapter models just one vector of features.
    Model taken from https://github.com/TencentARC/T2I-Adapter
    :param: channels, list. NUmber of channels of the Unet where the features need to be summed.
    :param: layer_depth, int. Depth of the fuser layers.
    :param: num_head, int. Number of heads for the attention layer.
    :param: num_layers, int. Number of layers for the attention layer.
    """
    @register_to_config
    def __init__(
        self,
        channels: List[int] = [320, 640, 1280, 1280],
        layer_depth: int = 768,
        num_head: int = 8,
        num_layers: int = 3
    ):
        super().__init__()
        scale = layer_depth ** 0.5
        # 16, maybe large enough for the number of adapters?
        self.task_embedding = nn.Parameter(scale * torch.randn(16, layer_depth))
        self.positional_embedding = nn.Parameter(scale * torch.randn(len(channels), layer_depth))
        self.spatial_feat_mapping = nn.ModuleList()
        self.spatial_ch_projs = nn.ModuleList()
        for channel in channels:
            self.spatial_feat_mapping.append(nn.Sequential(
                nn.SiLU(),
                nn.Linear(channel, layer_depth),
            ))
            self.spatial_ch_projs.append(zero_module(nn.Linear(layer_depth, channel)))
        self.transformer_layers = nn.Sequential(*[
            ResidualAttentionBlock(layer_depth, num_head) for _ in range(num_layers)
        ])
        self.ln_post = LayerNorm(layer_depth)
        self.ln_pre = LayerNorm(layer_depth)
        self.seq_proj = nn.Parameter(torch.zeros(layer_depth, layer_depth))
    
    def enable_half_bypass_for_layer_norms(self):
        for name, module in deepcopy(self).named_modules():
            if module.__class__.__name__ == 'LayerNorm':
                parts = name.split('.')
                if len(parts) > 1:
                    setattr(getattr(self, parts[0])[int(parts[1])], parts[2], module.type(torch.float32))
                else:
                    setattr(self, name, module.type(torch.float32))

    def forward(self, features, cond_names: List[str]):
        if len(features) == 0:
            return None, None
        inputs = []
        for feature, cond_name in zip(features, cond_names):
            task_idx = getattr(AvailableAdapters, cond_name).value
            if not isinstance(feature, list):
                inputs.append(feature + self.task_embedding[task_idx])
                continue
            feat_seq = []
            for idx, feature_map in enumerate(feature):
                feature_vec = torch.mean(feature_map, dim=(2, 3))
                feature_vec = self.spatial_feat_mapping[idx](feature_vec)
                feat_seq.append(feature_vec)
            feat_seq = torch.stack(feat_seq, dim=1)  # Nx4xC
            feat_seq = feat_seq + self.task_embedding[task_idx]
            feat_seq = feat_seq + self.positional_embedding
            inputs.append(feat_seq)

        x = torch.cat(inputs, dim=1) # NxLxC
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.transformer_layers(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.ln_post(x)

        ret_feat_map = None
        ret_feat_seq = None
        cur_seq_idx = 0
        for feature in features:
            if not isinstance(features, list):
                length = feature.size(1)
                transformed_feature = feature * ((x[:, cur_seq_idx:cur_seq_idx+length] @ self.seq_proj) + 1)
                ret_feat_seq = transformed_feature if ret_feat_seq is None else torch.cat(
                    [ret_feat_seq, transformed_feature], dim=1
                )
                cur_seq_idx += length
                continue

            length = len(feature)
            transformed_feature_list = []
            for idx in range(length):
                alpha = self.spatial_ch_projs[idx](x[:, cur_seq_idx+idx])
                alpha = alpha.unsqueeze(-1).unsqueeze(-1) + 1
                transformed_feature_list.append(feature[idx] * alpha)
            ret_feat_map = transformed_feature_list if ret_feat_map is None else list(map(
                lambda x, y: x + y, ret_feat_map, transformed_feature_list
            ))
            cur_seq_idx += length

        assert cur_seq_idx == x.size(1)

        return ret_feat_map, ret_feat_map


# -- Fusion Wrappers -- #
class MultiFuserAdapter(ModelMixin):
    """
    MultiFuserAdapter is a wrapper class that wraps a `T2IAdapter` and a `Fuser` together.
    If fuser is None it has the same behaviour as using MultiAdapter from diffusers.

    Arguments:
        :param: adapter, T2IAdapter or List[T2IAdapter]. The adapter(s) to be fused.
        :param: fuser, Fuser. The fuser to be used. If needed.
        :param: adapter_cond_names, List[str]. The names of the adapters.
    """
    def __init__(
        self,
        adapters: Union[T2IAdapter, T2IAdapterXL, List[T2IAdapter]],
        fuser: Optional[Fuser] = None,
        adapter_cond_names: Optional[List[str]] = None,
    ):
        super(MultiFuserAdapter, self).__init__()

        if isinstance(adapters, T2IAdapter) or isinstance(adapters, T2IAdapterXL):
            adapters = [adapters]
        self.num_adapter = len(adapters)
        self.adapters = nn.ModuleList(adapters)
        self.fuser = fuser
        self.adapter_cond_names = adapter_cond_names

    @property
    def total_downscale_factor(self):
        return self.adapters[0].total_downscale_factor

    def mix_features(self, x, y, weights=None):
        assert len(x)==len(y), f'Features need to have the same number of features. Got {len(x)} != {len(y)}.'
        if weights is None:
            weights = torch.tensor([1.0] * len(x))
        else:
            assert len(weights)==2, f'The length of weights should be 2. Got {len(weights)}.'
        adapter_features = [x, y]
        if self.fuser is None:
            # Weighted sum of the different adapter features 
            accum_state = [0] * len(x)
            for features, w in zip(adapter_features, weights):
                for idx, feature_map in enumerate(features):
                    accum_state[idx] = feature_map * w + accum_state[idx]
        else:
            # Weight the adapter features and fuse them together
            for adapter_idx, (features, w) in enumerate(zip(adapter_features, weights)):
                for feat_idx, feature_map in enumerate(features):
                    adapter_features[adapter_idx][feat_idx] = feature_map * w
            accum_state = self.fuser(adapter_features, self.adapter_cond_names)[0]
        return accum_state

    def forward(
        self,
        xs: List[torch.Tensor],
        adapter_weights: Optional[Union[float, List[float]]] = None
    ) -> List[torch.Tensor]:
        """
        Args:
            xs: (batch, channel, height, width) input images for multiple adapter models.
            adapter_weights: A float or a list of floats representing the weight which will be multiply
                to each adapter's output before adding them together.
        """
        if adapter_weights is None:
            adapter_weights = torch.tensor([1.0] * self.num_adapter)
        else:
            if isinstance(adapter_weights, float): adapter_weights = [adapter_weights]
            adapter_weights = torch.tensor(adapter_weights)

        if len(xs) != self.num_adapter:
            raise ValueError(
                f"Expecting multi-adapter's input have length equal to the number of adapters, but got {len(xs)}"
                f"by num_adapter: {xs.shape[1]} % {self.num_adapter} != 0"
            )
        adapter_features = [adapter(x) for x, adapter in zip(xs, self.adapters)]
        if self.fuser is None:
            # Weighted sum of the different adapter features 
            accum_state = [0] * len(adapter_features[0])
            for features, w in zip(adapter_features, adapter_weights):
                for idx, feature_map in enumerate(features):
                    accum_state[idx] = feature_map * w + accum_state[idx]
        else:
            # Weight the adapter features and fuse them together
            for adapter_idx, (features, w) in enumerate(zip(adapter_features, adapter_weights)):
                for feat_idx, feature_map in enumerate(features):
                    adapter_features[adapter_idx][feat_idx] = feature_map * w
            accum_state = self.fuser(adapter_features, self.adapter_cond_names)[0]
        return accum_state