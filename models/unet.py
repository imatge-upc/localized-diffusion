import torch
from dataclasses import dataclass
from typing import Any, Union, List, Optional, Callable, Dict, Tuple

from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput

@dataclass
class Unet2DConditionWrappingOutput(UNet2DConditionOutput):
    """
    Unet2DCondition Output in order to return the samples from the downsample and mid layers and residual layers 
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
        down_block_res_samples (`Tuple` of size `(num_down_residual_blocks)`):
            Residual samples from the down blocks of the Unet.
        down_block_samples (`Tuple` of size `(num_down_blocks)`):
            Feature samples from the down_blocks of the Unet.
        mid_block_res_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Residual samples from the mid block of the Unet
    """
    down_block_res_samples: Tuple[torch.FloatTensor]
    down_block_samples: Tuple[torch.FloatTensor]
    mid_block_res_sample: torch.FloatTensor

class Unet2DConditionWrapping(UNet2DConditionModel):
    def __init__(self, *args, down_residual_factor=0.5, down_samples_factor=0.5, mid_sample_factor=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_residuals_factor = down_residual_factor
        self.down_samples_factor = down_samples_factor
        self.mid_sample_factor = mid_sample_factor

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        additional_down_block_samples: Optional[Tuple[torch.Tensor]] = None,
        additional_down_block_res_samples: Optional[Tuple[torch.Tensor]] = None,
        additional_mid_block_res_sample: Optional[torch.Tensor] = None,
        return_intermediate_features=False,
    ) -> Tuple:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`diffusers.models.unet_2d_condition.UNet2DConditionOutput`] or [`~models.unet.Unet2DConditionWrappingOutput`]:
            [`~models.unet.Unet2DConditionWrappingOutput`] if `return_intermediate_features` is True,
            otherwise [`diffusers.models.unet_2d_condition.UNet2DConditionOutput`].
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            print("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_samples = ()
        down_block_res_samples = (sample,)
        for idx, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                
            if additional_down_block_samples is not None:
                sample = (sample + additional_down_block_samples[idx]) * self.down_samples_factor

            down_block_samples += (sample,)
            down_block_res_samples += res_samples
            
        if additional_down_block_res_samples is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, additional_down_block_res_sample in zip(
                down_block_res_samples, additional_down_block_res_samples
            ):
                new_down_block_res_samples += (
                    (down_block_res_sample + additional_down_block_res_sample) * self.down_residuals_factor,
                )
            down_block_res_samples = new_down_block_res_samples
        
        usable_down_block_res_samples = down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample, emb, encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs
            )
            mid_block_res_sample = sample.clone()

        if additional_mid_block_res_sample is not None:
            sample = (sample + additional_mid_block_res_sample) * self.mid_sample_factor
        
        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = usable_down_block_res_samples[-len(upsample_block.resnets) :]
            usable_down_block_res_samples = usable_down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        if return_intermediate_features:
            return Unet2DConditionWrappingOutput(
                sample=sample,
                down_block_res_samples=down_block_res_samples,
                down_block_samples=down_block_samples,
                mid_block_res_sample=mid_block_res_sample
            )
        else:
            return UNet2DConditionOutput(sample=sample)
