import torch

from typing import Union, List, Optional, Dict, Callable, Any
from PIL import Image

from diffusers import StableDiffusionAdapterPipeline, MultiAdapter
from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_adapter import (
    StableDiffusionAdapterPipelineOutput, _preprocess_adapter_image
)

class StableDiffusionMaskedAdapterPipeline(StableDiffusionAdapterPipeline):
    """Modified version of StableDiffusionAdapterPipeline to support masked conditions"""
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]] = None,
        mask: Union[torch.Tensor, Image.Image, List[Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        adapter_conditioning_scale: Union[float, List[float]] = 1.0,
    ):
        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)
        device = self._execution_device

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        is_multi_adapter = isinstance(self.adapter, MultiAdapter)
        if is_multi_adapter:
            adapter_input = [_preprocess_adapter_image(img, height, width).to(device) for img in image]
            n, c, h, w = adapter_input[0].shape
            adapter_input = torch.stack([x.reshape([n * c, h, w]) for x in adapter_input])
        else:
            adapter_input = _preprocess_adapter_image(image, height, width).to(device)
        adapter_input = adapter_input.to(self.adapter.dtype)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        adapter_state = self.adapter(adapter_input)

        # Mask Adapter State if mask is present
        if mask is not None:
            adapter_state = [
                state * torch.nn.functional.interpolate(
                    mask, size=(mask.shape[-1] // size)
                ) for state, size in zip(adapter_state, [2**i for i in range(3,7)])
            ]
        
        # Scale Adapter State if adapter_conditioning_scale is present
        if isinstance(adapter_conditioning_scale, float):
            adapter_conditioning_scale = [adapter_conditioning_scale] * len(adapter_state)
            
        for k, (v, w) in enumerate(zip(adapter_state, adapter_conditioning_scale)):
            adapter_state[k] = v * w
        
        # Repeat Adapter State if num_images_per_prompt > 1
        if num_images_per_prompt > 1:
            for k, v in enumerate(adapter_state):
                adapter_state[k] = v.repeat(num_images_per_prompt, 1, 1, 1)

        # Repeat Adapter State if do_classifier_free_guidance
        if do_classifier_free_guidance:
            for k, v in enumerate(adapter_state):
                adapter_state[k] = torch.cat([v] * 2, dim=0)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=[state.clone() for state in adapter_state],
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionAdapterPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)