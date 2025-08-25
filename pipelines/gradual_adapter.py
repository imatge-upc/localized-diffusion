import torch
import logging

from typing import Union, List, Optional, Dict, Callable, Any
from PIL import Image

from diffusers import StableDiffusionAdapterPipeline, T2IAdapter
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_adapter import (
    StableDiffusionAdapterPipelineOutput, _preprocess_adapter_image
)

from models.adapter_fusion import MultiFuserAdapter, Fuser, T2IAdapterXL

class StableDiffusionGradualAdapterPipeline(StableDiffusionAdapterPipeline):
    def __init__(
        self, vae, text_encoder, tokenizer, unet,
        scheduler, safety_checker, feature_extractor,
        adapter: Union[
            T2IAdapter, T2IAdapterXL, List[T2IAdapter], List[T2IAdapterXL], MultiFuserAdapter
        ],
        fuser: Optional[Fuser] = None,
        adapter_cond_names: Optional[List[str]] = None,
        requires_safety_checker: bool = True,
    ):
        super(StableDiffusionAdapterPipeline, self).__init__()

        if safety_checker is None and requires_safety_checker:
            logging.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
        adapter = adapter if isinstance(adapter, MultiFuserAdapter) else MultiFuserAdapter(
            adapter, fuser=fuser, adapter_cond_names=adapter_cond_names
        )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            adapter=adapter,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.Tensor, Image.Image, List[Image.Image]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Union[str, List[str]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Union[torch._C.Generator, List[torch._C.Generator]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        adapter_weights: Union[float, List[float]] = 1.0,
        adapter_apply_percentage: float = 1.0
    ):
        """
        The only difference with it's super class are the following parameters:
        :param: adapter_weights, containing the weights for each adapter.
        :param: adapter_apply_percentage, specifying the percentage of the total number of steps 
                that the codition will be applied.
        """
        # 0. Default height and width to unet
        height, width = self._default_height_width(height, width, image)
        device = self._execution_device

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        if isinstance(image, (torch.Tensor, Image.Image)): image = [image]
        # 1.1 Check that length of images are equal to the number of adapters
        assert len(image) == len(self.adapter.adapters), (
            "Expecting the number of images to be equal to the number of adapters."
            f"Got {len(image)} != {len(self.adapter.adapters)}."
        )
        adapter_input = [
            _preprocess_adapter_image(img, height, width).to(device).to(self.adapter.dtype) for img in image
        ]

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
        adapter_state = self.adapter(adapter_input, adapter_weights)
            
        if num_images_per_prompt > 1:
            for k, v in enumerate(adapter_state):
                adapter_state[k] = v.repeat(num_images_per_prompt, 1, 1, 1)
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
                    down_block_additional_residuals=[
                        state.clone() for state in adapter_state
                    ] if i < num_inference_steps * adapter_apply_percentage else None
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