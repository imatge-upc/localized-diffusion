import gc
import os
import logging
import math
import json
import wandb
import hydra
import random
import torch

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from collections import defaultdict
from hydra.utils import instantiate
from omegaconf import OmegaConf
from accelerate import Accelerator
from transformers import CLIPTextModel, AutoTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, T2IAdapter, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from pipelines.masked_adapter import StableDiffusionMaskedAdapterPipeline
from data.transforms import MaskRandomizer
from data.utils import (
    visualize_results, convert_to_image_range, mask_condition, MAP_CONDITIONS
)

def perform_validation(cfg, vae, unet, t2iadapter, accelerator, val_dataset):
    logging.info("Running validation... ")

    # Evaluation Transforms
    mask_randomizer = MaskRandomizer(method=cfg.model.mask_method, mode='eval', bbox_min_area=10000)

    # Pipeline
    pipeline = StableDiffusionMaskedAdapterPipeline.from_pretrained(
        cfg.model.base_diffusion_ckpt_path,
        vae=vae,
        unet=unet,
        adapter=accelerator.unwrap_model(t2iadapter),
        safety_checker=None,
        requires_safety_checker=False
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Validation Loop
    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)
    image_logs, prompt_logs = [], []
    with torch.no_grad():
        for sample in val_dataset:
            prompt_used = "" if random.random() < cfg.model.validation_empty_prompts else sample['caption']
            tensor_mask, mask = mask_randomizer(
                cfg.dataset.img_size, device=cfg.device
            ) if cfg.model.mask_features else (
                torch.ones((1, 512, 512), device=cfg.device), Image.new('L', (512, 512), 255)
            )
            condition = mask_condition(sample['condition'], mask) if cfg.model.input_masked else sample['condition']
            generated = pipeline(
                prompt=prompt_used, image=condition,
                mask=tensor_mask.unsqueeze(0) if cfg.model.mask_features else None,
                num_inference_steps=30, generator=generator, output_type='np'
            ).images[0]
            generated_image = convert_to_image_range(generated)
            generated_condition = MAP_CONDITIONS[cfg.model.condition](generated_image)
            image_logs.append(visualize_results(sample, generated_image, generated_condition, mask))
            prompt_logs.append("Empty" if prompt_used == "" else prompt_used)
    if hasattr(accelerator, "trackers"):
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                tracker.log({f"Validation": [
                    wandb.Image(
                        image_to_log, caption=f'{prompt_to_log}'
                    ) for prompt_to_log, image_to_log in zip(prompt_logs, image_logs)
                ]})
            else:
                logging.warn(f"image logging not implemented for {tracker.name}")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts):
    captions = ["" if random.random() < proportion_empty_prompts else caption for caption in prompt_batch]
    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        prompt_embeds = text_encoder(
            text_inputs.to(text_encoder.device),
            output_hidden_states=True,
        )[0].to(dtype=torch.float32)
    return prompt_embeds

def merge_batch(batch, use_masks=False, mask_randomizer=None, size=None):
    new_batch = {
        'original': torch.stack([x['original'] for x in batch]),
        'condition': torch.stack([x['condition'] for x in batch]),
        'caption': [x['caption'] for x in batch]
    }
    if use_masks: new_batch['mask'] = torch.stack([mask_randomizer(size) for _ in batch])
    return new_batch

@hydra.main(config_path="config", config_name="train_adapter", version_base='1.3')
def main(cfg):
    # Disabling tokenizer logging
    os.environ['TOKENIZERS_PARALLELISM'] = 'true' if cfg.num_workers > 0 else 'false'
    # Create Output Directory if does not exists and corresponing folders
    (output_path := Path(cfg.output_path)).mkdir(parents=True, exist_ok=True)

    # Setting seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Initialize Accelerator
    use_logger = True if "logger" in cfg else False
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=cfg.logger.name if use_logger else None
    )
    if use_logger:
        accelerator.init_trackers(
            project_name=cfg.logger.project_name, 
            config=OmegaConf.to_container(cfg.logger.config_to_push, resolve=True),
            init_kwargs={cfg.logger.name: cfg.logger.init_arguments}
        )
    logging.info('Accelerator Initialized!')
    global_step = 0

    # Load Dataset, Transforms, and Dataloaders
    items = [path.stem for path in Path(cfg.dataset.get_items.data_root).iterdir()]
    [items.remove(val_item) for val_item in cfg.validation_samples]
    logging.info(f'Total Dataset Size: {len(items)}')
    # Datasets
    train_items, _ = torch.utils.data.random_split(
        items, (cfg.train_partition, 1 - cfg.train_partition),
        generator=torch.Generator(device='cpu').manual_seed(cfg.seed)
    )
    tfms = instantiate(cfg.dataset.transforms)
    train_dataset = instantiate(cfg.dataset.source, train_items, transforms=tfms, return_as_dict=True)
    logging.info(f'Train Dataset Size: {len(train_dataset)}')
    if cfg.perform_validation:
        val_dataset = instantiate(cfg.dataset.source, cfg.validation_samples, return_as_dict=True)
        logging.info(f'Validation Dataset Size: {len(val_dataset)}')

    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        collate_fn=partial(
            merge_batch,
            use_masks=cfg.model.mask_features,
            mask_randomizer=MaskRandomizer(
                method=cfg.model.mask_method, mode='train', bbox_min_area=10000
            ) if cfg.model.mask_features else None,
            size=cfg.dataset.img_size
        )
    )

    # Load UNet, VAE, Text Encoder, Tokenizer and Scheduler
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_diffusion_ckpt_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.model.base_diffusion_ckpt_path, subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained(cfg.model.base_diffusion_ckpt_path, subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(cfg.model.base_diffusion_ckpt_path, subfolder="unet")
    scheduler = DDPMScheduler.from_config(cfg.model.base_diffusion_ckpt_path, subfolder='scheduler')
    scheduler_total_timesteps = scheduler.config.num_train_timesteps
    logging.info(f'Loaded Models from: {cfg.model.base_diffusion_ckpt_path}')

    # Instantiate Losses
    loss_function = instantiate(cfg.loss.function, scheduler=scheduler, device=cfg.device)

    # Load T2IAdapter
    if cfg.model.pretrained_ckpt_path:
        logging.info(f'Loading Adapter from: {cfg.model.pretrained_ckpt_path}')
        t2iadapter = T2IAdapter.from_pretrained(cfg.model.pretrained_ckpt_path)
    else:
        t2iadapter = T2IAdapter(**cfg.model.parameters)

    # Deactivate Gradients (Freeze Models)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if not cfg.model.train_unet:
        unet.requires_grad_(False)

    # Load Optimizer
    learning_rate = cfg.learning_rate if not cfg.scale_lr else (
        cfg.learning_rate * cfg.batch_size * cfg.gradient_accumulation_steps * accelerator.num_processes
    )
    optimizer = torch.optim.AdamW(
        t2iadapter.parameters(),
        lr=learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        eps=cfg.epsilon
    )
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    num_training_steps = cfg.num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Prepare everything with our `accelerator`.
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=torch.float32)

    t2iadapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        t2iadapter, optimizer, train_dataloader, lr_scheduler
    )

    # Setup weights for the adapter
    if isinstance(cfg.model.adapter_weights, float):
        adapter_weights = [adapter_weights] * 4 if cfg.model.adapter_weights > 1.0 else None
    else:
        adapter_weights = cfg.model.adapter_weights

    # Training Loop
    stop = False
    if cfg.batch_size <= 8: logging.warning("Batch size must be at most 8 to avoid OOM")
    if cfg.track_timesteps: timesteps_tracker = defaultdict(list)
    for epoch in range(cfg.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f'Epoch {epoch}')
        for batch in train_dataloader:
            with accelerator.accumulate(t2iadapter):
                latents = vae.encode(batch['original']).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                current_batch_size = latents.shape[0]

                # Encode Prompt
                prompt_embeds = encode_prompt(
                    batch['caption'], text_encoder, tokenizer, cfg.model.proportion_empty_prompts
                )

                # For more details about why cubic sampling is used, refer to section 3.4 of https://arxiv.org/abs/2302.08453
                timesteps = torch.rand((current_batch_size, ), device=cfg.device)
                if cfg.model.apply_cubic_timestep:
                    timesteps = (1 - timesteps ** 3) * scheduler_total_timesteps
                    timesteps = timesteps.long().clamp(0, scheduler_total_timesteps - 1)
                else:
                    timesteps = (timesteps * scheduler_total_timesteps).long().clamp(0, scheduler_total_timesteps - 1)

                # Sample Noise
                noise = torch.randn_like(latents).to(device=cfg.device)

                # (Forward Diffusion) Add Noise to Latents
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                # Adapter conditioning
                down_block_additional_residuals = t2iadapter(
                    batch['condition'] * batch['mask'] if cfg.model.input_masked else batch['condition']
                )
                if cfg.track_adapter_statistics:
                    mean_std = [(
                        mean.detach().cpu().numpy(), std.detach().cpu().numpy()
                    ) for feature in down_block_additional_residuals for std, mean in [torch.std_mean(feature)]]

                # Mask Adapter Features
                if cfg.model.mask_features:
                    down_block_additional_residuals = [
                        state * torch.nn.functional.interpolate(
                            batch['mask'], size=(batch['mask'].shape[-1] // size)
                        ) for state, size in zip(down_block_additional_residuals, [2**i for i in range(3,7)])
                    ]
                if adapter_weights is not None:
                    for k, (v, w) in enumerate(zip(down_block_additional_residuals, adapter_weights)):
                        down_block_additional_residuals[k] = v * w

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_block_additional_residuals,
                ).sample

                # Get the target for loss depending on the prediction type
                if cfg.model.prediction_type == 'epsilon':
                    target = noise
                elif cfg.model.prediction_type == 'v_prediction':
                    target = scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {cfg.model.prediction_type}")

                # Loss
                loss, loss_terms = loss_function(
                    model_pred, target, mask=batch['mask'], timesteps=timesteps,
                    latents=latents, noisy_latents=noisy_latents, prediction_type=cfg.model.prediction_type
                )

                # Tracking
                if cfg.track_timesteps: 
                    timesteps_tracker['timesteps'] += [timestep.item() for timestep in timesteps]
                    for loss_name, loss_term in loss_terms.items():
                        timesteps_tracker[loss_name] += [loss_term.item()]

                # Backward Pass
                loss = loss.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(t2iadapter.parameters(), cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update Progress Bar
            progress_bar.update(1)
            # Log to progress bar
            logs = {f'{k}_loss': v.item() for k, v in loss_terms.items()}
            logs.update({'loss': loss.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]})
            progress_bar.set_postfix(logs)
            # Log tracking statistics
            if cfg.track_adapter_statistics:
                logs.update({'Adapter_Statistics': {
                    f'Channel {k}': {"mean": mean, "std": std} for k, (mean, std) in enumerate(mean_std)
                }})
            # Log to trackers
            if accelerator.log_with:
                accelerator.log(logs, step=global_step + 1)

            # Perform Validation & Save Checkpoints
            if accelerator.is_local_main_process and cfg.perform_step_valcheck:
                if (
                    ((global_step + 1) % cfg.val_step_interval == 0) and 
                    ((global_step + 1) % cfg.gradient_accumulation_steps == 0) and cfg.perform_validation
                ):
                    # Perform Validation only if the step is a multiple of the gradient accumulation steps
                    perform_validation(cfg, vae, unet, t2iadapter, accelerator, val_dataset)
                if ((global_step + 1) % cfg.save_step_interval == 0):
                    accelerator.save(
                        t2iadapter.state_dict(), f'{output_path}/adapter_step_{global_step + 1}.pt'
                    )
            global_step += 1
            if ((global_step + 1) >= cfg.stop_after_step and (global_step + 1) % cfg.gradient_accumulation_steps == 0):
                stop = True
                break
        if accelerator.is_local_main_process:
            if (epoch + 1) % cfg.val_epoch_interval == 0 and cfg.perform_validation:
                perform_validation(cfg, vae, unet, t2iadapter, accelerator, val_dataset)
            if (epoch + 1) % cfg.save_epoch_interval == 0 and not stop:
                accelerator.save(t2iadapter.state_dict(), f'{output_path}/adapter_epoch_{epoch + 1}.pt')
            if stop:
                accelerator.save(t2iadapter.state_dict(), f'{output_path}/adapter_step_{global_step + 1}.pt')
                if cfg.track_timesteps:
                    with open(f'{output_path}/timesteps_tracker.json', 'w') as f:
                        json.dump(timesteps_tracker, f)
                break
    logging.info('Training Finished!')
    logging.info(f'Outputs can be found at: {output_path}')

if __name__ == '__main__':
    main()