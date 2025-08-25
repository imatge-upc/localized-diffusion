import json
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF

from PIL import Image
from tqdm import tqdm

from pathlib import Path
from hydra import compose, initialize
from hydra.utils import instantiate
from diffusers import T2IAdapter, DDPMScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel 
from pipelines.masked_adapter import StableDiffusionMaskedAdapterPipeline
from pipelines.masked_controlnet import StableDiffusionMaskedControlNet

from data.transforms import MaskRandomizer
from data.utils import image_grid, convert_to_image_range, MAP_CONDITIONS

from metrics.analytics import Analytics

# -- Functions -- #
def load_configuration(configuration, output_path=None, checkpoint_name=None, mask_features=None, more_overrides=[]):
    pretrained_checkpoint_path = f'{output_path}/{checkpoint_name}'
    overrides = [
        f"seed={configuration['seed']}",
        f"dataset.get_items.data_root={configuration['dataset_root']}",
        f"model={configuration['model_type']}",
        f"model.mask_features={mask_features}",
        f"model.pretrained_ckpt_path={pretrained_checkpoint_path}",
        f"model.base_diffusion_ckpt_path={configuration['base_diffusion_ckpt']}",
        f"output_path={output_path}"
    ]
    if more_overrides:
        overrides += more_overrides
    cfg = compose(config_name='train_adapter.yaml', overrides=overrides)
    return cfg

def inference(sample, pipeline, random_mask, prompt=None, use_mask=True, condition=None, seed=None, device=None, generator=None):
    prompt = sample['caption'] if prompt is None else prompt
    condition = sample['condition'] if condition is None else condition
    seed_generator = generator if not seed else torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        generated = pipeline(
            prompt=prompt, image=condition,
            mask=random_mask.unsqueeze(0) if use_mask else None,
            num_inference_steps=30, generator=seed_generator, output_type='np'
        ).images[0]
    generated_image = convert_to_image_range(generated)
    generated_condition = MAP_CONDITIONS['canny'](generated_image)
    return generated_image, generated_condition

def inference_controlnet(sample, pipelines, random_mask, prompt=None, use_mask=True, condition=None, seed=None, device=None, generator=None):
    prompt = sample['caption'] if prompt else ''
    condition = sample['condition'] if condition is None else condition
    seed_generator = generator if not seed else torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        generated = pipelines(
            prompt=prompt, image=condition, num_inference_steps=30, generator=seed_generator, output_type='np'
        ).images[0]
    generated_image = convert_to_image_range(generated)
    generated_condition = MAP_CONDITIONS['canny'](generated_image)
    return generated_image, generated_condition

def colorize_inside_outside(condition, generated, mask, rows=1):
    blue_channel = np.zeros_like(np.array(condition))[:, :, None]
    final_cond = Image.fromarray(np.concatenate([
        np.where(255-np.array(mask)>0, np.array(condition), 0)[:, :, None],
        np.where(np.array(mask)>0, np.array(condition), 0)[:, :, None],
        blue_channel
    ], axis=-1))
    final_generated = [
        Image.fromarray(np.concatenate([
            np.where(255-np.array(mask)>0, np.array(gen_img), 0)[:, :, None],
            np.where(np.array(mask)>0, np.array(gen_img), 0)[:, :, None],
            blue_channel
        ], axis=-1)) for gen_img in generated
    ]
    return image_grid([final_cond]+final_generated, rows, int((len(final_generated)+1)/rows))

def load_pipeline(cfg):
    # Load Models
    adapter = T2IAdapter(**cfg.model.parameters)
    adapter.load_state_dict(torch.load(Path(cfg.model.pretrained_ckpt_path), map_location='cpu'))
    scheduler = DDPMScheduler.from_config(cfg.model.base_diffusion_ckpt_path, subfolder='scheduler')
    pipeline = StableDiffusionMaskedAdapterPipeline.from_pretrained(
        cfg.model.base_diffusion_ckpt_path, scheduler=scheduler, adapter=adapter, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False
    ).to(cfg.device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline

def load_controlnet(cfg, modified=False):
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    scheduler = DDPMScheduler.from_config(cfg.model.base_diffusion_ckpt_path, subfolder='scheduler')
    pipeline_class = StableDiffusionControlNetPipeline if not modified else StableDiffusionMaskedControlNet
    pipeline = pipeline_class.from_pretrained(
        cfg.model.base_diffusion_ckpt_path, scheduler=scheduler, controlnet=controlnet,
        safety_checker=None, torch_dtype=torch.float16
    ).to(cfg.device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline