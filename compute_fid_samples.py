import json
import torch
import random
import numpy as np
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from hydra import compose, initialize
from hydra.utils import instantiate
from diffusers import T2IAdapter, DDPMScheduler
from pipelines.masked_adapter import StableDiffusionMaskedAdapterPipeline

from data.transforms import MaskRandomizer
from data.utils import image_grid, convert_to_image_range, MAP_CONDITIONS
from metrics.fid import FID, FIDDataset
from metric_helper_functions import load_configuration, load_pipeline, inference, load_controlnet, inference_controlnet

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--same_sample', action='store_true')
    parser.add_argument('--controlnet', action='store_true')
    return parser.parse_args()

def directory_is_empty(directory: str) -> bool:
    return not any(Path(directory).iterdir())

if __name__ == "__main__":
    # -- Get Arguments -- #
    arguments = get_arguments()

    # -- Load Configuration -- #
    ROOT = Path('.')
    _ = initialize(version_base='1.3', config_path=str(ROOT / 'config'))
    configuration = {
        'seed': 42, 
        'device': 'cuda',
        'img_size': 512,
        'model_type': 'adapter_canny',
        'dataset_root': '/home/ubuntu/datasets/laion_600k/',
        'base_diffusion_ckpt': '/home/ubuntu/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b'
    }
    print('Configuration loaded.')

    # -- Set Seeds -- #
    random.seed(configuration['seed'])
    torch.manual_seed(configuration['seed'])
    generator = torch.Generator(device=configuration['device']).manual_seed(configuration['seed'])

    # -- Create Paths -- #
    real_samples_path = Path('/home/ubuntu/outputs/real_fid_samples')
    save_real = True if directory_is_empty(real_samples_path) else False
    if save_real: real_samples_path.mkdir(exist_ok=True, parents=True)
    model_name = arguments.model_name if not arguments.controlnet else 'controlnet'
    model_path = Path(f'/home/ubuntu/outputs/{model_name}')
    samples_path = model_path / ('fid_samples' if not arguments.same_sample else 'fid_same_samples')
    samples_path.mkdir(exist_ok=True, parents=True)

    # -- Load Configuration -- #
    cond_cfg = load_configuration(
        configuration,
        output_path=model_path,
        checkpoint_name=arguments.checkpoint,
        mask_features='true'
    )

    # -- Generate Samples if needed -- #
    if directory_is_empty(samples_path):
        # -- Load Dataset -- #
        if not arguments.same_sample:
            items = [path.stem for path in Path(cond_cfg.dataset.get_items.data_root).iterdir()]
            [items.remove(val_item) for val_item in cond_cfg.validation_samples]
            _, val_items = torch.utils.data.random_split(
                items, (cond_cfg.train_partition, 1 - cond_cfg.train_partition),
                generator=torch.Generator(device='cpu').manual_seed(configuration['seed'])
            )
            val_idx_items = [val_items.dataset[val_items.indices[idx]] for idx in range(len(val_items.indices))]
            random.Random(configuration['seed']).shuffle(val_idx_items)
            samples = instantiate(cond_cfg.dataset.source, val_idx_items[:arguments.samples], return_as_dict=True)
        else:
            samples = instantiate(cond_cfg.dataset.source, cond_cfg.validation_samples, return_as_dict=True)
            samples = [samples[0]] * arguments.samples
        print('Dataset loaded.')

        # -- Load Pipeline -- #
        cond_pipeline = load_pipeline(cond_cfg) if not arguments.controlnet else load_controlnet(cond_cfg)

        # -- Constants -- #
        mask_randomizer = MaskRandomizer(method='fixed', mode='eval', bbox_min_area=10000)
        (tensor_mask, bbox), mask = mask_randomizer(
            configuration['img_size'], return_bbox=True, points=([256, 0],[512, 512]), device=configuration['device']
        )
        black_image = Image.new('L', size=(512, 512))

        # -- Generate Samples -- #
        seeds = random.Random(configuration['seed']).sample(range(0, arguments.samples),  arguments.samples)
        for idx, (sample, seed) in enumerate(tqdm(zip(samples, seeds), total=len(samples))):
            prompt, no_prompt = sample['caption'], ''
            mod_sample = np.array(sample['condition'])
            mod_sample[256:512, 256:512]=0
            sample['condition'] = Image.fromarray(mod_sample)
            masked_condition = Image.composite(sample['condition'], black_image, mask)
            filename = f'{model_name}_{sample["key"].stem}'
            if arguments.same_sample: filename = f'{model_name}_same_{str(idx).zfill(4)}'
            if not (samples_path / f'{filename}_wp.png').exists():
                image_wp, _ = inference(
                    sample, cond_pipeline, tensor_mask, use_mask=True if model_name != 'canny_baseline' else False, 
                    prompt=prompt, condition=masked_condition, seed=seed, device=configuration['device'],
                    generator=generator
                ) if not arguments.controlnet else inference_controlnet(
                    sample, cond_pipeline, tensor_mask, use_mask=False, prompt=prompt, condition=masked_condition,
                    seed=seed, device=configuration['device'], generator=generator
                )
                T.ToPILImage()(image_wp).save(samples_path / f'{filename}_wp.png')
            if not (samples_path / f'{filename}_wop.png').exists() and not arguments.same_sample:
                image_wop, _ = inference(
                    sample, cond_pipeline, tensor_mask, use_mask=True if model_name != 'canny_baseline' else False, 
                    prompt=no_prompt, condition=masked_condition, seed=seed, device=configuration['device'], 
                    generator=generator
                ) if not arguments.controlnet else inference_controlnet(
                    sample, cond_pipeline, tensor_mask, use_mask=False, prompt=no_prompt, condition=masked_condition,
                    seed=seed, device=configuration['device'], generator=generator
                )
                T.ToPILImage()(image_wop).save(samples_path / f'{filename}_wop.png')
            if save_real: sample['condition'].save(real_samples_path / f'{sample["key"].stem}.png')
        print('Samples generated.')