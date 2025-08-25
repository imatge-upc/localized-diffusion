import sys
import cv2
import json

import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from metrics.analytics import Analytics
    from metrics.fid import FID
    from data.pair_dataset import Laion600KDataset
    from data.transforms import MaskRandomizer
    from data.utils import get_sobel_image

    # Dataset
    condition_name='sobel'
    dataset_root = '/home/ubuntu/datasets/laion_600k'
    outputs_root = '/home/ubuntu/outputs'
    items = [path.stem for path in Path(dataset_root).iterdir()]
    dataset = Laion600KDataset(data_root=dataset_root, condition_name=condition_name, mask_features=False, items=items)
    same_sample_target = dataset.get_item_by_key(Path(f'{dataset_root}/387721'))
    real_samples_paths = [str(path) for path in Path('/home/ubuntu/outputs/real_fid_samples').iterdir()]

    # Mask Randomizer
    mask_randomizer = MaskRandomizer(method='fixed', mode='eval', bbox_min_area=10000)
    (tensor_mask, bbox), mask = mask_randomizer(
        512, return_bbox=True, points=([128, 128],[384, 384]), device='cpu'
    )

    # Initialize Metrics
    analytics = Analytics(method=condition_name)
    fid_metric = FID(inception_device='cuda').to('cuda')

    # Models to evaluate
    models = [
        'canny_pred_x0_8e5_10e4_50K_all_steps_sobel',
    ]
    for model in models:
        # Reset
        analytics.reset()
        # Samples
        same_samples_path = Path(f'{outputs_root}/{model}/fid_same_samples')
        if not same_samples_path.exists(): same_samples = []
        else: same_samples = [] if model == 'controlnet' else [
            path for path in same_samples_path.iterdir()
        ]
        diff_samples_wp = [path for path in list(Path(f'{outputs_root}/{model}/fid_samples').glob('*_wp*'))]
        diff_samples_wop = [path for path in list(Path(f'{outputs_root}/{model}/fid_samples').glob('*_wop*'))]

        # Same Sample
        for sample in tqdm(same_samples, desc=f'{model.title()}: Same Samples'):
            pred_orig = np.array(Image.open(sample))
            pred = cv2.Canny(pred_orig, 100, 200) if condition_name == 'canny' else get_sobel_image(pred_orig)
            analytics.update(pred, same_sample_target['condition'], mask)
        results = analytics.compute()
        with open(f'{outputs_root}/{model}:same_sample.json', 'w') as file:
            json.dump(results, file)

        # Diff Sample
        for samples, name in [(diff_samples_wp, 'wp'), (diff_samples_wop, 'wop')]:
            analytics.reset()
            fid_metric.reset()
            for fake, real in tqdm(zip(samples, real_samples_paths), desc=f'{model.title()}: Diff Samples {name}'):
                key = str(fake).split('_')[-2]
                target = dataset.get_item_by_key(Path(f'{dataset_root}/{key}'))
                pred_orig = np.array(Image.open(fake))
                pred = cv2.Canny(pred_orig, 100, 200) if condition_name == 'canny' else get_sobel_image(pred_orig)
                analytics.update(
                    pred, target['condition'], mask,
                    caption=None, #target['caption'] if name != 'wop' else None,
                    pred_orig=None #pred_orig if name != 'wop' else None
                )
                fid_metric.update(real, real=True)
                fid_metric.update(fake, real=False)
            results = analytics.compute()
            results[f'FID_{name}'] = fid_metric.compute().item()
            with open(f'{outputs_root}/{model}:diff_sample:{name}.json', 'w') as file:
                json.dump(results, file)