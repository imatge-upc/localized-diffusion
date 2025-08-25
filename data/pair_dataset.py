import numpy as np
from PIL import Image

from stg_core.data.datasets import FolderDataset

from .utils import load_image

class PairDataset(FolderDataset):
    """Dataset for pairs of images"""

    def __init__(self, *args, return_as_dict=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_as_dict = return_as_dict

    @classmethod
    def get_dataset_items(cls, *args, **kwargs):
        """Get dataset pair folders instead of files."""
        kwargs['get_files'] = False
        return super().get_dataset_items(*args, **kwargs)

class Laion600KDataset(PairDataset):
    """Dataset of pairs from Laion 600K Dataset. Check https://laion.ai/blog/laion-aesthetics/"""
    def __init__(self, *args, condition_name=None, mask_features=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_name = condition_name
        self.mask_features = mask_features

    def get_item_by_key(self, key):
        original = load_image(key / 'original.jpg')
        condition = load_image(key / f'{self.condition_name}.png')
        with open(key / 'caption.txt', 'r') as f:
            caption = f.read().strip()
        # Return Data
        if self.return_as_dict:
            return {key: value for key, value in zip(
                ['original', 'condition', 'caption', 'key'], [original, condition, caption, key]
            ) if value is not None}
        return tuple(value for value in [original, condition, caption, key] if value is not None)

class SceneConditionDataset(PairDataset):
    """Dataset for pairs of images in order to generate background images"""
    def __init__(self, *args, mask_features=False, condition_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition_name = condition_name
        self.mask_features = mask_features

    def get_item_by_key(self, key):
        scene_img = load_image(key / 'scene.jpeg').resize((512, 512), Image.BILINEAR)
        condition = load_image(
            key / f'scene_{self.condition_name}{"" if not self.mask_features else "_masked"}.jpeg'
        )
        # Return Data
        if self.return_as_dict:
            return {key: value for key, value in zip(
                ['original', 'condition', 'key'], [scene_img, condition, key]
            ) if value is not None}
        return tuple(value for value in [scene_img, condition, key] if value is not None)