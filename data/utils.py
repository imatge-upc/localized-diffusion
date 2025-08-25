import cv2
import torch
import numpy as np
import kornia as K
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from typing import List
from PIL import Image, ImageDraw

# -- Helper Functions -- #
def load_image(filename, mode=None):
    """Open and load a `PIL.Image` and convert to `mode`"""
    im = Image.open(filename)
    return im.convert(mode) if mode else im

def image_grid(display_images: List, num_rows: int, num_cols: int):
    """Display images using a grid of images."""
    assert len(display_images) == num_rows * num_cols
    w, h = display_images[0].size
    grid = Image.new('RGB', size=(num_cols * w, num_rows * h))
    [grid.paste(img, box=(idx % num_cols * w, idx // num_cols * h)) for idx, img in enumerate(display_images)]
    return grid

def get_color_image(image, color_size=8, image_size=512, **kwargs):
    return cv2.resize(
        cv2.resize(image, (color_size, color_size)),
        (image_size, image_size), interpolation=cv2.INTER_NEAREST
    )

def get_canny_image(image, **kwargs):
    return cv2.Canny(image, 100, 200)

def get_sobel_image(image, **kwargs):
    if isinstance(image, np.ndarray): image = Image.fromarray(image)
    tensor = T.Normalize(mean=0.5, std=0.5)(T.ToTensor()(image.convert('L')))
    return np.array(T.ToPILImage()(K.filters.sobel(tensor[None, :])[0]))

MAP_CONDITIONS = {
    'canny': get_canny_image,
    'color': get_color_image,
    'sobel': get_sobel_image
}

def convert_to_image_range(image, to_pil=False):
    return (np.array(image) * 255).astype(np.uint8)

# -- Main Functions -- #
def mask_condition(image, mask):
    return Image.composite(image, Image.new('L', size=image.size), mask)

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

def visualize_results(sample, generated_image, generated_condition, mask):
    conditions = colorize_inside_outside(sample['condition'], [generated_condition], mask, rows=1)
    images = image_grid([sample['original']] + [Image.fromarray(generated_image)], 1, 2)
    return image_grid([conditions, images], 2, 1)