import cv2
import requests
import argparse
import json
import numpy as np
import torchvision.transforms as T

from io import BytesIO
from PIL import Image
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from functools import partial

def get_arguments():
    parser = argparse.ArgumentParser(description='Preprocess images from laion dataset')
    parser.add_argument('-f', '--function', type=str, required=True,
                        choices=['download', 'clean', 'assert'], help='Function to be used.')
    parser.add_argument('-c', '--compute_conditions', action='store_true', help='Compute the conditions for the dataset.')
    parser.add_argument('-d', '--arrow_dataset', type=str, help='Arrow File with the Dataset to be used.')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Output path for the downloaded images.')
    parser.add_argument('-imgs', '--image_size', type=int, default=512, help='The size of the images to be downloaded.')
    parser.add_argument('-cols', '--color_size', type=int, default=8, help='The size of the Color Image')
    parser.add_argument('-w', '--workers', type=int, default=8, help='Number of workers to download the images.')

    return parser.parse_args()

def apply_aspect_ratio(image: Image, max_img_size: int, divisible_by: int = 64):
    """
    Resizes an image to a max width or height while maintaining aspect ratio.
    The image is resized to the max_img_size parameter and then scaled to be divisible by the divisible_by parameter.
        Args:
            image: Image to resize
            max_img_size: Maximum width or height of the image
            divisible_by: The image is scaled to be divisible by this number
        Returns:
            Resized image
    """
    width, height = image.size
    if width >= height:
        # Resize width to max_img_size and scale height taking into account the divisible_by parameter
        scaled_w, scaled_h = max_img_size, round(int(height / (width / max_img_size)) / divisible_by) * divisible_by
    else:
        # Resize height to max_img_size and scale width taking into account the divisible_by parameter
        scaled_w, scaled_h = round(int(width / (height / max_img_size)) / divisible_by) * divisible_by, max_img_size
    return image.resize((scaled_w, scaled_h))

def check_image_size(image, image_size):
    W, H = image.size
    if min(H, W) < image_size:
        aspect_ratio = H / W if W < H else W / H
        image = image.resize(
            (image_size, int(image_size*aspect_ratio))
        ) if W < H else image.resize((int(image_size*aspect_ratio), image_size))
    return image

def download_images(url, caption, idx, compute_conditions=False, image_size=512, color_size=8, output_path=None, center_crop=None):
    identifier = str(idx).zfill(6)
    save_path = Path(f"{output_path}/{identifier}")
    if save_path.exists():
        return 0
    try:
        response = requests.get(url, timeout=5)
        image_from_bytes = Image.open(BytesIO(response.content)).convert('RGB')
        image = apply_aspect_ratio(image_from_bytes, image_size)
        image = check_image_size(image, image_size)
        image = center_crop(image)
        if compute_conditions:
            color = image.resize((color_size, color_size)).resize(
                (image_size, image_size), resample=Image.Resampling.NEAREST
            )
            canny = Image.fromarray(cv2.Canny(np.array(image), 100, 200))
            color.save(f"{output_path}/{identifier}/color.png")
            canny.save(f"{output_path}/{identifier}/canny.png")
        # Saving Results
        save_path.mkdir(parents=True, exist_ok=True)
        image.save(f"{output_path}/{identifier}/original.jpg")
        with open(f"{output_path}/{identifier}/info.json", "w") as file:
            json.dump({'caption': caption, 'url': url}, file)
        return 1
    except Exception:
        return 0
        
if __name__ == "__main__":
    arguments = get_arguments()

    # Load Dataset
    if arguments.arrow_dataset:
        dataset = load_dataset('arrow', data_files={'train': arguments.arrow_dataset})
    else:
        dataset = load_dataset("ChristophSchuhmann/improved_aesthetics_6.5plus")
    Path(arguments.output_path).mkdir(parents=True, exist_ok=True)
    center_crop = T.CenterCrop(arguments.image_size)

    if arguments.function == 'download':
        # Download Images
        new_download_images = partial(
            download_images,
            compute_conditions=arguments.compute_conditions,
            image_size=arguments.image_size,
            color_size=arguments.color_size,
            output_path=arguments.output_path,
            center_crop=center_crop
        )

        with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
                result = list(tqdm(
                    executor.map(
                            new_download_images,
                            dataset['train']['URL'],
                            dataset['train']['TEXT'],
                            list(range(len(dataset['train'])))
                        ),
                        total=len(dataset['train'])
                    )
                )

    elif arguments.function == 'clean':
        # Clean Images
        for image_path in tqdm(list(Path(arguments.output_path).iterdir())):
            original = Image.open(f"{image_path}/original.jpg")
            # Check original Image
            if original.mode != 'RGB':
                original = original.convert('RGB')
                original.save(f"{image_path}/original.jpg")
            if original.size != (arguments.image_size, arguments.image_size):
                original = check_image_size(original, arguments.image_size)
                original = center_crop(original)
                original.save(f"{image_path}/original.jpg")
            # Check Color Image
            if not (image_path / 'color.png').exists():
                color = original.resize((arguments.color_size, arguments.color_size)).resize(
                    (arguments.image_size, arguments.image_size), resample=Image.Resampling.NEAREST
                )
                color = color.convert('RGB')
                color.save(f"{image_path}/color.png")
            else:
                color = Image.open(f"{image_path}/color.png")
                if color.mode != 'RGB':
                    color = color.convert('RGB')
                    color.save(f"{image_path}/color.png")
            # Check Canny Image
            if not (image_path / 'canny.png').exists():
                canny = Image.fromarray(cv2.Canny(np.array(original), 100, 200))
                canny.save(f"{image_path}/canny.png")
            # Check Caption
            if not (image_path / 'caption.txt').exists():
                with open(f"{image_path}/caption.txt", "w") as file:
                    file.write(dataset['train']['TEXT'][int(image_path.stem)])
                    
    elif arguments.function == 'assert':
        # Assert Images
        errors = []
        for image_path in tqdm(list(Path(arguments.output_path).iterdir())):
            if not (image_path / 'original.jpg').exists(): errors.append(image_path.stem)
            if not (image_path / 'color.png').exists(): errors.append(image_path.stem)
            if not (image_path / 'canny.png').exists(): errors.append(image_path.stem)
            if not (image_path / 'caption.txt').exists(): errors.append(image_path.stem)
        if len(errors) > 0:
            raise ValueError(f"Found {len(errors)} errors: {errors}")
        
    else:
        raise ValueError(f"Function {arguments.function} not implemented.")
            