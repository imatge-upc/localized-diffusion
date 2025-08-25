import torch
import random
import numpy as np
import torchvision.transforms as T

def circle(height, width):
    radius = random.randint(50, 128)
    center = random.randint(radius, width - radius), random.randint(radius, height - radius)
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    return torch.from_numpy(dist_from_center <= radius).to(torch.float32)

def square(height, width):
    radius = random.randint(50, 128)
    center = random.randint(radius, width - radius), random.randint(radius, height - radius)
    Y, X = np.ogrid[:height, :width]
    mask = ((X >= center[0] - radius) & (X <= center[0] + radius) &
            (Y >= center[1] - radius) & (Y <= center[1] + radius))
    return torch.from_numpy(mask).to(torch.float32)

# -- Transforms -- #
class MaskRandomizer(object):
    available_modes = ['train', 'eval']
    available_methods = ['bbox', 'fixed', 'alternate']

    def __init__(self, method, mode='train', bbox_min_area=500):
        # Check Inputs
        assert method in MaskRandomizer.available_methods, (
            f'Invalid method. Got {method}. Available methods are {MaskRandomizer.available_methods}'
        )
        assert mode in MaskRandomizer.available_modes, (
            f'Invalid mode. Got {mode}. Available modes are {MaskRandomizer.available_modes}'
        )
        self.method = method
        self.mode = mode
        self.bbox_min_area = bbox_min_area
        self.tensor_to_pil = T.ToPILImage()
        self.randomizer = {'bbox': self.random_bbox, 'fixed': self.fixed_mask, 'alternate': self.alternate}

    def __call__(self, size, device='cpu', return_bbox=False, **kwargs):
        random_mask = torch.zeros((1, size, size), dtype=torch.float32)
        random_mask, bbox = self.randomizer[self.method](random_mask, **kwargs)
        random_mask = random_mask.to(device)
        if self.mode == 'train':
            return (random_mask, bbox) if return_bbox else random_mask
        else:
            mask = self.tensor_to_pil(random_mask)
            return ((random_mask, bbox), mask) if return_bbox else (random_mask, mask)

    def random_bbox(self, mask, **kwargs):
        _, height, width = mask.shape
        # Randomly determine the size of the bounding box until the minimum area is reached
        while True:
            x1, y1 = random.randint(0, width - 2), random.randint(0, height - 2)
            x2, y2 = random.randint(x1 + 1, width - 1), random.randint(y1 + 1, height - 1)
            if (x2 - x1) * (y2 - y1) >= self.bbox_min_area:
                break
        # Fill the bounding box area with ones
        mask[0, y1:y2, x1:x2] = 1.0
        return mask, (x1, y1, x2, y2)
    
    def fixed_mask(self, mask, points, **kwargs):
        x1, y1 = points[0]
        x2, y2 = points[1]
        # Fill the bounding box area with ones
        mask[0, y1:y2, x1:x2] = 1.0
        return mask, (x1, y1, x2, y2)
    
    def alternate(self, mask, **kwargs):
        _, height, width = mask.shape
        mode = random.choice(['circle', 'rectangle'])
        multiple = random.choice([True, False])
        for _ in range(2 if multiple else 1):
            mask += circle(height, width) if mode == 'circle' else square(height, width)
        return mask.clamp(0.0, 1.0), None     


class Laion600KTransforms(object):
    def __init__(self, work_size, mean, std):
        self.resize = T.Resize((work_size, work_size))
        self.resize_nearest = T.Resize((work_size, work_size), interpolation=T.InterpolationMode.NEAREST)
        self.to_tensor = T.ToTensor()
        self.mask_to_tensor = T.PILToTensor()
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        new_sample = {
            'original': self.normalize(self.to_tensor(self.resize(sample['original']))),
            'condition': self.to_tensor(self.resize(sample['condition'])),
            'caption': sample['caption'],
        }
        if (mask := sample.get('mask', None)) is not None:
            new_sample['mask'] = self.mask_to_tensor(self.resize_nearest(mask))
        return new_sample


class SceneConditionTransforms(object):
    def __init__(self, work_size):
        self.bilinear = T.Resize((work_size, work_size))
        self.nearest = T.Resize((work_size, work_size), interpolation=T.InterpolationMode.NEAREST)
        self.img_to_tensor = T.ToTensor()
        self.mask_to_tensor = T.PILToTensor()

    def __call__(self, sample):
        transformed_sample = {}
        for key, value in sample.items():
            if key == 'scene':
                transformed_sample[key] = self.img_to_tensor(self.bilinear(value))
            elif key == 'canny':
                transformed_sample[key] = self.img_to_tensor(self.bilinear(value))
            elif key == 'color':
                transformed_sample[key] = self.img_to_tensor(self.nearest(value))
            elif key == 'mask':
                transformed_sample[key] = self.mask_to_tensor(self.nearest(value))
        return transformed_sample

