import torch
import cv2
import numpy as np

from torchmetrics import Metric

class DiffCC(Metric):
    def __init__(self, donwsample_factor=8, kernel_size=11, gaussian_std=5, **kwargs):
        super().__init__(**kwargs)
        self.down_factor = donwsample_factor
        self.k_size = kernel_size
        self.sigma = gaussian_std
        self.add_state("coincidence_percent", default=[], dist_reduce_fx="cat")
        self.add_state("allucination_in_percent", default=[], dist_reduce_fx="cat")
        self.add_state("allucination_out_percent", default=[], dist_reduce_fx="cat")

    def update(self, prediction, target, mask):
        down_size = target.size[0] // self.down_factor
        size = (int(down_size), int(down_size))
        # Resize Mask
        resized_mask = torch.from_numpy(cv2.resize(np.array(mask), size, interpolation=cv2.INTER_NEAREST)) / 255.0
        # Compute LF Target
        target_tensor = torch.from_numpy(
            cv2.resize(cv2.GaussianBlur(np.array(target), (self.k_size, self.k_size), self.sigma), size),
        ) / 255.0
        # Compute LF Prediction
        pred_tensor = torch.from_numpy(
            cv2.resize(cv2.GaussianBlur(np.array(prediction), (self.k_size, self.k_size), self.sigma), size),
        ) / 255.0
        # Compute CC
        total_ones_inside = torch.count_nonzero(target_tensor.view(-1)[resized_mask.view(-1) > 0])
        total_pixels_inside = torch.count_nonzero(resized_mask)
        total_pixels_outside = torch.count_nonzero(1-resized_mask)
        diff_in = (target_tensor - pred_tensor) * resized_mask
        pred_out = pred_tensor*(1-resized_mask)
    
        # Update
        self.coincidence_percent.append(1 - (diff_in[diff_in > 0].shape[0] / total_ones_inside))
        self.allucination_in_percent.append(diff_in[diff_in < 0].shape[0] / total_pixels_inside)
        self.allucination_out_percent.append(pred_out[pred_out > 0].shape[0] / total_pixels_outside)
        
    def compute(self):
        coincidence_percent_mean = torch.mean(torch.stack(self.coincidence_percent))
        allucination_percent_in_mean = torch.mean(torch.stack(self.allucination_in_percent))
        allucination_percent_out_mean = torch.mean(torch.stack(self.allucination_out_percent))
        return {
            'coincidence': coincidence_percent_mean,
            'allucinations': {'inside': allucination_percent_in_mean, 'outside': allucination_percent_out_mean}
        }
        
        