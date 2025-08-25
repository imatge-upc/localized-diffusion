import torch
import cv2
import numpy as np

from PIL import Image

from torchmetrics import Metric

class LFMSE(Metric):
    def __init__(self, method='canny', donwsample_factor=8, kernel_size=11, gaussian_std=5, metric_factor=100, **kwargs):
        super().__init__(**kwargs)
        self.down_factor = donwsample_factor
        self.k_size = kernel_size
        self.sigma = gaussian_std
        self.m_factor = metric_factor
        methods = {'canny': self.update_canny, 'sobel': self.update_sobel}
        self.function = methods[method]
        self.add_state("lfmse_in", default=[], dist_reduce_fx="cat")
        self.add_state("lfmse_out", default=[], dist_reduce_fx="cat")
        self.add_state("lfmse_tot", default=[], dist_reduce_fx="cat")
    
    def masked_mse(self, pred, target, mask):
        # Flatten mask
        flatten_mask = mask.flatten()
        # Compute MSE
        squared_error = (pred - target) ** 2
        return squared_error.flatten()[flatten_mask > 0].mean()

    def update_canny(self, prediction, target, mask):
        down_size = target.size[0] // self.down_factor
        size = (int(down_size), int(down_size))
        # Resize Mask
        resized_mask = cv2.resize(np.array(mask), size, interpolation=cv2.INTER_NEAREST) / 255.0
        # Compute LF Target
        target_lf = cv2.resize(cv2.GaussianBlur(np.array(target), (self.k_size, self.k_size), self.sigma), size) / 255.0
        # Compute LF Prediction
        pred_lf = cv2.resize(cv2.GaussianBlur(np.array(prediction), (self.k_size, self.k_size), self.sigma), size) / 255.0
        # Compute MSE
        in_mse = self.masked_mse(pred_lf, target_lf, resized_mask)
        modified_target = np.where(resized_mask>0, target_lf, 0) + np.where(1-resized_mask>0, 1.0, 0.0)
        out_mse = self.masked_mse(pred_lf, modified_target, 1-resized_mask)
        return in_mse, out_mse

    def update_sobel(self, prediction, target, mask):
        in_target = np.where(np.array(mask)>0, np.array(target), 0)
        modified_target = in_target/np.max(in_target) + np.where(1-np.array(mask)>0, 1.0, 0.0)
        in_mse = self.masked_mse(prediction/np.max(prediction), in_target/np.max(in_target), np.array(mask))
        out_mse = self.masked_mse(prediction/np.max(prediction), modified_target, 1-np.array(mask))
        return in_mse, out_mse
    
    def update(self, prediction, target, mask):
        in_mse, out_mse = self.function(prediction, target, mask)
        self.lfmse_in.append(in_mse)
        self.lfmse_out.append(out_mse)
        self.lfmse_tot.append(in_mse+out_mse)

    def compute(self):
        self.lfmse_tot = np.array(self.lfmse_tot)[~np.isnan(np.array(self.lfmse_tot))]
        self.lfmse_in = np.array(self.lfmse_in)[~np.isnan(np.array(self.lfmse_in))]
        self.lfmse_out = np.array(self.lfmse_out)[~np.isnan(np.array(self.lfmse_out))]
        std, mean = torch.std_mean(torch.from_numpy(self.lfmse_tot))
        std_in, mean_in = torch.std_mean(torch.from_numpy(self.lfmse_in))
        std_out, mean_out = torch.std_mean(torch.from_numpy(self.lfmse_out))
        return {
            'tot': f'{round(float(mean), 4)}±{round(float(std), 4)}',
            'in': f'{round(float(mean_in), 4)}±{round(float(std_in), 4)}',
            'out': f'{round(float(mean_out), 4)}±{round(float(std_out), 4)}'
        }