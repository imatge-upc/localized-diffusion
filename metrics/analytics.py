import torch
from .lfmse import LFMSE
from torchmetrics.multimodal.clip_score import CLIPScore


class Analytics():
    def __init__(self, method='canny'):
        self.lfmse = LFMSE(method=method)
        self.clip = CLIPScore('openai/clip-vit-base-patch16')

    def reset(self):
        self.lfmse.reset()
        self.clip.reset()

    def update(self, prediction, target, mask, caption=None, pred_orig=None):
        self.lfmse.update(prediction, target, mask)
        if caption: self.clip.update(torch.from_numpy(pred_orig), caption)

    def compute(self):
        return {
            'LFMSE': self.lfmse.compute(),
            'CLIPScore': round(float(self.clip.compute().numpy()), 4)
        }