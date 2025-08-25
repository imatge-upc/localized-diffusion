import os
import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from scipy import linalg
from torchmetrics import Metric
from torch.nn.functional import adaptive_avg_pool2d

from .inception import InceptionV3

# -- Metric Class -- #
class FID(Metric):
    def __init__(self, dims=2048, batch_size=50, inception_device=None, **kwargs):
        super().__init__(**kwargs)
        self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dims]]).to(inception_device)
        self.num_workers = min(len(os.sched_getaffinity(0)), 8)
        self.batch_size = batch_size
        # -- State -- #
        self.add_state("real", default=[], dist_reduce_fx="cat")
        self.add_state("fake", default=[], dist_reduce_fx="cat")

    def update(self, sample_path, real=True):
        if real: self.real.append(sample_path)
        else: self.fake.append(sample_path)

    def compute(self):
        real_dataset = FIDDataset(self.real)
        fake_dataset = FIDDataset(self.fake)
        real_loader = torch.utils.data.DataLoader(
            real_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers
        )
        fake_loader = torch.utils.data.DataLoader(
            fake_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers
        )
        real_activations = []
        fake_activations = []
        self.model.eval()

        # -- Compute Real Activations -- #
        for real_batch in tqdm(real_loader):
            real_batch = real_batch.to(self.device)
            with torch.no_grad():
                real_pred = self.model(real_batch)[0]
            if real_pred.size(2) != 1 or real_pred.size(3) != 1:
                real_pred = adaptive_avg_pool2d(real_pred, output_size=(1, 1))
            real_activations.append(real_pred.squeeze(3).squeeze(2).cpu().numpy())
        # -- Compute Fake Activations -- #
        for fake_batch in tqdm(fake_loader):
            fake_batch = fake_batch.to(self.device)
            with torch.no_grad():
                fake_pred = self.model(fake_batch)[0]
            if fake_pred.size(2) != 1 or fake_pred.size(3) != 1:
                fake_pred = adaptive_avg_pool2d(fake_pred, output_size=(1, 1))
            fake_activations.append(fake_pred.squeeze(3).squeeze(2).cpu().numpy())
        # -- Compute FID -- #
        real_activations = np.concatenate(real_activations)
        fake_activations = np.concatenate(fake_activations)
        m_real, s_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
        m_fake, s_fake = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)
        return frechet_distance_computation(m_fake, s_fake, m_real, s_real)


# -- Helper Functions -- #
class FIDDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return T.ToTensor()(Image.open(self.image_paths[idx]).convert('RGB'))


def frechet_distance_computation(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean