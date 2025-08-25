import torch
import kornia as K
from functools import partial
import torch.nn.functional as F

class SimpleMSELoss():
    def __init__(self, weight, **kwargs):
        self.weight = weight

    def __call__(self, pred, target, **kwargs):
        loss = self.weight * F.mse_loss(pred.float(), target.float(), reduction='none').mean(axis=[0,2,3])
        return loss, {'simple': loss.mean().detach()}

class MaskedMSELoss():
    def __init__(self, weight, **kwargs):
        self.weight = weight

    def __call__(self, pred, target, mask=None, **kwargs):
        mask = F.interpolate(mask, size=pred.shape[-1])
        loss = self.weight * F.mse_loss(pred.float() * mask, target.float() * mask, reduction='none').mean(axis=[0,2,3])
        return loss, {'simple': loss.mean().detach()}
    
class PredX0MSELoss():
    def __init__(self, weight, method, method_parameters, scheduler, device=None, **kwargs):
        self.available_methods = {'constant': self.constant, 'step_unit': self.step_unit, 'time_unit': self.time_unit}
        self.loss_weight_function = self.available_methods[method]
        self.method_parameters = method_parameters[method] if method_parameters[method] is not None else {}
        self.simple_weight = weight.simple
        self.pred_x0_weight = weight.pred_x0
        self.sqrt_alphas = torch.sqrt(scheduler.alphas_cumprod).to(device)
        self.sqrt_betas = torch.sqrt(1 - scheduler.alphas_cumprod).to(device)

    @staticmethod
    def extract_into_tensor(a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def predict_origin(self, model_output, timesteps, sample, prediction_type):
        if prediction_type == "epsilon":
            sqrt_beta = self.extract_into_tensor(self.sqrt_betas, timesteps, sample.shape)
            sqrt_alpha = self.extract_into_tensor(self.sqrt_alphas, timesteps, sample.shape)
            pred_x_0 = (sample - sqrt_beta * model_output) / sqrt_alpha
        elif prediction_type == "v_prediction":
            pred_x_0 = sqrt_alpha[timesteps] * sample - sqrt_beta[timesteps] * model_output
        else:
            raise ValueError(f"Prediction type {prediction_type} currently not supported.")
        return pred_x_0

    def __call__(
        self, pred, target, mask=None, timesteps=None,
        latents=None, noisy_latents=None, prediction_type=None, **kwargs
    ):
        loss_pred_weights = self.loss_weight_function(timesteps, **self.method_parameters)
        pred_x0 = self.predict_origin(pred, timesteps, noisy_latents, prediction_type)
        mask = torch.nn.functional.interpolate(mask, size=pred_x0.shape[-1])
        simple_loss = self.simple_weight * F.mse_loss(pred.float(), target.float(), reduction='none').mean(axis=[1,2,3])
        pred_x0_loss = loss_pred_weights * F.mse_loss(pred_x0 * mask, latents * mask, reduction='none').mean(axis=[1,2,3])
        return simple_loss + pred_x0_loss, {'simple': simple_loss.mean().detach(), 'pred_x0': pred_x0_loss.mean().detach()}
    
    def constant(self, *args, **kwargs):
        return self.pred_x0_weight

    def step_unit(self, timesteps, lower_bound_t=None, upper_bound_t=None, **kwargs):
        lower_bound_t = lower_bound_t if lower_bound_t is not None else 0
        upper_bound_t = upper_bound_t if upper_bound_t is not None else timesteps.max().item()
        condition = lambda x: (x >= lower_bound_t) & (x <= upper_bound_t)
        return torch.tensor([
            self.pred_x0_weight if condition(t.item()) else 0.0 for t in timesteps
        ], device=timesteps.device)

    def time_unit(self, timesteps, **kwargs):
        return torch.tensor([(self.sqrt_betas[t.item()] ** 2) for t in timesteps], device=timesteps.device)
    
class PredX0EdgeMSELoss(PredX0MSELoss):
    filter_map = {
        'sobel': K.filters.sobel,
        'laplacian': partial(K.filters.laplacian, kernel_size=5)
    }
    def __init__(self, *args, device=None, filter_type='sobel', **kwargs):
        # Check method
        super().__init__(*args, device=device, **kwargs)
        if filter_type not in list(PredX0EdgeMSELoss.filter_map.keys()): raise ValueError(
            f'The available methods are {list(PredX0EdgeMSELoss.filter_map.keys())}. Got {filter_type}'
        )
        self.filter = PredX0EdgeMSELoss.filter_map[filter_type]

    def __call__(
        self, pred, target, mask=None, timesteps=None,
        latents=None, noisy_latents=None, prediction_type=None, **kwargs
    ):
        loss_pred_weights = self.loss_weight_function(timesteps, **self.method_parameters)
        pred_x0 = self.predict_origin(pred, timesteps, noisy_latents, prediction_type)
        filtered_latents = self.filter(latents)
        filtered_pred_x0 = self.filter(pred_x0)
        mask = torch.nn.functional.interpolate(mask, size=filtered_latents.shape[-1])
        simple_loss = self.simple_weight * F.mse_loss(pred.float(), target.float(), reduction='none').mean(axis=[1,2,3])
        pred_x0_loss = loss_pred_weights * F.mse_loss(filtered_pred_x0 * mask, filtered_latents * mask, reduction='none').mean(axis=[1,2,3])
        return simple_loss + pred_x0_loss, {'simple': simple_loss.mean().detach(), 'pred_x0': pred_x0_loss.mean().detach()}
    
    def time_unit(self, timesteps, **kwargs):
        return torch.tensor([(self.sqrt_betas[t.item()] ** 2) * self.pred_x0_weight for t in timesteps], device=timesteps.device)

