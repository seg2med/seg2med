"""DDPMScheduler with a configurable pred_original_sample clamp range (P030 contract fix).

MONAI generative 0.2.3's DDPMScheduler has only ``clip_sample: bool`` with a
hardcoded ``[-1, 1]`` clamp in ``step()``. ``RangedDDPMScheduler`` carries the same
schedule state but clamps ``pred_original_sample`` to ``clip_sample_range`` at every
DDPM step, so a ``[0.0, 1.0]`` target-scale model never admits a negative ``x_0``
prediction inside the sampling loop. ``step()`` is the upstream 0.2.3 implementation
verbatim with only the clamp bounds parameterized; ``wrap()`` reuses an existing
scheduler's tensors without copying.
"""
from __future__ import annotations

import torch
from generative.networks.schedulers import DDPMScheduler


class RangedDDPMScheduler(DDPMScheduler):
    clip_sample_range = (-1.0, 1.0)

    @classmethod
    def wrap(cls, scheduler, clip_range):
        """A RangedDDPMScheduler sharing `scheduler`'s schedule state (tensors not copied)."""
        lo, hi = (float(v) for v in clip_range)
        if not lo < hi:
            raise ValueError(f"clip_sample_range must be increasing, got {list(clip_range)}")
        new = cls.__new__(cls)
        new.__dict__.update(vars(scheduler))
        new.clip_sample_range = (lo, hi)
        return new

    def step(self, model_output, timestep, sample, generator=None):
        """Upstream generative 0.2.3 DDPMScheduler.step with the clamp bounds parameterized."""
        if not self.clip_sample or self.clip_sample_range == (-1.0, 1.0):
            return super().step(model_output, timestep, sample, generator=generator)
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        pred_original_sample = torch.clamp(pred_original_sample, *self.clip_sample_range)
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * self.betas[timestep]) / beta_prod_t
        current_sample_coeff = self.alphas[timestep] ** 0.5 * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        variance = 0
        if timestep > 0:
            noise = torch.randn(model_output.size(), dtype=model_output.dtype,
                                layout=model_output.layout, generator=generator).to(model_output.device)
            variance = (self._get_variance(timestep, predicted_variance=predicted_variance) ** 0.5) * noise
        return pred_prev_sample + variance, pred_original_sample


def clip_sample_range(scheduler):
    """The scheduler's pred_original_sample clamp range as a list; [-1, 1] for stock schedulers."""
    return list(getattr(scheduler, "clip_sample_range", (-1.0, 1.0)))
