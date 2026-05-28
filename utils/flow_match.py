import math

import torch


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, shift=None, dynamic_shift_len=None, exponential_shift_mu=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = exponential_shift_mu if exponential_shift_mu is not None else self.exponential_shift_mu
            if mu is None and dynamic_shift_len is not None:
                mu = self.calculate_shift(dynamic_shift_len)
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(self, model_output, timestep, sample, to_final=False):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_next = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_next - sigma)

    def streaming_step(self, model_output, timestep, sample, to_final=False):
        batch, frames = timestep.shape
        device = sample.device
        dtype = sample.dtype
        timestep = timestep.cpu()
        diff = torch.abs(self.timesteps.view(1, 1, -1) - timestep.unsqueeze(-1))
        timestep_id = torch.argmin(diff, dim=-1)
        sigma = self.sigmas[timestep_id].to(device=device, dtype=dtype)
        final_value = 1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0
        sigma_next = torch.full_like(sigma, final_value)
        valid_mask = timestep_id + 1 < len(self.timesteps)
        if valid_mask.any():
            sigma_next[valid_mask] = self.sigmas[(timestep_id + 1)[valid_mask]].to(device=device, dtype=dtype)
        if to_final:
            sigma_next.fill_(final_value)
        sigma = sigma.view(batch, 1, frames, *([1] * (len(sample.shape) - 3)))
        sigma_next = sigma_next.view(batch, 1, frames, *([1] * (len(sample.shape) - 3)))
        return sample + model_output * (sigma_next - sigma)

    def calculate_shift(self, image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        return image_seq_len * slope + base_shift - slope * base_seq_len

