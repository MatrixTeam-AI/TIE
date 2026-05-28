import math
from typing import Optional, Union

import numpy as np
import torch
from einops import rearrange, reduce
from PIL import Image
from tqdm import tqdm

from utils.flow_match import FlowMatchScheduler
from utils.model_manager import load_wan_dit, load_wan_text_encoder, load_wan_vae
from utils.prompter import WanEventPrompter
from modules.wan_video_dit import WanModel, sinusoidal_embedding_1d
from modules.wan_video_text_encoder import WanTextEncoder
from modules.wan_video_vae import WanVideoVAE


def wasserstein_l2_uniform_interval(start_time, end_time, weight=1.0):
    mid = (end_time + start_time) / 2
    length = end_time - start_time
    return mid, length / 3.4641 / weight


def gen_physical_rope_for(
    start,
    end,
    dim,
    theta=10000.0,
    alpha=1.0,
    rope_type="original",
    scaling_factor=1.0,
    scaling_weight=1.0,
    multi_scale_levels=(1.0, 0.5),
    apply_sinc=False,
    enable_mapping=False,
    sinc_l2_norm=False,
):
    hdim = dim // 2
    if rope_type == "original":
        l1, l2 = scaling_factor * start, scaling_factor * end
    elif rope_type == "l2_wasserstein":
        l1, l2 = wasserstein_l2_uniform_interval(start, end, scaling_weight)
        l1, l2 = scaling_factor * l1, scaling_factor * l2
    elif rope_type == "t_rope":
        if hdim % 2 != 0:
            raise ValueError("(dim // 2) must be even for t_rope.")
        half_hdim = hdim // 2
        freqs = alpha / (theta ** (torch.arange(0, half_hdim, 2).double() / half_hdim)).to(start.device)
        start_term = torch.einsum("bi,j->bij", start, freqs)
        end_term = torch.einsum("bi,j->bij", end, freqs)
        cos = torch.cat((start_term.cos(), start_term.cos(), end_term.cos(), end_term.cos()), dim=-1)
        sin = torch.cat((start_term.sin(), start_term.sin(), end_term.sin(), end_term.sin()), dim=-1)
        return cos[:, :, None, :], sin[:, :, None, :]
    elif rope_type == "multi_component":
        components = (start, end, (start + end) / 2, torch.log1p(torch.clamp(end - start, min=1e-6)))
        if hdim % len(components) != 0:
            raise ValueError(f"(dim // 2) must be divisible by {len(components)} for multi_component.")
        comp_dim = hdim // len(components)
        if comp_dim % 2 != 0:
            raise ValueError("Each multi_component sub-dimension must be even.")
        freqs = alpha / (theta ** (torch.arange(0, comp_dim, 2, dtype=torch.double, device=start.device) / comp_dim))
        cos_parts, sin_parts = [], []
        for value in components:
            term = torch.einsum("bi,j->bij", value, freqs.to(value.dtype))
            cos_parts.append(torch.cat((term.cos(), term.cos()), dim=-1))
            sin_parts.append(torch.cat((term.sin(), term.sin()), dim=-1))
        return torch.cat(cos_parts, dim=-1)[:, :, None, :], torch.cat(sin_parts, dim=-1)[:, :, None, :]
    elif rope_type == "multi_scale_wasserstein":
        if hdim % len(multi_scale_levels) != 0:
            raise ValueError("hdim must be divisible by the number of multi-scale levels.")
        scale_hdim = hdim // len(multi_scale_levels)
        freqs = alpha / (theta ** (torch.arange(0, scale_hdim, 2).double() / scale_hdim)).to(start.device)
        cos_parts, sin_parts = [], []
        center = (start + end) / 2
        for scale in multi_scale_levels:
            length = (end - start) * scale
            l1, l2 = wasserstein_l2_uniform_interval(center - length / 2, center + length / 2)
            freqs1 = torch.einsum("bi,j->bij", l1, freqs)
            freqs2 = torch.einsum("bi,j->bij", l2, freqs)
            cos_parts.append(torch.cat([freqs1.cos(), freqs2.cos()], dim=-1))
            sin_parts.append(torch.cat([freqs1.sin(), freqs2.sin()], dim=-1))
        return torch.cat(cos_parts, dim=-1)[:, :, None, :], torch.cat(sin_parts, dim=-1)[:, :, None, :]
    elif rope_type == "sinc_form":
        center = (start + end) * scaling_factor / 2.0
        if enable_mapping:
            map_list = [
                0.3, 0.34, 0.4, 0.47, 0.6, 0.7, 0.8, 1.0, 1.1, 1.2, 1.3, 1.4,
                1.5, 1.7, 1.74, 1.8, 1.9, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.9, 3.94,
                4.0, 4.1, 4.2, 4.3, 4.4, 4.6, 4.64, 4.8, 4.84, 5.0, 5.1, 5.2,
                5.24, 5.3, 5.5, 5.54, 5.6, 5.8, 5.9,
            ]
            map_tensor = torch.tensor(map_list, device=start.device, dtype=start.dtype)[None, None, :]
            radius = torch.argmin(torch.abs(((end - start) / 2.0)[:, :, None] - map_tensor), dim=-1).to(start.dtype) / 10.0
            radius = radius * scaling_factor
        else:
            radius = (end - start) * scaling_factor / 2.0
        freqs = alpha / (theta ** (torch.arange(0, dim, 2).double() / dim)).to(start.device)
        phase = torch.einsum("bi,j->bij", center, freqs)
        cos, sin = phase.cos(), phase.sin()
        if apply_sinc:
            phi = torch.einsum("bi,j->bij", radius, freqs)
            sincs = torch.sinc(alpha * phi / torch.pi)
            if sinc_l2_norm:
                sincs = sincs / torch.sqrt(torch.mean(sincs ** 2, dim=-1, keepdim=True))
            else:
                sincs = sincs / torch.mean(sincs, dim=-1, keepdim=True)
            cos, sin = cos * sincs, sin * sincs
        return cos[:, :, None, :], sin[:, :, None, :]
    elif rope_type == "point":
        mid = (start + end) * scaling_factor / 2.0
        freqs = alpha / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim)).to(start.device)
        phase = torch.einsum("bi,j->bij", mid, freqs)
        return phase.cos()[:, :, None, :], phase.sin()[:, :, None, :]
    else:
        raise NotImplementedError(f"rope_type {rope_type} is not implemented in the TIE release.")

    freqs = alpha / (theta ** (torch.arange(0, hdim, 2)[: (hdim // 2)].double() / hdim)).to(start.device)
    freqs1 = torch.einsum("bi,j->bij", l1, freqs)
    freqs2 = torch.einsum("bi,j->bij", l2, freqs)
    cos = torch.cat([freqs1.cos(), freqs2.cos()], dim=-1)
    sin = torch.cat([freqs1.sin(), freqs2.sin()], dim=-1)
    return cos[:, :, None, :], sin[:, :, None, :]


def fuse_start_end_timestamps(start, end, level=2, offset=1):
    length = start.shape[-1] - offset
    length = length - length % level
    tail_length = start.shape[-1] - length - offset
    start_fused = torch.cat([
        start[:, :offset],
        rearrange(rearrange(start[:, offset : offset + length], "b (s l) -> b s l", l=level)[:, :, 0:1].repeat(1, 1, level), "b s l -> b (s l)"),
        start[:, offset + length : offset + length + 1].repeat(1, tail_length),
    ], dim=-1)
    end_fused = torch.cat([
        end[:, :offset],
        rearrange(rearrange(end[:, offset : offset + length], "b (s l) -> b s l", l=level)[:, :, -1:].repeat(1, 1, level), "b s l -> b (s l)"),
        end[:, offset + length + tail_length - 1 : offset + length + tail_length].repeat(1, tail_length),
    ], dim=-1)
    return start_fused, end_fused


def model_fn_wan_video_tie(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    prompt_start_timestamps=None,
    prompt_end_timestamps=None,
    prompt_indices=None,
    video_start_timestamps=None,
    video_end_timestamps=None,
    len_context=None,
    rope_alpha=1.0,
    rope_type="original",
    scaling_factor=1.0,
    scaling_weight=1.0,
    theta=10000.0,
    multi_scale_levels=(1.0, 0.5),
    use_multiscale_time_interval=False,
    video_timestamp_levels=(1, 2, 4),
    enable_mapping=False,
    sinc_l2_norm=False,
    fourier=False,
):
    timestep = timestep.to(dtype=torch.bfloat16)
    latents = latents.to(dtype=torch.bfloat16)
    batch, _channels, frames, _height, _width = latents.shape
    if len(timestep.shape) < 2:
        timestep = timestep.expand(batch, frames)

    timestep = timestep.flatten(0)
    time_emb = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    time_mod = dit.time_projection(time_emb).view(batch, frames, 6, dit.dim)
    physical_rope_args = None
    physical_rope_args_lists = None

    if prompt_indices is not None:
        if use_multiscale_time_interval:
            img_cos_list, img_sin_list = [], []
            for level in video_timestamp_levels:
                start_fused, end_fused = fuse_start_end_timestamps(video_start_timestamps, video_end_timestamps, level=level)
                img_cos, img_sin = gen_physical_rope_for(
                    start_fused, end_fused, 128, alpha=rope_alpha, rope_type=rope_type,
                    scaling_factor=scaling_factor, scaling_weight=scaling_weight,
                    multi_scale_levels=multi_scale_levels, theta=theta,
                )
                img_cos_list.append(img_cos)
                img_sin_list.append(img_sin)
        else:
            img_cos, img_sin = gen_physical_rope_for(
                video_start_timestamps, video_end_timestamps, 128, alpha=rope_alpha,
                rope_type=rope_type, scaling_factor=scaling_factor,
                scaling_weight=scaling_weight, multi_scale_levels=multi_scale_levels, theta=theta,
            )
        text_cos, text_sin = gen_physical_rope_for(
            prompt_start_timestamps, prompt_end_timestamps, 128, alpha=rope_alpha,
            rope_type=rope_type, scaling_factor=scaling_factor, scaling_weight=scaling_weight,
            multi_scale_levels=multi_scale_levels, theta=theta, apply_sinc=True,
            enable_mapping=enable_mapping, sinc_l2_norm=sinc_l2_norm,
        )
        indices = prompt_indices[:, :len_context]
        text_cos = text_cos.gather(1, indices[:, :, None, None].expand(indices.shape[0], indices.shape[1], 1, text_cos.shape[-1]))
        text_sin = text_sin.gather(1, indices[:, :, None, None].expand(indices.shape[0], indices.shape[1], 1, text_sin.shape[-1]))
        text_cos = torch.cat([text_cos, torch.ones((text_cos.shape[0], context.shape[1] - len_context, 1, text_cos.shape[-1]), dtype=text_cos.dtype, device=text_cos.device)], dim=1)
        text_sin = torch.cat([text_sin, torch.zeros((text_sin.shape[0], context.shape[1] - len_context, 1, text_sin.shape[-1]), dtype=text_sin.dtype, device=text_sin.device)], dim=1)
        if use_multiscale_time_interval:
            physical_rope_args_lists = [(text_cos, text_sin, img_cos, img_sin) for img_cos, img_sin in zip(img_cos_list, img_sin_list)]
        else:
            physical_rope_args = (text_cos, text_sin, img_cos, img_sin)

    time_emb = time_emb.view(batch, frames, -1)
    context = dit.text_embedding(context)
    x = dit.patchify(latents)
    latent_frames, latent_height, latent_width = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    freqs = torch.cat([
        dit.freqs[0][:latent_frames].view(latent_frames, 1, 1, -1).expand(latent_frames, latent_height, latent_width, -1),
        dit.freqs[1][:latent_height].view(1, latent_height, 1, -1).expand(latent_frames, latent_height, latent_width, -1),
        dit.freqs[2][:latent_width].view(1, 1, latent_width, -1).expand(latent_frames, latent_height, latent_width, -1),
    ], dim=-1).reshape(latent_frames * latent_height * latent_width, 1, -1).to(x.device)

    for block_id, block in enumerate(dit.blocks):
        if use_multiscale_time_interval and physical_rope_args_lists is not None:
            physical_rope_args = physical_rope_args_lists[block_id % len(video_timestamp_levels)]
        x = block(x, context, time_mod, freqs, len_context, physical_rope_args, frames, fourier)

    x = dit.head(x, time_emb)
    return dit.unpatchify(x, (latent_frames, latent_height, latent_width))


class WanVideoPipeline(torch.nn.Module):
    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.bfloat16,
        tokenizer_path=None,
        event_video_length=41,
        use_special_token=False,
        split_timeline=False,
        fuse_global_timeline=False,
        enable_mapping=False,
        sinc_l2_norm=False,
        fourier=False,
    ):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.time_division_factor = 4
        self.time_division_remainder = 1
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanEventPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.event_video_length = event_video_length
        self.lps = 4
        self.use_special_token = use_special_token
        self.split_timeline = split_timeline
        self.fuse_global_timeline = fuse_global_timeline
        self.enable_mapping = enable_mapping
        self.sinc_l2_norm = sinc_l2_norm
        self.fourier = fourier

    def to(self, *args, **kwargs):
        device, dtype, _non_blocking, _convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.torch_dtype = dtype
        super().to(*args, **kwargs)
        return self

    def check_resize_height_width(self, height, width, num_frames=None):
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            print(f"height % {self.height_division_factor} != 0. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            print(f"width % {self.width_division_factor} != 0. We round it up to {width}.")
        if num_frames is None:
            return height, width
        if num_frames % self.time_division_factor != self.time_division_remainder:
            num_frames = (num_frames + self.time_division_factor - 1) // self.time_division_factor * self.time_division_factor + self.time_division_remainder
            print(f"num_frames % {self.time_division_factor} != {self.time_division_remainder}. We round it up to {num_frames}.")
        return height, width, num_frames

    def load_models_to_device(self, model_names=()):
        for name in model_names:
            model = getattr(self, name, None)
            if model is not None:
                model.to(self.device)

    def generate_noise(self, shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.float32, device=None, torch_dtype=None):
        generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
        return noise.to(dtype=torch_dtype or self.torch_dtype, device=device or self.device)

    def vae_output_to_image(self, vae_output, pattern="B C H W", min_value=-1, max_value=1):
        if pattern != "H W C":
            vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
        image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
        return Image.fromarray(image.to(device="cpu", dtype=torch.uint8).numpy())

    def vae_output_to_video(self, vae_output, pattern="B C T H W", min_value=-1, max_value=1):
        if pattern != "T H W C":
            vae_output = reduce(vae_output, f"{pattern} -> T H W C", reduction="mean")
        return [self.vae_output_to_image(image, pattern="H W C", min_value=min_value, max_value=max_value) for image in vae_output]

    @staticmethod
    def from_pretrained(
        dit_path: str,
        text_encoder_path: str,
        vae_path: str,
        tokenizer_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        event_caption=True,
        event_video_length=41,
        use_special_token=False,
        enable_mapping=False,
        fourier=False,
        split_timeline=False,
        fuse_global_timeline=False,
        sinc_l2_norm=False,
        **_unused,
    ):
        if not event_caption:
            raise ValueError("TIE release expects event_caption=True.")
        pipe = WanVideoPipeline(
            device=device,
            torch_dtype=torch_dtype,
            event_video_length=event_video_length,
            use_special_token=use_special_token,
            split_timeline=split_timeline,
            fuse_global_timeline=fuse_global_timeline,
            enable_mapping=enable_mapping,
            sinc_l2_norm=sinc_l2_norm,
            fourier=fourier,
        )
        pipe.dit = load_wan_dit(dit_path, torch_dtype=torch_dtype, device=device)
        pipe.text_encoder = load_wan_text_encoder(text_encoder_path, torch_dtype=torch_dtype, device=device)
        pipe.vae = load_wan_vae(vae_path, torch_dtype=torch_dtype, device=device)
        pipe.height_division_factor = pipe.vae.upsampling_factor * 2
        pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_path)
        return pipe

    def _encode_prompt(self, prompt, positive=True):
        self.load_models_to_device(["text_encoder"])
        if isinstance(prompt, tuple):
            if self.use_special_token:
                split = list(set(prompt[0] + prompt[1]))
                prompt = (prompt[0] + split, prompt[1] + split, prompt[2] + ["~"] * len(split))
            prompt_emb, indices, start, end, len_context = self.prompter.encode_event_prompt(
                prompt,
                positive=positive,
                device=self.device,
                split_timeline=self.split_timeline,
                fuse_global_timeline=self.fuse_global_timeline,
            )
            return {
                "context": prompt_emb,
                "prompt_indices": indices,
                "prompt_start_timestamps": start,
                "prompt_end_timestamps": end,
                "len_context": len_context,
            }
        return {"context": self.prompter.encode_prompt(prompt, positive=positive, device=self.device)}

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        rope_alpha=1.0,
        rope_type="original",
        scaling_factor=1.0,
        scaling_weight=1.0,
        theta=10000.0,
        multi_scale_levels=(1.0, 0.5),
        progress_bar_cmd=tqdm,
        i2v=False,
        image_tensor: Optional[torch.Tensor] = None,
        **_unused,
    ):
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        height, width, num_frames = self.check_resize_height_width(height, width, num_frames)
        pos_inputs = self._encode_prompt(prompt, positive=True)
        neg_inputs = self._encode_prompt(negative_prompt, positive=False)

        latent_frames = (num_frames - 1) // 4 + 1
        latent_shape = (1, self.vae.model.z_dim, latent_frames, height // self.vae.upsampling_factor, width // self.vae.upsampling_factor)
        latents = self.generate_noise(latent_shape, seed=seed, rand_device=rand_device)
        batch, _channels, frames, _latent_h, _latent_w = latents.shape
        video_start = (torch.linspace(0, frames - 1, frames, device=self.device)[None, :].repeat(batch, 1).float() - 0.75) / self.lps
        video_end = (torch.linspace(0, frames - 1, frames, device=self.device)[None, :].repeat(batch, 1).float() + 0.25) / self.lps
        video_start[:, 0] = 0.0

        first_frame_latents = None
        if i2v:
            if image_tensor is None:
                raise ValueError("image_tensor is required when i2v=True.")
            self.load_models_to_device(["vae"])
            video = image_tensor.to(device=self.device, dtype=self.torch_dtype)
            first_frame_latents = self.vae.encode([video], device=self.device, tiled=True, tile_size=tile_size, tile_stride=tile_stride).to(device=self.device, dtype=self.torch_dtype)
            self.load_models_to_device([])
            latents[:, :, 0:1] = first_frame_latents
            print("success encode first frame latent")

        self.load_models_to_device(["dit"])
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            if i2v:
                timestep = timestep.expand(batch, frames).clone()
                timestep[:, 0] = 0

            common = dict(
                dit=self.dit,
                latents=latents,
                timestep=timestep,
                video_start_timestamps=video_start,
                video_end_timestamps=video_end,
                rope_alpha=rope_alpha,
                rope_type=rope_type,
                scaling_factor=scaling_factor,
                scaling_weight=scaling_weight,
                theta=theta,
                multi_scale_levels=multi_scale_levels,
                enable_mapping=self.enable_mapping,
                sinc_l2_norm=self.sinc_l2_norm,
                fourier=self.fourier,
            )
            noise_pred_pos = model_fn_wan_video_tie(**common, **pos_inputs)
            if cfg_scale != 1.0:
                noise_pred_neg = model_fn_wan_video_tie(**common, **neg_inputs)
                noise_pred = noise_pred_neg + cfg_scale * (noise_pred_pos - noise_pred_neg)
            else:
                noise_pred = noise_pred_pos

            if i2v:
                latents_tail = self.scheduler.streaming_step(noise_pred[:, :, 1:], timestep[:, 1:], latents[:, :, 1:])
                latents = torch.cat([first_frame_latents, latents_tail], dim=2)
            else:
                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        self.load_models_to_device(["vae"])
        video = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video

