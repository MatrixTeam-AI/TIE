import os

import torch

from . import init_weights_on_device, load_state_dict
from modules.wan_video_dit import WanModel
from modules.wan_video_text_encoder import WanTextEncoder
from modules.wan_video_vae import WanVideoVAE, WanVideoVAE38


def load_state_dict_from_path(path):
    if isinstance(path, list):
        state_dict = {}
        for item in path:
            state_dict.update(load_state_dict(item))
        return state_dict
    if os.path.isdir(path):
        state_dict = {}
        for file_name in sorted(os.listdir(path)):
            if file_name.endswith((".safetensors", ".bin", ".ckpt", ".pth", ".pt")):
                state_dict.update(load_state_dict(os.path.join(path, file_name)))
        if not state_dict:
            raise FileNotFoundError(f"No checkpoint files found in directory: {path}")
        return state_dict
    return load_state_dict(path)


def _convert_state_dict(model_class, state_dict, source):
    converter = model_class.state_dict_converter()
    if source == "diffusers":
        return converter.from_diffusers(state_dict)
    return converter.from_civitai(state_dict)


def _instantiate(model_class, converted, torch_dtype, device):
    if isinstance(converted, tuple):
        model_state_dict, extra_kwargs = converted
    else:
        model_state_dict, extra_kwargs = converted, {}
    model_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False) else torch_dtype
    with init_weights_on_device():
        model = model_class(**extra_kwargs)
    model = model.eval()
    model.load_state_dict(model_state_dict, assign=True)
    return model.to(dtype=model_dtype, device=device)


def load_model_explicit(path, model_classes, torch_dtype, device, sources=("civitai", "diffusers")):
    state_dict = load_state_dict_from_path(path)
    errors = []
    for model_class in model_classes:
        for source in sources:
            try:
                converted = _convert_state_dict(model_class, state_dict, source)
                return _instantiate(model_class, converted, torch_dtype, device)
            except Exception as exc:
                errors.append(f"{model_class.__name__}/{source}: {exc}")
    details = "\n".join(errors[-6:])
    raise RuntimeError(f"Failed to load {path} as {', '.join(cls.__name__ for cls in model_classes)}. Recent errors:\n{details}")


def load_wan_dit(path, torch_dtype=torch.bfloat16, device="cuda"):
    print(f"Loading Wan DiT from: {path}")
    return load_model_explicit(path, (WanModel,), torch_dtype, device)


def load_wan_text_encoder(path, torch_dtype=torch.bfloat16, device="cuda"):
    print(f"Loading Wan T5 text encoder from: {path}")
    return load_model_explicit(path, (WanTextEncoder,), torch_dtype, device, sources=("civitai",))


def load_wan_vae(path, torch_dtype=torch.bfloat16, device="cuda"):
    print(f"Loading Wan VAE from: {path}")
    return load_model_explicit(path, (WanVideoVAE, WanVideoVAE38), torch_dtype, device, sources=("civitai",))
