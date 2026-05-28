

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TIE/Wan event-caption video inference.")
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--start_line", type=int, default=0)
    parser.add_argument("--end_line", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--wan_model_path", type=str, default=None, help="Wan model root. When set, DiT defaults to this path and common files default to <wan_model_path>/.")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="DiT checkpoint path or safetensors shard list directory. Overrides --wan_model_path for DiT.")
    parser.add_argument("--text_encoder_path", type=str, default=None, help="T5 checkpoint path. Defaults to <wan_model_path>/models_t5_umt5-xxl-enc-bf16.pth.")
    parser.add_argument("--vae_path", type=str, default=None, help="Wan VAE checkpoint path. Defaults to <wan_model_path>/Wan2.2_VAE.pth.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="UMT5 tokenizer path. Defaults to <wan_model_path>/umt5-xxl.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=161)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--rope_type", type=str, default="sinc_form")
    parser.add_argument("--rope_alpha", type=float, default=1.0)
    parser.add_argument("--scaling_factor", type=float, default=4.0)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--image_source", type=str, default=None)
    parser.add_argument("--i2v", action="store_true")
    parser.add_argument("--enable_mapping", action="store_true")
    parser.add_argument("--base", action="store_true", help="Flatten event prompts into normal text captions.")
    parser.add_argument("--dry_run", action="store_true", help="Validate parsing and path resolution without loading model weights.")
    return parser


def torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def obj2prompt(x, noise_scale=0.0):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return tuple(x)

    start, end, prompt = [], [], []
    if "global_description" in x:
        start.append(0.0)
        end.append(10.0)
        prompt.append(x["global_description"])
        for entity in x.get("entities", []):
            for event in entity.get("events", []):
                start.append(float(event["start_time"]))
                end.append(float(event["end_time"]))
                prompt.append(event["description"])
    else:
        start.append(0.0)
        end.append(10.0)
        global_caption = x.get("global_caption", "")
        prompt.append(global_caption if isinstance(global_caption, str) else global_caption.get("long_caption", ""))
        for participant in x.get("participants", {}).values():
            start.append(float(participant["start_time"]) + random.gauss(0.0, 1.0) * noise_scale)
            end.append(float(participant["end_time"]) + random.gauss(0.0, 1.0) * noise_scale)
            prompt.append(participant["long_description"])
            for item in participant.get("timeline", []):
                start.append(float(item["start_time"]) + random.gauss(0.0, 1.0) * noise_scale)
                end.append(float(item["end_time"]) + random.gauss(0.0, 1.0) * noise_scale)
                prompt.append(item["long_caption"])
        for entity in x.get("entities", []):
            for event in entity.get("behaviors", []):
                start.append(float(event["start_time"]))
                end.append(float(event["end_time"]))
                prompt.append(event["description"])
    return (start, end, prompt)


def obj2prompt_norm(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return " ".join(str(item) for item in x[2])

    parts = []
    if "global_description" in x:
        parts.append(f'{x["global_description"]} in 0.0 s - 10.0 s')
        for entity in x.get("entities", []):
            for event in entity.get("events", []):
                parts.append(f'{event["description"]} in {event["start_time"]} s - {event["end_time"]} s')
    else:
        global_caption = x.get("global_caption", "")
        text = global_caption if isinstance(global_caption, str) else global_caption.get("long_caption", "")
        parts.append(f"{text} in 0.0 s - 10.0 s")
        for participant in x.get("participants", {}).values():
            parts.append(f'{participant["long_description"]} in {participant["start_time"]} s - {participant["end_time"]} s')
            for item in participant.get("timeline", []):
                parts.append(f'{item["long_caption"]} in {item["start_time"]} s - {item["end_time"]} s')
    return " ".join(parts)


def load_prompt_rows(path: str, start_line: int, end_line: int | None):
    rows = Path(path).read_text().splitlines()
    selected = rows[start_line:end_line]
    objs = [json.loads(row) for row in selected]
    return objs


def resize_with_ffmpeg(image_path: Path, height: int, width: int):
    from PIL import Image

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(tmp_fd)
    temp = Path(tmp_path)
    target_ratio = width / height
    scale_filter = f"scale='if(gt(a,{target_ratio}),-1,{width})':'if(gt(a,{target_ratio}),{height},-1)'"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(image_path), "-vf", f"{scale_filter},crop={width}:{height}", "-frames:v", "1", str(temp)]
    try:
        subprocess.run(cmd, check=True)
        with Image.open(temp) as img:
            return img.convert("RGB")
    finally:
        temp.unlink(missing_ok=True)


def load_image_tensor(image_path: Path, height: int, width: int, device: str):
    import numpy as np
    import torch

    image = resize_with_ffmpeg(image_path, height, width)
    array = np.array(image, dtype=np.float32)
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    tensor = tensor * (2.0 / 255.0) - 1.0
    video = tensor.permute(2, 0, 1).unsqueeze(0).bfloat16()
    return video.permute(1, 0, 2, 3).contiguous()


def resolve_model_paths(args):
    if args.wan_model_path is None and args.ckpt_dir is None:
        raise ValueError("Either --wan_model_path or --ckpt_dir is required.")

    wan_root = Path(args.wan_model_path) if args.wan_model_path else None
    args.ckpt_dir = args.ckpt_dir or str(wan_root)
    if wan_root is not None:
        args.text_encoder_path = args.text_encoder_path or str(wan_root / "models_t5_umt5-xxl-enc-bf16.pth")
        args.vae_path = args.vae_path or str(wan_root / "Wan2.2_VAE.pth")
        args.tokenizer_path = args.tokenizer_path or str(wan_root / "umt5-xxl")

    missing = [name for name in ("ckpt_dir", "text_encoder_path", "vae_path", "tokenizer_path") if getattr(args, name) is None]
    if missing:
        raise ValueError("Missing model path arguments: " + ", ".join(f"--{name}" for name in missing))


def validate_args(args):
    resolve_model_paths(args)
    for attr in ("prompt_file", "ckpt_dir", "text_encoder_path", "vae_path", "tokenizer_path"):
        path = Path(getattr(args, attr))
        if not path.exists():
            raise FileNotFoundError(f"--{attr} does not exist: {path}")
    if args.i2v and not args.image_source:
        raise ValueError("--image_source is required when --i2v is set")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    objs = load_prompt_rows(args.prompt_file, args.start_line, args.end_line)
    prompts = [obj2prompt(obj["prompt"], args.noise_scale) for obj in objs] if not args.base else [obj2prompt_norm(obj["prompt"]) for obj in objs]
    ids = [obj.get("video", f"sample_{i:05d}.mp4") for i, obj in enumerate(objs)]

    if args.dry_run:
        print(json.dumps({
            "num_prompts": len(prompts),
            "first_id": ids[0] if ids else None,
            "first_prompt_type": type(prompts[0]).__name__ if prompts else None,
            "first_event_count": len(prompts[0][2]) if prompts and isinstance(prompts[0], tuple) else None,
            "ckpt_dir": args.ckpt_dir,
            "text_encoder_path": args.text_encoder_path,
            "vae_path": args.vae_path,
            "tokenizer_path": args.tokenizer_path,
        }, ensure_ascii=False, indent=2))
        return 0

    import torch
    from utils import save_video
    from pipeline.wan_video_tie import WanVideoPipeline

    torch.manual_seed(random.randint(0, sys.maxsize))
    pipe = WanVideoPipeline.from_pretrained(
        dit_path=args.ckpt_dir,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        tokenizer_path=args.tokenizer_path,
        torch_dtype=torch_dtype(args.dtype),
        device=args.device,
        event_caption=True,
        event_video_length=41,
        use_special_token=False,
        enable_mapping=args.enable_mapping,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, prompt in zip(ids, prompts):
        image_tensor = None
        if args.i2v:
            image_path = Path(args.image_source) / f"{Path(name).stem}.png"
            image_tensor = load_image_tensor(image_path, args.height, args.width, args.device)
        for repeat_idx in range(args.repeat):
            video = pipe(
                prompt=prompt,
                negative_prompt="",
                seed=args.seed,
                tiled=True,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.sample_steps,
                rope_type=args.rope_type,
                rope_alpha=args.rope_alpha,
                scaling_factor=args.scaling_factor,
                theta=args.theta,
                scaling_weight=1.0,
                i2v=args.i2v,
                image_tensor=image_tensor,
            )
            suffix = "" if args.repeat == 1 else f"-{repeat_idx}"
            out_name = f"{Path(name).stem}{suffix}.mp4"
            save_video(video, save_dir / out_name, fps=16, quality=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
