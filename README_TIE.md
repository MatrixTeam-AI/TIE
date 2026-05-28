# TIE Release

This directory is a self-contained inference-oriented TIE/Wan release. The DiffSynth source tree has been reduced to the runtime pieces needed by `scripts/tie_t2v.py`.

- `pipeline/wan_video_tie.py`
- `modules/wan_video_dit.py`
- `modules/prompter.py`
- `scripts/tie_t2v.py`

## Dry run

Dry run validates imports, argument paths, JSONL parsing, and event prompt conversion without loading model weights:
Model checkpoints and tokenizer are loaded directly from paths. Recommended usage is `--wan_model_path /path/to/wan_model`, where common files are resolved directly from `/path/to/wan_model/`: `models_t5_umt5-xxl-enc-bf16.pth`, `Wan2.2_VAE.pth`, and `umt5-xxl/`. Explicit `--ckpt_dir`, `--text_encoder_path`, `--vae_path`, and `--tokenizer_path` still override these defaults.


```bash
python scripts/tie_t2v.py \
  --prompt_file /path/to/prompts.jsonl \
  --save_dir /tmp/tie_out \
  --wan_model_path /path/to/wan_model \
  --dry_run
```

## Inference

Remove `--dry_run` to run model inference. The script supports the Pisces/TIE rope flags:

```bash
python scripts/tie_t2v.py \
  --prompt_file prompts.jsonl \
  --save_dir outputs \
  --wan_model_path /path/to/wan_model \
  --sample_steps 50 \
  --height 480 --width 832 --num_frames 161 \
  --rope_type original --rope_alpha 1.0 --scaling_factor 1.0 --theta 10000
```

For image-to-video, add `--i2v --image_source /path/to/first_frames`; each image should be named after the video stem, for example `abc.png` for `abc.mp4`.
