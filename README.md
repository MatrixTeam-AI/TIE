# TIE Release

## Inference

Remove `--dry_run` to run model inference. The script supports the Pisces/TIE rope flags:

```bash
python scripts/tie_t2v.py \
  --prompt_file prompts.jsonl \
  --save_dir /path/to/save/folder \
  --ckpt_dir /path/to/dit/folder \
  --wan_model_path /path/to/wan_model \
  --sample_steps 50 \
  --height 704 --width 1280 --num_frames 161 \
```

For image-to-video, add `--i2v --image_source /path/to/first_frames`; each image should be named after the video stem, for example `abc.png` for `abc.mp4`.
