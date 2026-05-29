# TIE Release

[![Project Page](https://img.shields.io/badge/project-page-5b5bd6)](https://matrixteam-ai.github.io/pages/TIE/)
[![Dataset](https://img.shields.io/badge/dataset-Hugging%20Face-f5c542)](https://huggingface.co/collections/MatrixTeam/omnievents)
[![arXiv](https://img.shields.io/badge/arXiv-2605.10543-b31b1b)](https://arxiv.org/abs/2605.10543)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

**TIE** (**Time Interval Embedding**) accepts a series of events with descriptions and start/end timestamp, and generates videos with multi-event respond and accurate time onsets/offsets. Our implementation is based on Wan2.2-TI2V-5B.

[Jump to Quick Start](#quick-start)

<center>
<img src="assets/teaser.png"/>
</center>

## Demo

### Comparison with Seedance 2.0

<p align="center">
  <video src="assets/timestamp_comparison_example1.mp4" controls width="86%"></video>
</p>

<p align="center">
  <video src="assets/timestamp_comparison_example3.mp4" controls width="86%"></video>
</p>

<p align="center">
  <video src="assets/timestamp_comparison_example5.mp4" controls width="86%"></video>
</p>

### Long Video

<p align="center">
  <video src="assets/long_robotic_task.mp4" controls width="86%"></video>
</p>

**Find more on our [project pages](https://matrixteam-ai.github.io/pages/TIE/)!**

## Release Status

The current release is in **Inference Only** status.

Roadmap:

- **2026.05.12**: Paper released.
- **2026.05.26**: Inference code and models released.
- **2026.05.28**: Dataset released.
- **Future**: Training code and more datasets.

## Quick Start

Create an environment and install the package:

```bash
conda create -n tie python=3.10 -y
conda activate tie
pip install -r requirements.txt
pip install -e .
```

Install a CUDA-enabled PyTorch build that matches your driver and GPU before running inference. Image-to-video preprocessing also requires `ffmpeg` on `PATH`.

`xformers` or `flashattn` backend is also suggested for inference efficiency.

Full run of videos with `161` frames can take about `60 GiB` of VRAM.

## Prepare Weights

The CLI can resolve common Wan assets from a single model root (just clone them from [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) and place them in one folder):

```text
/path/to/wan_model/
  models_t5_umt5-xxl-enc-bf16.pth
  umt5-xxl/
  Wan2.2_VAE.pth
  ... DiT checkpoint files ...
```
Get DiT checkpoint from [TIE](https://huggingface.co/MatrixTeam/TIE), there are three separated models trained on different dataset.

Use `--wan_model_path /path/to/wan_model` when the DiT checkpoint and common Wan assets are colocated. If the DiT checkpoint is stored separately, pass `--ckpt_dir` as well.

You can also provide every component explicitly:

```bash
--ckpt_dir /path/to/dit_or_shards \
--text_encoder_path /path/to/models_t5_umt5-xxl-enc-bf16.pth \
--vae_path /path/to/Wan2.2_VAE.pth \
--tokenizer_path /path/to/umt5-xxl
```

## Prompt File

`--prompt_file` should point to a JSONL file. Each line is one sample.

Event-caption format:

```json
{"video": "sample_0001.mp4", "prompt": {"global_description": "A cinematic combat scene.", "entities": [{"events": [{"start_time": 0.0, "end_time": 4.0, "description": "The warrior steps forward."}, {"start_time": 4.0, "end_time": 10.0, "description": "The opponent raises a weapon."}]}]}} 
# [Start Timestamps, End Timestamps, Descriptions]
```

Simple caption format:

```json
{"video": "sample_0002.mp4", "prompt": [[0.0, 0.0, 4.0], [10.0, 4.0, 10.0],["A cinematic combat scene.", "The warrior steps forward.", "The opponent raises a weapon."]]}
```


## Run Inference

Validate inputs without loading model weights:

```bash
python scripts/tie_t2v.py \
  --prompt_file prompts.jsonl \
  --save_dir outputs/t2v \
  --wan_model_path /path/to/wan_model \
  --dry_run
```

Run text-to-video inference:

```bash
python scripts/tie_t2v.py \
  --prompt_file prompts.jsonl \
  --save_dir outputs/t2v \
  --wan_model_path /path/to/wan_model \
  --ckpt_dir /path/to/dit_or_shards \
  --sample_steps 50 \
  --height 704 \
  --width 1280 \
  --num_frames 161
```

Run image-to-video inference from first frames:

```bash
python scripts/tie_t2v.py \
  --prompt_file prompts.jsonl \
  --save_dir outputs/i2v \
  --wan_model_path /path/to/wan_model \
  --ckpt_dir /path/to/dit_or_shards \
  --i2v \
  --image_source /path/to/first_frames \
  --sample_steps 50 \
  --height 480 \
  --width 832 \
  --num_frames 161
```

For I2V, each first frame must be named after the output video stem. For example, a prompt row with `"video": "abc.mp4"` expects `/path/to/first_frames/abc.png`.

Generated videos are written to `--save_dir`.

### Examples: 

#### t2v
```bash
python scripts/tie_t2v.py \
  --prompt_file examples/t2v/test.jsonl \
  --save_dir outputs/t2v \
  --wan_model_path /path/to/wan_model \
  --ckpt_dir /path/to/TIE-generalcase.safetensors \
  --sample_steps 50 \
  --height 704 \
  --width 1280 \
  --num_frames 161
```

#### i2v
```bash
python scripts/tie_t2v.py \
  --prompt_file examples/i2v/test.jsonl \
  --save_dir outputs/i2v \
  --wan_model_path /path/to/wan_model \
  --ckpt_dir /path/to/TIE-gaming.safetensors \
  --i2v \
  --image_source examples/i2v/first_frame \
  --sample_steps 50 \
  --height 480 \
  --width 832 \
  --num_frames 161
```

## Other Options

```text
--start_line / --end_line    Select a slice of the JSONL prompt file.
--repeat                    Generate multiple samples for each prompt row.
--seed                      Sampling seed passed to the pipeline.
--dtype                     bfloat16, float16, or float32.
--device                    Inference device, usually cuda.
--base                      Flatten event prompts into normal text captions.
--rope_type                 original, sinc_form, point, t_rope, and other implemented variants. 
--rope_alpha                RoPE frequency multiplier.
--scaling_factor            Temporal scaling used by physical RoPE.
--theta                     RoPE theta.
--enable_mapping            Enable the radius mapping used by sinc_form.
--noise_scale               Add Gaussian noise to timeline boundaries.
```

## Notes

- `height` and `width` are rounded up to the VAE-required multiples when needed.
- `num_frames` is rounded to match the pipeline temporal division rule.
- The default event video length is 41 latent frames inside the pipeline (161 frames, 10s at 16 fps).
## License

This repository is released under the [Apache License 2.0](LICENSE).
