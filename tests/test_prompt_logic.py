import json
import subprocess
import sys
from pathlib import Path


def test_tie_t2v_dry_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(json.dumps({
        "video": "demo.mp4",
        "prompt": {
            "global_description": "A person walks across a room.",
            "entities": [{"events": [{"start_time": 0.0, "end_time": 1.0, "description": "The person starts walking."}]}],
        },
    }) + "\n")
    for name in ("dit.safetensors", "text.pth", "vae.pth"):
        (tmp_path / name).write_text("placeholder")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()

    cmd = [
        sys.executable, str(root / "scripts" / "tie_t2v.py"),
        "--prompt_file", str(prompt_file),
        "--save_dir", str(tmp_path / "out"),
        "--ckpt_dir", str(tmp_path / "dit.safetensors"),
        "--text_encoder_path", str(tmp_path / "text.pth"),
        "--vae_path", str(tmp_path / "vae.pth"),
        "--tokenizer_path", str(tokenizer),
        "--dry_run",
    ]
    result = subprocess.run(cmd, cwd=root, check=True, text=True, capture_output=True)
    data = json.loads(result.stdout)
    assert data["num_prompts"] == 1
    assert data["first_id"] == "demo.mp4"
    assert data["first_prompt_type"] == "tuple"
    assert data["first_event_count"] == 2
