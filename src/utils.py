import os
import json
import random
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(data: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check_gpu() -> dict:
    info: dict = {}
    if not torch.cuda.is_available():
        warnings.warn("No CUDA GPU detected. Experiments will be extremely slow.")
        info["available"] = False
        return info

    info["available"] = True
    info["device_count"] = torch.cuda.device_count()
    info["devices"] = []

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        dev_info = {
            "index": i,
            "name": props.name,
            "vram_gb": round(vram_gb, 2),
        }
        if vram_gb < 16:
            warnings.warn(
                f"GPU {i} ({props.name}) has only {vram_gb:.1f} GB VRAM. "
                "At least 16 GB recommended for a 4-bit quantized 7B model."
            )
        info["devices"].append(dev_info)

    return info


def git_push(message: str, repo_dir=".") -> str:
    repo_dir = str(repo_dir)
    commands = [
        ["git", "-C", repo_dir, "add", "-A"],
        ["git", "-C", repo_dir, "commit", "-m", message],
        ["git", "-C", repo_dir, "push"],
    ]
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 and "nothing to commit" not in result.stdout:
            raise RuntimeError(
                f"Git command failed: {' '.join(cmd)}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

    hash_result = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return hash_result.stdout.strip()
