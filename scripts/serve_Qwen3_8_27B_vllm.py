"""Serve a local Qwen checkpoint for Dynamic Memory evaluation."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldmembench.paths import DEFAULT_WEIGHTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_WEIGHTS_DIR
                        / "metrics/dynamic_memory/Qwen3.8-27B")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="GPUs for tensor-parallel model serving")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    args = parser.parse_args()
    devices = args.gpus.split(",")
    if any(not device.strip().isdigit() for device in devices) or len(set(map(int, devices))) != len(devices):
        parser.error("--gpus requires unique comma-separated GPU indices")
    model = args.model_path.expanduser().resolve()
    if not model.is_dir() or not (model / "preprocessor_config.json").is_file():
        parser.error(f"local Qwen model with its image processor is required: {model}")
    os.environ.update(CUDA_VISIBLE_DEVICES=",".join(device.strip() for device in devices),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
               "--served-model-name", "Qwen3.8-27B", "--host", "127.0.0.1", "--port", str(args.port),
               "--tensor-parallel-size", str(len(devices)), "--dtype", "bfloat16",
               "--max-model-len", "65536", "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--max-num-seqs", "2", "--limit-mm-per-prompt", '{"image": 128}',
               "--generation-config", "vllm", "--trust-remote-code"]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
