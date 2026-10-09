"""Download upstream model packages, or verify local files with --verify.

For the consolidated release: hf download rescuerz/WorldMemBench-weights --local-dir ./weights
Use --verify afterwards to check local files and create metric links without downloading.

buffalo_l is fully extracted. SAM2, DINOv2, Grounding DINO, DA3 and SegFormer retain the
safetensors weights and accompanying repository files (including tokenizers),
without duplicate .pt/.bin weights. Metrics share existing model files.
Environment includes the DA3-Streaming and SALAD weights for geometry preparation.
DA3, ViPE Hugging Face packages and the opt-in Dynamic Qwen repository use hf
download. ViPE's two Google Drive checkpoints use gdown; other files use curl.
Existing files are kept by default; rerun the same command to fill missing files.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldmembench.weights import (
    BUFFALO_L, DA3_WEIGHTS, DEFAULT_WEIGHTS_DIR, WEIGHTS, HF_REPOSITORIES,
    download_hf_weight, download_weight, download_vlm_weights, download_vipe_weights,
    extract_buffalo_l, link_metric_weights, verify_weights,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--metrics", nargs="+", choices=[*WEIGHTS, *HF_REPOSITORIES],
                        help="Metrics to select (download default: non-Dynamic; --verify default: all)")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify local SHA256 values and create metric links; never download")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep existing files without rechecking them (default); --no-skip-existing rejects existing files")
    args = parser.parse_args(argv)
    args.metrics = args.metrics or ([*WEIGHTS, *HF_REPOSITORIES] if args.verify else list(WEIGHTS))
    root = args.weights_dir.expanduser().resolve()
    try:
        if args.verify:
            count = verify_weights(root, args.metrics)
            link_metric_weights(root, args.metrics)
            print(f"Verified {count} files. Metric links are ready in {root / 'metrics'}.", flush=True)
            return 0
        selected = dict.fromkeys(weight for metric in args.metrics for weight in WEIGHTS.get(metric, ()))
        for weight in selected:
            if args.skip_existing and (root / weight.path).is_file():
                print(f"Keeping existing file (not reverified): {root / weight.path}", flush=True)
            elif weight in DA3_WEIGHTS:
                download_hf_weight(weight, root)
            else:
                download_weight(weight, root)
            if weight == BUFFALO_L:
                extract_buffalo_l(root, skip_existing=args.skip_existing)
        if "dynamic_memory" in args.metrics:
            download_vlm_weights(root, skip_existing=args.skip_existing)
        if "camera_translation" in args.metrics:
            download_vipe_weights(root, skip_existing=args.skip_existing)
        link_metric_weights(root, args.metrics)
    except (OSError, ValueError) as error:
        action = "Weight verification" if args.verify else "Download"
        print(f"{action} failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
