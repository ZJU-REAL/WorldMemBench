"""Prepare DA3 poses and scene geometry for one video.mp4 or a folder of cases."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def streaming_config(weights: Path) -> dict:
    """Frozen reference settings; Python Sim3 is the reference runtime's solver."""
    model = weights / "DA3NESTED-GIANT-LARGE-1.1"
    return {
        "Weights": {"DA3": str(model / "model.safetensors"),
                    "DA3_CONFIG": str(model / "config.json"), "SALAD": str(weights / "dino_salad.ckpt")},
        "Model": {
            "chunk_size": 120, "overlap": 60, "loop_chunk_size": 20,
            "loop_enable": True, "useDBoW": False, "delete_temp_files": True,
            "align_lib": "torch", "align_method": "sim3", "scale_compute_method": "auto",
            "align_type": "dense", "ref_view_strategy": "middle", "ref_view_strategy_loop": "saddle_balanced",
            "depth_threshold": 15.0, "save_depth_conf_result": True, "save_debug_info": False,
            "Sparse_Align": {"keypoint_select": "orb", "keypoint_num": 5000},
            # Upstream evaluates the reference YAML's scientific-notation strings.
            "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"},
            "Pointcloud_Save": {"sample_ratio": 0.015, "conf_threshold_coef": 0.75},
        },
        "Loop": {
            "SALAD": {"image_size": [336, 336], "batch_size": 32, "similarity_threshold": 0.85,
                      "top_k": 5, "use_nms": True, "nms_threshold": 25},
            "SIM3_Optimizer": {"lang_version": "python", "max_iterations": 30, "lambda_init": "1e-6"},
        },
    }


def extract_frames(video: Path, output: Path) -> int:
    import cv2
    from worldmembench.errors import EvaluationError

    output.mkdir()
    capture = cv2.VideoCapture(str(video))
    try:
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or expected <= 0:
            raise EvaluationError(f"cannot decode video: {video}")
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not cv2.imwrite(str(output / f"frame_{count:06d}.png"), frame):
                raise EvaluationError(f"cannot write extracted frame {count}")
            count += 1
        if count != expected:
            raise EvaluationError(f"decoded {count} frames but video reports {expected}")
        return count
    finally:
        capture.release()


def convert_outputs(native: Path, output: Path, total_frames: int) -> None:
    """Preserve full-frame poses and stride-four float16 geometry, including the final frame."""
    import numpy as np
    from worldmembench.errors import EvaluationError

    c2w = np.loadtxt(native / "camera_poses.txt", dtype=np.float64).reshape(-1, 4, 4)
    if len(c2w) != total_frames or not np.isfinite(c2w).all():
        raise EvaluationError("DA3 camera poses must be finite and cover the entire video")
    if not np.allclose(c2w[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise EvaluationError("DA3 camera poses have invalid homogeneous rows")
    frame_dir = native / "results_output"
    files = sorted(frame_dir.glob("frame_*.npz"), key=lambda path: int(path.stem.split("_")[-1]))
    if [path.name for path in files] != [f"frame_{index}.npz" for index in range(total_frames)]:
        raise EvaluationError("DA3 must export depth/confidence for every video frame")
    selected = np.arange(total_frames, dtype=np.int64)[::4]
    if selected[-1] != total_frames - 1:
        selected = np.concatenate([selected, np.asarray([total_frames - 1], dtype=np.int64)])
    depths, confidence, intrinsics = [], [], []
    for index in selected:
        with np.load(files[int(index)], allow_pickle=False) as frame:
            depth, conf = np.asarray(frame["depth"], dtype=np.float32), np.asarray(frame["conf"], dtype=np.float32)
            intrinsic = np.asarray(frame["intrinsics"], dtype=np.float32)
        if depth.ndim != 2 or conf.shape != depth.shape or intrinsic.shape != (3, 3):
            raise EvaluationError("invalid DA3 depth/confidence/intrinsics shapes")
        if (not all(np.isfinite(array).all() for array in (depth, conf, intrinsic))
                or np.any(depth < 0) or not np.any(depth > 0) or np.any(conf < 0)):
            raise EvaluationError("invalid DA3 depth/confidence values")
        depths.append(depth.astype(np.float16))
        confidence.append(conf.astype(np.float16))
        intrinsics.append(intrinsic)
    output.mkdir()
    np.savez(output / "results.npz", extrinsics=np.linalg.inv(c2w)[:, :3],
             frame_indices=np.arange(total_frames, dtype=np.int64))
    np.savez_compressed(output / "scene_geometry.npz", schema_version=np.asarray([1], dtype=np.int64),
                        source_frame_count=np.asarray([total_frames], dtype=np.int64), frame_indices=selected,
                        depths=np.stack(depths), confidence=np.stack(confidence), intrinsics=np.stack(intrinsics),
                        c2w=c2w[selected], source=np.asarray(["da3_streaming_depth_conf_v1"]),
                        sample_stride=np.asarray([4], dtype=np.int64))


def ensure_artifacts(sample, context, *, weight_metric=None, require_scene=True) -> None:
    """Reuse complete artifacts; generate into an empty standard case/da3 directory."""
    import yaml
    from worldmembench.errors import EvaluationError

    paths = sample["artifacts"]
    pose = Path(paths["da3"])
    scene = Path(paths.get("da3_scene") or pose.parent / "scene_geometry.npz")
    if pose.is_file() and (not require_scene or scene.is_file()):
        return
    # Keep outputs beside the case's video link, not beside its external target.
    video = Path(sample["video"]).absolute()
    destination = video.parent / "da3"
    if pose != destination / "results.npz" or scene != destination / "scene_geometry.npz":
        raise EvaluationError("missing manifest DA3 artifacts; automatic generation uses the standard case/da3 paths")
    if destination.exists() and any(destination.iterdir()):
        raise EvaluationError(f"incomplete DA3 artifacts in {destination}; move the existing directory aside before regenerating")
    if not context.device.startswith("cuda"):
        raise EvaluationError("DA3 generation requires CUDA; CPU scoring can reuse existing DA3 artifacts")
    config = context.config.get("da3") or {}
    source = Path(config.get("source") or PROJECT_ROOT / "third_party/Depth-Anything-3").expanduser().resolve()
    dinov2 = Path(config.get("dinov2_source") or PROJECT_ROOT / "third_party/dinov2").expanduser().resolve()
    runner = source / "da3_streaming/da3_streaming.py"
    if not runner.is_file() or not (dinov2 / "hubconf.py").is_file():
        raise EvaluationError("DA3 source is missing; run git submodule update --init --recursive, then bash scripts/setup_da3_sources.sh")
    weights = Path(config["weights"]).expanduser().resolve() if config.get("weights") else (
        context.cache_dir / "metrics" / weight_metric if weight_metric is not None else context.cache_dir / "models")
    settings = streaming_config(weights)
    for path in settings["Weights"].values():
        if not Path(path).is_file():
            raise EvaluationError(f"missing DA3 weight: {path}; download the weights with hf download, then run scripts/download_weights.py --verify")
    python = config.get("python")
    if python:
        command = [str(Path(python).expanduser())]
    else:
        conda = shutil.which("conda")
        if conda is None:
            raise EvaluationError("conda is required to run the worldmembench-da3 environment")
        command = [conda, "run", "--no-capture-output", "-n", "worldmembench-da3", "python"]
    index = int(context.device.split(":")[1]) if ":" in context.device else 0
    env = dict(os.environ)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index].strip() if visible else str(index)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               DINOV2_REPO_DIR=str(dinov2))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(source / "src"), env.get("PYTHONPATH"))))
    with TemporaryDirectory(prefix=".da3-", dir=video.parent) as directory:
        work = Path(directory)
        count = extract_frames(video, work / "frames")
        config_path = work / "config.yaml"
        config_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        # Keep verbose native output off the terminal and out of memory.
        with TemporaryFile(dir=work) as log:
            try:
                subprocess.run([*command, str(runner), "--image_dir", str(work / "frames"),
                                "--config", str(config_path), "--output_dir", str(work / "native")],
                               cwd=runner.parent, env=env, check=True,
                               stdout=log, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as error:
                end = log.seek(0, os.SEEK_END)
                log.seek(max(0, end - 8192))
                tail = "\n".join(log.read().decode("utf-8", errors="replace").splitlines()[-20:])
                raise EvaluationError(f"DA3 exited with code {error.returncode}\n{tail}".rstrip()) from error
        convert_outputs(work / "native", work / "artifacts", count)
        # Publish both files together. rename refuses to overwrite a non-empty directory.
        (work / "artifacts").rename(destination)


def prepare_videos(source: str | Path, context) -> bool:
    """Prepare one case/video.mp4 or recursively prepare case folders, without scoring."""
    source = Path(source).expanduser().absolute()
    if source.is_dir():
        videos = sorted(path for path in source.rglob("video.mp4") if path.is_file())
    elif source.is_file() and source.name == "video.mp4":
        videos = [source]
    else:
        raise ValueError(f"expected a case video.mp4 or directory of cases: {source}")
    if not videos:
        raise ValueError(f"no video.mp4 files found: {source}")
    success = True
    print(f"[{context.device}][DA3] 0/{len(videos)}", flush=True)
    for index, video in enumerate(videos, 1):
        directory = video.parent / "da3"
        sample = {"video": str(video), "artifacts": {
            "da3": str(directory / "results.npz"), "da3_scene": str(directory / "scene_geometry.npz"),
        }}
        try:
            ensure_artifacts(sample, context)
            print(f"[{context.device}][DA3] {index}/{len(videos)} ready: {directory.parent}", flush=True)
        except Exception as error:
            success = False
            print(f"[{context.device}][DA3] {index}/{len(videos)} failed: {video}: {error}", file=sys.stderr, flush=True)
    return success


# SALAD loads its trained backbone from its checkpoint; initialization stays offline.
SALAD_OFFLINE_PATCH = """--- a/models/backbones/dinov2.py
+++ b/models/backbones/dinov2.py
@@ -1,2 +1,3 @@
+import os
 import torch
 import torch.nn as nn
@@ -36,3 +37,6 @@
         assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
-        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
+        self.model = torch.hub.load(
+            os.environ['DINOV2_REPO_DIR'], model_name,
+            source='local', pretrained=False,
+        )
         self.num_channels = DINOV2_ARCHS[model_name]
"""


def patch_salad(source: Path) -> None:
    """Apply the install-time offline patch without importing evaluation dependencies."""
    source = source.expanduser().resolve()
    applied = subprocess.run(
        ["git", "apply", "--reverse", "--check", "-"], cwd=source,
        input=SALAD_OFFLINE_PATCH, text=True, capture_output=True,
    )
    if applied.returncode != 0:
        subprocess.run(["git", "apply", "-"], cwd=source,
                       input=SALAD_OFFLINE_PATCH, text=True, check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        help="Case video.mp4 or directory (recursively finds video.mp4)")
    parser.add_argument("--device", default="cuda:0", help="Device for DA3, e.g. cuda:0")
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--patch-salad", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.patch_salad is not None:
        if args.input is not None:
            parser.error("--patch-salad cannot be combined with video input")
        try:
            patch_salad(args.patch_salad)
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        return 0
    if args.input is None:
        parser.error("a video.mp4 or directory of cases is required")

    from worldmembench.context import EvaluationContext
    from worldmembench.errors import WorldMemBenchError
    from worldmembench.weights import DEFAULT_WEIGHTS_DIR

    try:
        context = EvaluationContext(device=args.device, cache_dir=args.weights_dir or DEFAULT_WEIGHTS_DIR)
        try:
            success = prepare_videos(args.input, context)
        finally:
            context.close()
    except (WorldMemBenchError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
