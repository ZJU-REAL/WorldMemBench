"""Prepare full-frame ViPE poses for video.mp4 cases using the default pose-only pipeline."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory, TemporaryFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def vipe_command(python, source, video, output):
    return [*python, str(source / "run.py"), "pipeline=default", "streams=raw_mp4_stream",
            "streams.base_path=" + json.dumps(str(video), ensure_ascii=False), "streams.frame_start=0", "streams.frame_end=-1",
            "streams.frame_skip=1", "streams.cached=true", "pipeline.post.depth_align_model=null",
            "pipeline.output.path=" + json.dumps(str(output), ensure_ascii=False), "pipeline.output.save_artifacts=true",
            "pipeline.output.save_slam_map=false", "pipeline.output.save_viz=false"]


def convert_output(native: Path, output: Path, total_frames: int):
    import numpy as np
    from worldmembench.errors import EvaluationError
    from worldmembench.metrics.camera.schemas import PoseTrajectory

    with np.load(native, allow_pickle=False) as data:
        poses = np.asarray(data["data"], dtype=np.float64)
        indices = np.asarray(data["inds"])
    if poses.shape != (total_frames, 4, 4) or not np.array_equal(indices, np.arange(total_frames)):
        raise EvaluationError("ViPE must register every frame in exact 0..F-1 order; sparse output is invalid")
    PoseTrajectory(poses, source="vipe")
    np.savez(output, c2w=poses, frame_indices=indices.astype(np.int64))


def ensure_artifacts(sample, context):
    from worldmembench.errors import EvaluationError
    from worldmembench.video import VideoReader
    from worldmembench.weights import require_vipe_weights

    output = Path(sample["artifacts"]["vipe"])
    if output.is_file():
        return
    video = Path(sample["video"]).absolute()
    if output != video.parent / "vipe/results.npz":
        raise EvaluationError("missing manifest ViPE artifact; automatic generation uses case/vipe/results.npz")
    if output.parent.exists() and any(output.parent.iterdir()):
        raise EvaluationError(f"incomplete ViPE artifacts in {output.parent}; move them aside before regenerating")
    if not context.device.startswith("cuda"):
        raise EvaluationError("ViPE generation requires CUDA; CPU scoring can reuse existing poses")
    config = context.config.get("vipe") or {}
    source = Path(config.get("source") or PROJECT_ROOT / "third_party/vipe").expanduser().resolve()
    if not (source / "run.py").is_file():
        raise EvaluationError("ViPE source is missing; run git submodule update --init --recursive, then bash scripts/setup_vipe_sources.sh")
    weights = Path(config.get("weights") or context.cache_dir / "metrics/camera_translation/vipe").expanduser().resolve()
    require_vipe_weights(weights)
    if config.get("python"):
        python = [str(Path(config["python"]).expanduser())]
    else:
        conda = shutil.which("conda")
        if conda is None:
            raise EvaluationError("conda is required to run the worldmembench-vipe environment")
        python = [conda, "run", "--no-capture-output", "-n", "worldmembench-vipe", "python"]
    count = VideoReader().probe(str(video)).total_frames
    env = dict(os.environ)
    index = int(context.device.split(":")[1]) if ":" in context.device else 0
    visible = env.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index].strip() if visible else str(index)
    env.update(TORCH_HOME=str(weights / "torch"), HF_HOME=str(weights / "huggingface"),
               HF_HUB_CACHE=str(weights / "huggingface/hub"),
               TRANSFORMERS_CACHE=str(weights / "huggingface/hub"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    with TemporaryDirectory(prefix=".vipe-", dir=video.parent) as directory:
        work = Path(directory)
        native = work / "native"
        with TemporaryFile(dir=work) as log:
            try:
                subprocess.run(vipe_command(python, source, video, native), cwd=source, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                end = log.seek(0, os.SEEK_END)
                log.seek(max(0, end - 8192))
                tail = "\n".join(log.read().decode("utf-8", errors="replace").splitlines()[-20:])
                raise EvaluationError(f"ViPE failed (exit {error.returncode}):\n{tail}") from error
        converted = work / "results.npz"
        convert_output(native / "pose" / f"{video.stem}.npz", converted, count)
        output.parent.mkdir(exist_ok=True)
        output.hardlink_to(converted)


def main():
    from worldmembench.context import EvaluationContext
    from worldmembench.weights import DEFAULT_WEIGHTS_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A video.mp4 or directory containing case folders")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    args = parser.parse_args()
    source = args.input.expanduser().absolute()
    videos = sorted(source.rglob("video.mp4")) if source.is_dir() else [source]
    if not videos or any(not video.is_file() or video.name != "video.mp4" for video in videos):
        parser.error("provide video.mp4 or a directory containing case folders with video.mp4")
    context = EvaluationContext(device=args.device, cache_dir=args.weights_dir)
    failed = False
    try:
        for index, video in enumerate(videos, 1):
            try:
                ensure_artifacts({"video": str(video), "artifacts": {"vipe": str(video.parent / "vipe/results.npz")}}, context)
            except Exception as error:
                failed = True
                print(f"[{args.device}][ViPE] {index}/{len(videos)} failed: {video}: {error}", flush=True)
            else:
                print(f"[{args.device}][ViPE] {index}/{len(videos)} ready: {video}", flush=True)
    finally:
        context.close()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
