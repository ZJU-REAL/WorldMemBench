"""Publish WorldMemBench packages in two stages after `hf auth login`.

`initial` uploads public examples and weights, then the private inference subset.
After approval, `inference-full --publish` makes inference public and fills it in.
Re-run a failed stage to resume using Hugging Face's upload deduplication.
Public examples and third-party weights must be cleared for publication first.
Weights retain all checkpoint formats, excluding download caches. ViPE snapshot
files and revision refs are retained; metric symlinks are recreated on download.
"""

import argparse
from pathlib import Path
import shlex
import subprocess

from huggingface_hub import HfApi
from huggingface_hub.utils import DEFAULT_IGNORE_PATTERNS, filter_repo_objects

from worldmembench.paths import DEFAULT_DATASETS_DIR


COMMON_EXCLUDES = [
    ".git/**", "**/.git/**", ".cache/**", "**/.cache/**",
    "eval_results/**", "**/eval_results/**", "**/__pycache__/**",
]
INFERENCE_PARTIAL = [
    "README.md", "PUBLIC_RELEASE.md", "LICENSE*", "testsets/**",
    "human/**", "object/**", "dynamic_memory/**", "camera_translation/**",
]
STAGES = {
    "examples": ("WorldMemBench-examples", "dataset", False,
                 ["README.md", "LICENSE*", "data/**", "manifests/**"]),
    "weights": ("WorldMemBench-weights", "model", False,
                ["README.md", "LICENSE*", "models/**"]),
    "inference-partial": ("WorldMemBench", "dataset", True, INFERENCE_PARTIAL),
    "inference-full": ("WorldMemBench", "dataset", False,
                       [*INFERENCE_PARTIAL, "environment/**", "camera_rotation/**",
                        "general_video_quality/**"]),
}


def upload_plan(stage, datasets_dir, namespace):
    name, repo_type, private, includes = STAGES[stage]
    folder = datasets_dir.expanduser().resolve() / name
    if not folder.is_dir():
        raise FileNotFoundError(f"Package directory not found: {folder}")
    excludes = list(COMMON_EXCLUDES)
    if stage == "weights":
        excludes += ["**/blobs/**", "**/trees/**", "**/.locks/**", "**/.hf-download/**",
                     "**/huggingface/download/**", "**/huggingface/.gitignore", "**/CACHEDIR.TAG"]
    paths = list(filter_repo_objects(
        (p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()),
        allow_patterns=includes, ignore_patterns=[*excludes, *DEFAULT_IGNORE_PATTERNS],
    ))
    if not paths:
        raise ValueError(f"No files selected in {folder}")
    repo_id = f"{namespace}/{name}"
    command = ["hf", "upload", repo_id, str(folder), ".", "--repo-type", repo_type,
               "--private" if private else "--no-private"]
    for pattern in includes:
        command += ["--include", pattern]
    for pattern in excludes:
        command += ["--exclude", pattern]
    return {
        "stage": stage, "repo_id": repo_id, "repo_type": repo_type, "private": private,
        "folder": folder, "paths": paths,
        "bytes": sum((folder / name).stat().st_size for name in paths), "command": command,
    }


def upload(plan, api):
    repo_id, repo_type = plan["repo_id"], plan["repo_type"]
    api.create_repo(repo_id, repo_type=repo_type, private=plan["private"], exist_ok=True)
    info = api.repo_info(repo_id, repo_type=repo_type)
    if plan["private"]:
        if not info.private:
            raise RuntimeError(
                f"{repo_id} is already public; refusing the private inference-partial stage. "
                "Use inference-full --publish after approval."
            )
    elif info.private:
        api.update_repo_settings(repo_id, repo_type=repo_type, visibility="public")
        info = api.repo_info(repo_id, repo_type=repo_type)
        if info.private:
            raise RuntimeError(f"Could not confirm public visibility for {repo_id}")
    subprocess.run(plan["command"], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["initial", *STAGES])
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR,
                        help="Maintainer staging root containing WorldMemBench, WorldMemBench-examples and WorldMemBench-weights")
    parser.add_argument("--namespace", default="rescuerz", help="HF user or organization")
    parser.add_argument("--dry-run", action="store_true", help="Show files/size/commands without network calls")
    parser.add_argument("--publish", action="store_true",
                        help="Explicitly authorize making the entire inference repository public after approval")
    args = parser.parse_args(argv)
    if args.stage == "inference-full" and not args.publish and not args.dry_run:
        parser.error("inference-full requires --publish; run it only after publication approval")
    stages = ["examples", "weights", "inference-partial"] if args.stage == "initial" else [args.stage]
    # Plan all stages before creating or changing any repositories.
    plans = [upload_plan(stage, args.datasets_dir, args.namespace) for stage in stages]
    for plan in plans:
        visibility = "private" if plan["private"] else "public"
        print(f"[{plan['stage']}] {plan['repo_id']} ({visibility}): "
              f"{len(plan['paths']):,} files, {plan['bytes'] / 10**9:.2f} GB", flush=True)
        print(shlex.join(plan["command"]), flush=True)
    if args.dry_run:
        return 0
    print("Public stages expose the repository's existing files as well as new uploads. "
          "Check publication permissions and available HF storage before proceeding.", flush=True)
    api = HfApi()
    for plan in plans:
        print(f"Uploading {plan['stage']}...", flush=True)
        upload(plan, api)
    print("Selected upload stages completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
