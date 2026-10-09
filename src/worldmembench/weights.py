"""Download evaluation checkpoints and share them across metrics."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from worldmembench.paths import DEFAULT_WEIGHTS_DIR


FILE_CHECKSUMS = json.loads(Path(__file__).with_name("weight_checksums.json").read_text(encoding="utf-8"))["files"]


@dataclass(frozen=True)
class Weight:
    path: str
    url: str
    sha256_prefix: str = ""


WEIGHTS = {
    "imaging_quality": (
        Weight(
            "models/musiq_spaq_ckpt-358bb6af.pth",
            "https://github.com/chaofengc/IQA-PyTorch/releases/download/"
            "v0.1-weights/musiq_spaq_ckpt-358bb6af.pth",
            "358bb6af275e28ea56821d44fb55c6cb83645db11f394d3ad65b2d149965ab50",
        ),
    ),
    "aesthetic_quality": (
        Weight(
            "models/ViT-L-14.pt",
            "https://openaipublic.azureedge.net/clip/models/"
            "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
        ),
        Weight(
            "models/sa_0_4_vit_l_14_linear.pth",
            "https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/"
            "main/sa_0_4_vit_l_14_linear.pth",
            "2cd4e60f4f24ae3bcd57b847b13c1f3ba27edc28cc1a7f9ce74ee9f421243cba",
        ),
    ),
}


BUFFALO_L = Weight(
    "models/buffalo_l.zip",
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
)
CLIP_REID_WEIGHT = Weight("models/ViT-B-32.pt",
           "https://openaipublic.azureedge.net/clip/models/"
           "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
           "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af")
# Physical model files live in models/; metric directories contain only links.
SAM2_WEIGHTS = tuple(Weight(f"models/sam2-hiera-large/{name}",
             "https://huggingface.co/facebook/sam2-hiera-large/resolve/"
             f"e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251/{name}",
             "ca82d5aa03964f3ff14a215b6da8e6689958893cd43682bf631054afed34133c"
             if name == "model.safetensors" else "")
      for name in ("config.json", "preprocessor_config.json", "processor_config.json",
                   "video_preprocessor_config.json", "model.safetensors",
                   "sam2_hiera_l.yaml", "README.md", ".gitattributes"))
DINOV2_WEIGHTS = tuple(Weight(f"models/dinov2-large/{name}",
             "https://huggingface.co/facebook/dinov2-large/resolve/"
             f"47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/{name}",
             "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"
             if name == "model.safetensors" else "")
      for name in ("config.json", "model.safetensors", "preprocessor_config.json",
                   "README.md", ".gitattributes"))
HUMAN_TRACKING_WEIGHTS = (
    BUFFALO_L,
    Weight("models/yolov8s-worldv2.pt",
           "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt",
           "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792"),
    CLIP_REID_WEIGHT,
    *SAM2_WEIGHTS,
)
WEIGHTS["human_identity"] = HUMAN_TRACKING_WEIGHTS
WEIGHTS["human_appearance"] = (*HUMAN_TRACKING_WEIGHTS, *DINOV2_WEIGHTS)

# Object detection uses the Grounding DINO Base checkpoint.
GROUNDING_DINO_WEIGHTS = tuple(
    Weight(f"models/grounding-dino-base/{name}",
           "https://huggingface.co/IDEA-Research/grounding-dino-base/resolve/"
           f"12bdfa3120f3e7ec7b434d90674b3396eccf88eb/{name}",
           "5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21"
           if name == "model.safetensors" else "")
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json",
                 "tokenizer.json", "special_tokens_map.json", "vocab.txt",
                 "model.safetensors", "README.md", ".gitattributes")
)
OBJECT_TRACKING_WEIGHTS = (*SAM2_WEIGHTS, CLIP_REID_WEIGHT, *GROUNDING_DINO_WEIGHTS)
WEIGHTS["object_appearance"] = (*OBJECT_TRACKING_WEIGHTS, *DINOV2_WEIGHTS)
WEIGHTS["object_geometry"] = OBJECT_TRACKING_WEIGHTS

# Shared video geometry preparation, not a Camera Following score.
DA3_WEIGHTS = tuple(
    Weight(f"models/DA3NESTED-GIANT-LARGE-1.1/{name}",
           "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/"
           f"b2359bdf726fb44ef62acca04d629dcf158053e7/{name}",
           "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c"
           if name == "model.safetensors" else "")
    for name in ("config.json", "model.safetensors", "README.md", ".gitattributes")
)
SALAD_WEIGHT = Weight(
    "models/dino_salad.ckpt",
    "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
    "6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada",
)
GEOMETRY_WEIGHTS = (*DA3_WEIGHTS, SALAD_WEIGHT)
SEGFORMER_WEIGHTS = tuple(
    Weight(f"models/segformer-b0-ade20k/{name}",
           "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/"
           f"b9175de73a0a34f7843135853d27629aa6987b2f/{name}",
           "6ae39addd01de6b1b8bde2cf677d43a5cd733424b8d186de3f95d1c51fee23f9"
           if name == "model.safetensors" else "")
    for name in ("config.json", "preprocessor_config.json", "model.safetensors",
                 "README.md", ".gitattributes")
)
WEIGHTS["environment_visible_memory"] = (*GEOMETRY_WEIGHTS, *DINOV2_WEIGHTS)
WEIGHTS["environment_depth_fidelity"] = GEOMETRY_WEIGHTS
WEIGHTS["environment_scene_reconstruction"] = (*GEOMETRY_WEIGHTS, *SEGFORMER_WEIGHTS)

VIPE_WEIGHTS = (
    Weight("models/vipe/torch/hub/sam/sam_vit_b_01ec64.pth",
           "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
           "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"),
    Weight("models/vipe/torch/hub/aot/R50_DeAOTL_PRE_YTB_DAV.pth", "gdrive:1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ",
           "7e8a8d83310739bac02817f6bf48b6bbe2bbd7d5325722f1084088eb3aee1e06"),
    Weight("models/vipe/torch/hub/droid_slam/droid.pth", "gdrive:1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh",
           "46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526"),
    Weight("models/vipe/torch/hub/geocalib/pinhole.tar",
           "https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar",
           "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac"),
    Weight("models/vipe/torch/hub/checkpoints/groundingdino_swint_ogc.pth",
           "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/"
           "a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swint_ogc.pth",
           "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"),
)
# Keep the upstream cache layout so ViPE can load everything offline without patches.
VIPE_HF_REPOS = (
    ("bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594",
     ("config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json", "vocab.txt")),
    ("lpiccinelli/unidepth-v2-vitl14", "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9",
     ("config.json", "model.safetensors")),
)
WEIGHTS["camera_rotation"] = GEOMETRY_WEIGHTS
WEIGHTS["camera_translation"] = (*GEOMETRY_WEIGHTS, *VIPE_WEIGHTS)


def download_vipe_weights(root: Path, *, skip_existing: bool = True) -> None:
    cache = root / "models/vipe/huggingface/hub"
    for repo, revision, required in VIPE_HF_REPOS:
        repository = cache / ("models--" + repo.replace("/", "--"))
        snapshot = repository / "snapshots" / revision
        complete = all((snapshot / name).is_file() and (snapshot / name).stat().st_size for name in required)
        if complete and not skip_existing:
            raise FileExistsError(f"refusing to replace existing model: {snapshot}")
        if complete:
            print(f"Keeping existing ViPE model (not reverified): {snapshot}", flush=True)
        else:
            command = ["hf", "download", repo, "--revision", revision, "--cache-dir", str(cache)]
            for pattern in ("*.json", "*.txt", "*.md", "LICENSE*", ".gitattributes", "model.safetensors"):
                command.extend(["--include", pattern])
            print(f"Downloading ViPE model package: {repo}\n  -> {snapshot}", flush=True)
            try:
                subprocess.run(command, check=True)
            except FileNotFoundError as error:
                raise OSError("hf is required; install worldmembench[camera]") from error
            except subprocess.CalledProcessError as error:
                raise OSError(f"hf download failed for {repo} (exit {error.returncode}); see the download error above") from error
            if not all((snapshot / name).is_file() and (snapshot / name).stat().st_size for name in required):
                raise ValueError(f"incomplete ViPE model package: {snapshot}")
        # The upstream model loaders request 'main'; bind it to the selected snapshot.
        (repository / "refs").mkdir(parents=True, exist_ok=True)
        (repository / "refs/main").write_text(revision, encoding="utf-8")


def require_vipe_weights(directory: Path) -> None:
    missing = [str(directory / Path(weight.path).relative_to("models/vipe")) for weight in VIPE_WEIGHTS
               if not (directory / Path(weight.path).relative_to("models/vipe")).is_file()]
    for repo, revision, required in VIPE_HF_REPOS:
        base = directory / "huggingface/hub" / ("models--" + repo.replace("/", "--"))
        ref = base / "refs/main"
        if not ref.is_file() or ref.read_text().strip() != revision:
            missing.append(str(ref))
        missing.extend(str(base / "snapshots" / revision / name) for name in required
                       if not (base / "snapshots" / revision / name).is_file())
    if missing:
        raise FileNotFoundError("ViPE weights are incomplete; run scripts/download_weights.py --metrics camera_translation. "
                                "Missing: " + ", ".join(missing))

# Large serving models are opt-in; the default download remains unchanged.
HF_REPOSITORIES = {"dynamic_memory": "Qwen/Qwen3.8-27B"}
QWEN_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def download_vlm_weights(root: Path, *, skip_existing: bool = True) -> Path:
    repo = HF_REPOSITORIES["dynamic_memory"]
    destination = root / "models" / repo.rsplit("/", 1)[1]

    def complete():
        index = destination / "model.safetensors.index.json"
        if not index.is_file():
            return False
        shards = set(json.loads(index.read_text())["weight_map"].values())
        required = {"config.json", "preprocessor_config.json", "tokenizer_config.json",
                    "tokenizer.json", "chat_template.jinja", "generation_config.json",
                    "video_preprocessor_config.json", "vocab.json", "merges.txt", "README.md", "LICENSE", *shards}
        return bool(shards) and all((destination / name).is_file() and (destination / name).stat().st_size
                                    for name in required)

    if skip_existing and complete():
        print(f"Keeping existing Qwen package (not reverified): {destination}", flush=True)
        return destination
    if destination.exists() and not skip_existing:
        raise FileExistsError(f"refusing to overwrite existing model directory: {destination}")
    print(f"Downloading full model repository via hf download: {repo}\n  -> {destination}", flush=True)
    try:
        subprocess.run(["hf", "download", repo, "--revision", QWEN_REVISION,
                        "--local-dir", str(destination)], check=True)
    except FileNotFoundError as error:
        raise OSError("hf is required; install worldmembench[dynamic]") from error
    except subprocess.CalledProcessError as error:
        raise OSError(f"hf download exited with code {error.returncode}") from error
    if not complete():
        raise ValueError(f"incomplete Qwen model package: {destination}")
    return destination


def _metric_models(metric):
    if metric in HF_REPOSITORIES:
        return [HF_REPOSITORIES[metric].rsplit("/", 1)[1]]
    return list(dict.fromkeys("buffalo_l" if weight == BUFFALO_L else Path(weight.path).parts[1]
                              for weight in WEIGHTS[metric]))


def verify_weights(root: Path, metrics) -> int:
    """Verify local release files without downloading or replacing anything."""
    models = {name for metric in metrics for name in _metric_models(metric)}
    files = {path: checksum for path, checksum in FILE_CHECKSUMS.items()
             if Path(path).parts[1] in models}
    missing_checksums = models - {Path(path).parts[1] for path in files}
    if missing_checksums:
        raise ValueError(f"No verification checksums for: {', '.join(sorted(missing_checksums))}")
    for index, (relative, expected) in enumerate(files.items(), 1):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing weight file: {path}")
        print(f"[{index}/{len(files)}] Verifying {relative}", flush=True)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return len(files)


def link_metric_weights(root: Path, metrics) -> None:
    """Expose downloaded models under each metric using relative symlinks."""
    for metric in metrics:
        for name in _metric_models(metric):
            source = root / "models" / name
            link = root / "metrics" / metric / name
            if not source.exists():
                raise FileNotFoundError(f"model not found: {source}")
            if link.is_symlink() and link.resolve() == source.resolve():
                continue
            if link.exists() or link.is_symlink():
                raise FileExistsError(f"refusing to replace existing model path: {link}")
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(Path("../..") / "models" / name, target_is_directory=source.is_dir())
            print(f"Linked {link} -> {link.readlink()}", flush=True)


def extract_buffalo_l(root: Path, *, skip_existing: bool = False) -> None:
    """Extract all five models in buffalo_l, retaining already extracted files."""
    destination = root / "models" / "buffalo_l"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / BUFFALO_L.path) as archive:
        for name in ("det_10g.onnx", "w600k_r50.onnx", "1k3d68.onnx",
                     "2d106det.onnx", "genderage.onnx"):
            target = destination / name
            if target.is_file() and skip_existing:
                print(f"Keeping existing model: {target}", flush=True)
                continue
            # Fixed member names prevent archive path traversal. Publish only
            # after the ZIP reader has checked the member's CRC.
            with archive.open(name) as source, tempfile.NamedTemporaryFile(dir=destination) as temporary:
                shutil.copyfileobj(source, temporary)
                temporary.flush()
                target.hardlink_to(temporary.name)
            print(f"Extracted {target}", flush=True)


def _verify_weight(path: Path, weight: Weight) -> tuple[int, str]:
    name = Path(weight.path).name
    print(f"Verifying {name}...", flush=True)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    if not size:
        raise ValueError(f"empty download: {name}")
    sha256 = digest.hexdigest()
    if weight.sha256_prefix and not sha256.startswith(weight.sha256_prefix):
        raise ValueError(f"SHA256 mismatch for {name}: {sha256}")
    return size, sha256


def download_hf_weight(weight: Weight, root: Path) -> Path:
    """Use the HF CLI for DA3, retaining its download cache between attempts."""
    target = root / weight.path
    if target.exists():
        raise FileExistsError(f"refusing to overwrite an existing checkpoint: {target}")
    repo, resource = weight.url.removeprefix("https://huggingface.co/").split("/resolve/", 1)
    revision, filename = resource.split("/", 1)
    # Keep HF's partial downloads across interruptions; publish only after our
    # pinned checksum passes, so --skip-existing cannot skip an invalid result.
    staging = target.parent / ".hf-download"
    command = ["hf", "download", repo, filename, "--revision", revision, "--local-dir", str(staging)]
    print(f"Downloading via hf download: {repo}/{filename}\n  -> {target}", flush=True)
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise OSError("hf is required; install worldmembench[environment]") from error
    except subprocess.CalledProcessError as error:
        raise OSError(f"hf download exited with code {error.returncode}; see its error above") from error
    downloaded = staging / filename
    try:
        size, sha256 = _verify_weight(downloaded, weight)
    except ValueError:
        downloaded.unlink(missing_ok=True)
        raise
    target.hardlink_to(downloaded)
    downloaded.unlink()
    print(f"Saved {target.name}: {size} bytes, sha256={sha256}", flush=True)
    return target


def download_weight(weight: Weight, root: Path) -> Path:
    target = root / weight.path
    if target.exists():
        raise FileExistsError(f"refusing to overwrite or reuse an existing checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {weight.url}\n  -> {target}", flush=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part") as handle:
        temporary = Path(handle.name)
        command = [
            "curl", "--fail", "--location", "--show-error",
            "--connect-timeout", "15",
            "--output", str(temporary),
        ]
        if weight.url.startswith("gdrive:"):
            command = [sys.executable, "-m", "gdown", weight.url.removeprefix("gdrive:"), "--output", str(temporary)]
            transport = "gdown"
        else:
            command.append(weight.url)
            transport = "curl"
        print("Connecting (15s timeout); curl shows bytes, speed and time remaining below."
              if transport == "curl" else "Downloading from Google Drive; gdown shows progress below.", flush=True)
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as error:
            raise OSError(f"{transport} is required for this download") from error
        except subprocess.CalledProcessError as error:
            raise OSError(f"{transport} exited with code {error.returncode}; see its error above") from error
        size, sha256 = _verify_weight(temporary, weight)
        target.hardlink_to(temporary)
        print(f"Saved {target.name}: {size} bytes, sha256={sha256}", flush=True)
        return target
