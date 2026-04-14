import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_DISTRO = "Ubuntu"
DEFAULT_USER = "root"
DEFAULT_WORKSPACE = "/opt/xiaona_densepose"
DEFAULT_CONFIG = "/opt/xiaona_densepose/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
DEFAULT_CHECKPOINT = "/opt/xiaona_densepose/weights/model_final_162be9.pkl"


def _image_files(input_dir: Path) -> List[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}{resolved.as_posix()[2:]}"


def _run(command: List[str], *, timeout_ms: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=max(1, int(timeout_ms / 1000)))


def _wsl_python_args(distro: str, user: str) -> List[str]:
    return ["wsl.exe", "-d", distro, "-u", user, "--exec", f"{DEFAULT_WORKSPACE}/.venv/bin/python"]


def _wsl_exec_args(distro: str, user: str) -> List[str]:
    return ["wsl.exe", "-d", distro, "-u", user, "--exec"]


def _stable_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{path.stem}_{digest}"


def _ensure_wsl_ready(distro: str, user: str) -> None:
    command = _wsl_exec_args(distro, user) + ["/bin/true"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if completed.returncode != 0:
        raise RuntimeError(f"WSL is not ready: {completed.stderr or completed.stdout}")


def _copy_to_wsl(distro: str, user: str, source_image: Path, workspace: str) -> str:
    stable_id = _stable_id(source_image)
    target = f"{workspace}/inference_inputs/{stable_id}{source_image.suffix.lower()}"
    subprocess.run(
        _wsl_exec_args(distro, user) + ["/bin/mkdir", "-p", f"{workspace}/inference_inputs", f"{workspace}/inference_dumps"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        _wsl_exec_args(distro, user) + ["/bin/cp", _windows_to_wsl(source_image), target],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return target


def _run_densepose_dump(
    distro: str,
    user: str,
    *,
    workspace: str,
    config: str,
    checkpoint: str,
    staged_image: str,
    dump_file: str,
) -> None:
    command = _wsl_python_args(distro, user) + [
        f"{workspace}/detectron2/projects/DensePose/apply_net.py",
        "dump",
        config,
        checkpoint,
        staged_image,
        "--output",
        dump_file,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)


def _convert_dump_to_sidecar(
    distro: str,
    user: str,
    *,
    dump_file: str,
    output: Path,
    source_image: Path,
    workspace: str,
    min_score: float,
) -> None:
    command = _wsl_python_args(distro, user) + [
        _windows_to_wsl(Path(__file__).resolve().parent / "export_surface_occlusion_densepose_wsl.py"),
        "--dump-file",
        dump_file,
        "--output",
        _windows_to_wsl(output),
        "--source-image",
        _windows_to_wsl(source_image),
        "--min-score",
        str(min_score),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)


def _process_image(
    distro: str,
    user: str,
    *,
    workspace: str,
    config: str,
    checkpoint: str,
    source_image: Path,
    output: Path,
    min_score: float,
) -> dict:
    staged_image = _copy_to_wsl(distro, user, source_image, workspace)
    dump_file = f"{workspace}/inference_dumps/{_stable_id(source_image)}.pkl"
    _run_densepose_dump(
        distro,
        user,
        workspace=workspace,
        config=config,
        checkpoint=checkpoint,
        staged_image=staged_image,
        dump_file=dump_file,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _convert_dump_to_sidecar(
        distro,
        user,
        dump_file=dump_file,
        output=output,
        source_image=source_image,
        workspace=workspace,
        min_score=min_score,
    )
    return {"image": str(source_image.resolve()), "output": str(output.resolve()), "dump_file": dump_file}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deployed DensePose in WSL and emit XiaoNa .densepose.json sidecars.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Single image to process.")
    group.add_argument("--input-dir", type=Path, help="Directory of images to process sequentially.")
    group.add_argument("--dump-file", type=Path, help="Existing DensePose dump file to convert only.")
    parser.add_argument("--output", type=Path, default=None, help="Output file for --image or --dump-file mode.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for --input-dir mode.")
    parser.add_argument("--source-image", type=Path, default=None, help="Required with --dump-file when the dump does not contain a /mnt/<drive>/ path.")
    parser.add_argument("--distro", default=DEFAULT_DISTRO)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--min-score", type=float, default=0.5)
    args = parser.parse_args()

    _ensure_wsl_ready(args.distro, args.user)

    if args.dump_file:
        if args.output is None:
            raise ValueError("--output is required with --dump-file")
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        command = _wsl_python_args(args.distro, args.user) + [
            _windows_to_wsl(Path(__file__).resolve().parent / "export_surface_occlusion_densepose_wsl.py"),
            "--dump-file",
            _windows_to_wsl(args.dump_file.resolve()),
            "--output",
            _windows_to_wsl(output),
            "--min-score",
            str(args.min_score),
        ]
        if args.source_image is not None:
            command.extend(["--source-image", _windows_to_wsl(args.source_image.resolve())])
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
        print(json.dumps({"status": "ok", "items": [{"dump_file": str(args.dump_file.resolve()), "output": str(output)}]}, ensure_ascii=False))
        return

    images: Iterable[Path]
    if args.image is not None:
        images = [args.image.resolve()]
    else:
        images = [path.resolve() for path in _image_files(args.input_dir.resolve())]

    rows = []
    for image_path in images:
        if args.output is not None and args.image is not None:
            output = args.output.resolve()
        else:
            output_dir = args.output_dir.resolve() if args.output_dir is not None else image_path.parent
            output = output_dir / f"{image_path.name}.densepose.json"
        rows.append(
            _process_image(
                args.distro,
                args.user,
                workspace=args.workspace,
                config=args.config,
                checkpoint=args.checkpoint,
                source_image=image_path,
                output=output,
                min_score=float(args.min_score),
            )
        )
    print(json.dumps({"status": "ok", "count": len(rows), "items": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
