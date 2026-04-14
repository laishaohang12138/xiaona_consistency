import argparse
import json
import subprocess
from pathlib import Path

DEFAULT_DISTRO = "Ubuntu"
DEFAULT_USER = "root"
DEFAULT_WORKSPACE = "/opt/xiaona_densepose"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
DEFAULT_DENSEPOSE_WEIGHT_URL = (
    "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
)
DEFAULT_DEPLOY_DIR = Path(__file__).resolve().parent / "external" / "models" / "DensePose"
DEFAULT_LOCAL_SOURCE_DIR = Path(__file__).resolve().parent / "external" / "vendor" / "detectron2-main"
SCRIPT_SCHEMA = "densepose_wsl_bootstrap_v1"


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16le", "gbk", "cp936"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def _run_wsl_probe(distro: str, user: str, timeout: int = 25) -> dict:
    command = ["wsl.exe", "-d", distro, "-u", user, "--exec", "/bin/true"]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=timeout)
        return {
            "ready": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": _decode(completed.stdout or b""),
            "stderr": _decode(completed.stderr or b""),
            "command": command,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ready": False,
            "returncode": None,
            "stdout": _decode(exc.stdout or b""),
            "stderr": _decode(exc.stderr or b""),
            "command": command,
            "timeout_seconds": timeout,
        }


def _windows_path_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    return f"/mnt/{drive}{resolved.as_posix()[2:]}"


def _bootstrap_script(workspace: str, torch_index_url: str, weight_url: str, local_source_dir: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="{workspace}"
ENV_DIR="${{WORKSPACE}}/.venv"
REPO_DIR="${{WORKSPACE}}/detectron2"
WEIGHT_DIR="${{WORKSPACE}}/weights"
WEIGHT_FILE="${{WEIGHT_DIR}}/densepose_rcnn_R_50_FPN_s1x.pkl"
LOCAL_SOURCE_DIR="{local_source_dir}"

export DEBIAN_FRONTEND=noninteractive

proxy_vars=(http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy no_proxy NO_PROXY)
for proxy_name in "${{proxy_vars[@]}}"; do
  proxy_value="${{!proxy_name:-}}"
  case "$proxy_value" in
    *127.0.0.1*|*localhost*)
      unset "$proxy_name"
      ;;
  esac
done

if [ "$(id -u)" = "0" ]; then
  APT_GET="apt-get"
else
  APT_GET="sudo apt-get"
fi

$APT_GET -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false update
$APT_GET -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false install -y \\
  build-essential \\
  cmake \\
  ffmpeg \\
  git \\
  libglib2.0-0 \\
  libsm6 \\
  libxext6 \\
  libxrender-dev \\
  python3 \\
  python3-pip \\
  python3-venv \\
  wget

mkdir -p "${{WORKSPACE}}" "${{WEIGHT_DIR}}"

if [ -d "${{LOCAL_SOURCE_DIR}}/projects/DensePose" ]; then
  rm -rf "${{REPO_DIR}}"
  mkdir -p "${{REPO_DIR}}"
  cp -a "${{LOCAL_SOURCE_DIR}}/." "${{REPO_DIR}}/"
elif [ ! -d "${{REPO_DIR}}" ]; then
  git clone https://github.com/facebookresearch/detectron2.git "${{REPO_DIR}}"
else
  git -C "${{REPO_DIR}}" pull --ff-only
fi

python3 -m venv "${{ENV_DIR}}"
source "${{ENV_DIR}}/bin/activate"
unset PIP_PROXY
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url {torch_index_url}
python -m pip install -e "${{REPO_DIR}}" --no-build-isolation
python -m pip install -e "${{REPO_DIR}}/projects/DensePose" --no-build-isolation --no-deps
if [ ! -f "${{WEIGHT_FILE}}" ]; then
  wget -O "${{WEIGHT_FILE}}" {weight_url}
fi

python - <<'PY'
import importlib
mods = ['detectron2', 'densepose']
for name in mods:
    module = importlib.import_module(name)
    print(name, getattr(module, '__file__', ''))
PY

echo "DensePose bootstrap complete"
echo "workspace=${{WORKSPACE}}"
echo "venv=${{ENV_DIR}}"
echo "weights=${{WEIGHT_FILE}}"
"""


def _write_bootstrap_file(deploy_dir: Path, content: str) -> Path:
    deploy_dir.mkdir(parents=True, exist_ok=True)
    script_path = deploy_dir / "bootstrap_densepose_wsl.sh"
    script_path.write_text(content, encoding="utf-8", newline="\n")
    return script_path


def _run_bootstrap(distro: str, user: str, script_path: Path, timeout: int) -> dict:
    linux_script = "/mnt/" + script_path.drive[0].lower() + script_path.as_posix()[2:]
    command = ["wsl.exe", "-d", distro, "-u", user, "--exec", "/bin/bash", linux_script]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=timeout)
        return {
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout": _decode(completed.stdout or b""),
            "stderr": _decode(completed.stderr or b""),
            "command": command,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "returncode": None,
            "stdout": _decode(exc.stdout or b""),
            "stderr": _decode(exc.stderr or b""),
            "command": command,
            "timeout_seconds": timeout,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run official DensePose deployment inside WSL Ubuntu for XiaoNa clothing-invariant sidecars.")
    parser.add_argument("--distro", default=DEFAULT_DISTRO)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--torch-index-url", default=DEFAULT_TORCH_INDEX_URL)
    parser.add_argument("--weight-url", default=DEFAULT_DENSEPOSE_WEIGHT_URL)
    parser.add_argument("--deploy-dir", type=Path, default=DEFAULT_DEPLOY_DIR)
    parser.add_argument("--local-source-dir", type=Path, default=DEFAULT_LOCAL_SOURCE_DIR)
    parser.add_argument("--run", action="store_true", help="Attempt to run the bootstrap script inside WSL after writing it.")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    bootstrap = _bootstrap_script(
        args.workspace,
        args.torch_index_url,
        args.weight_url,
        _windows_path_to_wsl(args.local_source_dir),
    )
    script_path = _write_bootstrap_file(args.deploy_dir.resolve(), bootstrap)
    manifest = {
        "schema_version": SCRIPT_SCHEMA,
        "distro": args.distro,
        "user": args.user,
        "workspace": args.workspace,
        "torch_index_url": args.torch_index_url,
        "weight_url": args.weight_url,
        "local_source_dir": str(args.local_source_dir.resolve()),
        "script_path": str(script_path),
    }
    (args.deploy_dir.resolve() / "bootstrap_densepose_wsl.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    result = {
        "status": "prepared",
        "deployment": manifest,
        "wsl_probe": _run_wsl_probe(args.distro, args.user),
    }
    if args.run:
        result["run_result"] = _run_bootstrap(args.distro, args.user, script_path.resolve(), args.timeout_seconds)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
