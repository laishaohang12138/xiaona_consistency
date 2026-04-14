import argparse
import json
import shutil
from datetime import datetime, UTC
from pathlib import Path

DEFAULT_MODEL_ID = "facebook/sam2.1-hiera-tiny"
DEFAULT_DEPLOY_DIR = Path(__file__).resolve().parent / "external" / "models" / "SAM2"
DEPLOY_SCHEMA = "sam2_local_deploy_v1"


def _deploy(model_id: str, deploy_dir: Path, force: bool = False) -> dict:
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES

    filenames = HF_MODEL_ID_TO_FILENAMES.get(model_id)
    if filenames is None:
        raise ValueError(f"unsupported model id: {model_id}")
    config_name, checkpoint_name = filenames
    cache_checkpoint = Path(hf_hub_download(repo_id=model_id, filename=checkpoint_name))
    deploy_dir.mkdir(parents=True, exist_ok=True)
    deployed_checkpoint = deploy_dir / checkpoint_name
    if force or not deployed_checkpoint.exists():
        shutil.copy2(cache_checkpoint, deployed_checkpoint)
    manifest = {
        "schema_version": DEPLOY_SCHEMA,
        "model_id": model_id,
        "config_name": config_name,
        "checkpoint_name": checkpoint_name,
        "cache_checkpoint_path": str(cache_checkpoint.resolve()),
        "deployed_checkpoint_path": str(deployed_checkpoint.resolve()),
        "deployed_at_utc": datetime.now(UTC).isoformat(),
    }
    manifest_path = deploy_dir / f"{checkpoint_name}.deploy.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SAM2 checkpoint if needed and deploy it into the project-local model directory.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--deploy-dir", type=Path, default=DEFAULT_DEPLOY_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = _deploy(args.model_id, args.deploy_dir.resolve(), force=args.force)
    print(json.dumps({"status": "ok", "deployment": manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
