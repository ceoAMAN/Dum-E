#!/usr/bin/env python
from __future__ import annotations
import os
import sys
import shutil
import argparse
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
from huggingface_hub import HfApi, login

def main():
    parser = argparse.ArgumentParser(description="Synchronize Sturnus checkpoints and states with HuggingFace.")
    parser.add_argument("--push", action="store_true", help="Upload local state directory to HF.")
    parser.add_argument("--pull", action="store_true", help="Download HF state snapshot to local state.")
    args = parser.parse_args()

    if not args.push and not args.pull:
        parser.print_help()
        sys.exit(1)

    token = configs.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if not token:
        print("[error] HF_TOKEN is not configured in configs.py or environment variables.", file=sys.stderr)
        sys.exit(1)

    print("[hf_sync] Authenticating with HuggingFace Hub...")
    try:
        login(token=token, add_to_git_credential=False)
        api = HfApi(token=token)
        user = api.whoami()["name"]
    except Exception as e:
        print(f"[error] HuggingFace authentication failed: {e}", file=sys.stderr)
        sys.exit(1)

    repo_id = f"{user}/sturnus-state"
    print(f"[hf_sync] HF User: {user} | Target Repository: {repo_id}")

    state_path = ROOT / "state"

    if args.push:
        if not state_path.exists() or not any(state_path.iterdir()):
            print("[error] Local 'state' directory is empty or does not exist. Nothing to push.", file=sys.stderr)
            sys.exit(1)
        
        print(f"[hf_sync] Creating private dataset repository {repo_id} if it doesn't exist...")
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        except Exception as e:
            print(f"[error] Failed to create repository: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"[hf_sync] Uploading local 'state/' contents to {repo_id}...")
        try:
            api.upload_folder(
                folder_path=str(state_path),
                repo_id=repo_id,
                repo_type="dataset",
            )
            print("[hf_sync] Push completed successfully!")
        except Exception as e:
            print(f"[error] Push failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.pull:
        print(f"[hf_sync] Pulling files from dataset {repo_id}...")
        try:
            # Safely recreate local state directory
            if state_path.exists():
                print("[hf_sync] Cleaning existing local 'state/' directory...")
                shutil.rmtree(state_path)
            state_path.mkdir(parents=True, exist_ok=True)

            api.snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(state_path),
            )
            print("[hf_sync] Pull completed successfully! Local state restored.")
        except Exception as e:
            print(f"[error] Pull failed: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
