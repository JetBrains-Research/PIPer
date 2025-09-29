"""
Script to upload a checkpoint after verl training to HuggingFace.
"""

import logging
import os
import subprocess

import boto3
import hydra
from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_merger_script(cfg: DictConfig, ckpt_dir: str):
    script = ["python3 scripts/model_merger.py"]
    script.append(f"--backend={cfg.actor_rollout_ref.actor.strategy}")
    script.append(f"--hf_model_path={cfg.actor_rollout_ref.model.path}")
    script.append(f"--local_dir={os.path.join(cfg.trainer.default_local_dir, ckpt_dir, 'actor')}")
    script.append(f"--target_dir={os.path.join(cfg.trainer.default_local_dir, ckpt_dir, 'actor_hf')}")
    return " ".join(script)


def upload_hf_checkpoint(cfg: DictConfig, ckpt_dir: str):
    api = HfApi()

    api.create_repo(
        repo_id=cfg.custom.hf.repo_id,
        repo_type="model",
        private=cfg.custom.hf.private,
        exist_ok=True,
    )

    # Upload to subdirectory named after the checkpoint step
    api.upload_folder(
        folder_path=os.path.join(cfg.trainer.default_local_dir, ckpt_dir, "actor_hf"),
        repo_id=cfg.custom.hf.repo_id,
        repo_type="model",
        path_in_repo=ckpt_dir,  # This creates the subdirectory structure
    )


def upload_s3_checkpoint(cfg: DictConfig, ckpt_dir: str):
    """Upload checkpoint to S3 bucket."""
    if not hasattr(cfg, "custom") or not hasattr(cfg.custom, "s3") or not hasattr(cfg.custom.s3, "bucket") or not hasattr(cfg.custom.s3, "dir") or not hasattr(cfg.custom.s3, "upload"):
        logger.error("cfg.custom.s3.bucket and cfg.custom.s3.dir must be set in the config to upload to S3.")
        return

    bucket_name = cfg.custom.s3.bucket
    s3_dir = cfg.custom.s3.dir

    s3 = boto3.client("s3")

    # Create the target directory path in S3 with checkpoint subdirectory
    s3_path = f"{s3_dir}/{ckpt_dir}"

    # Upload the actor_hf directory to S3
    local_path = os.path.join(cfg.trainer.default_local_dir, ckpt_dir, "actor_hf")
    logger.info(f"Uploading checkpoint from {local_path} to s3://{bucket_name}/{s3_path}")

    # Walk through all files in the directory and upload them
    for root, _, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Get the relative path from the base directory
            relative_path = os.path.relpath(local_file_path, local_path)
            # Construct the S3 key
            s3_key = f"{s3_path}/{relative_path}"

            logger.info(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
            s3.upload_file(local_file_path, bucket_name, s3_key)

    logger.info(f"Successfully uploaded checkpoint to s3://{bucket_name}/{s3_path}")


def get_checkpoint_directories(base_dir: str):
    """Get all checkpoint directories in the base directory."""
    try:
        result = subprocess.run(f"ls -d {os.path.join(base_dir, 'global_step_*')}", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Extract just the directory names (without full path)
            checkpoint_dirs = [os.path.basename(path) for path in result.stdout.strip().split("\n") if path.strip()]
            return sorted(checkpoint_dirs, key=lambda x: int(x.split("_")[-1]))  # Sort by step number
        else:
            return []
    except Exception as e:
        logger.error(f"Error getting checkpoint directories: {e}")
        return []


def process_checkpoint(cfg: DictConfig, checkpoint_dir: str):
    """Process a single checkpoint directory."""
    logger.info(f"Processing checkpoint at {cfg.trainer.default_local_dir}/{checkpoint_dir}.")

    dirs = subprocess.run(f"ls {os.path.join(cfg.trainer.default_local_dir, checkpoint_dir)}", shell=True, capture_output=True, text=True).stdout.splitlines()

    if "actor_hf" not in dirs:
        logger.info(f"No actor_hf directory found at {cfg.trainer.default_local_dir}/{checkpoint_dir}.")

        if "actor" not in dirs:
            logger.error(f"No actor directory found at {cfg.trainer.default_local_dir}/{checkpoint_dir}.")
            return False

        dirs_in_actor = subprocess.run(f"ls {os.path.join(cfg.trainer.default_local_dir, checkpoint_dir, 'actor')}", shell=True, capture_output=True, text=True).stdout.splitlines()

        if "huggingface" not in dirs_in_actor:
            model_merger_cmd = generate_merger_script(cfg, ckpt_dir=checkpoint_dir)
            logger.info(f"Running scripts/model_merger: {model_merger_cmd}")
            subprocess.run(model_merger_cmd, check=True, shell=True)
        else:
            logger.info(f"Found huggingface directory at {cfg.trainer.default_local_dir}/{checkpoint_dir}/actor/huggingface.")
            subprocess.run(f"cp -r {os.path.join(cfg.trainer.default_local_dir, checkpoint_dir, 'actor', 'huggingface')} {os.path.join(cfg.trainer.default_local_dir, checkpoint_dir, 'actor_hf')}", shell=True, check=True)

        # also save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path)
        tokenizer.save_pretrained(os.path.join(cfg.trainer.default_local_dir, checkpoint_dir, "actor_hf"))

    logger.info(f"Found actor_hf directory at {cfg.trainer.default_local_dir}/{checkpoint_dir}/actor_hf.")

    # also save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path)
    tokenizer.save_pretrained(os.path.join(cfg.trainer.default_local_dir, checkpoint_dir, "actor_hf"))

    # and training config
    with open(os.path.join(cfg.trainer.default_local_dir, checkpoint_dir, "actor_hf", "training_config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    if cfg.custom.hf.upload:
        logger.info(f"Uploading checkpoint {checkpoint_dir} to HuggingFace.")
        upload_hf_checkpoint(cfg, ckpt_dir=checkpoint_dir)

    if cfg.custom.s3.upload:
        logger.info(f"Uploading checkpoint {checkpoint_dir} to S3.")
        upload_s3_checkpoint(cfg, ckpt_dir=checkpoint_dir)

    # Delete the checkpoint directory to free up disk space (if configured)
    delete_after_upload = getattr(cfg.custom, "delete_after_upload", True)
    if delete_after_upload:
        logger.info(f"Deleting local checkpoint directory: {os.path.join(cfg.trainer.default_local_dir, checkpoint_dir)}")
        subprocess.run(f"rm -rf {os.path.join(cfg.trainer.default_local_dir, checkpoint_dir)}", shell=True, check=True)
        logger.info(f"Local checkpoint directory {checkpoint_dir} deleted successfully")
    else:
        logger.info(f"Keeping local checkpoint directory: {os.path.join(cfg.trainer.default_local_dir, checkpoint_dir)}")

    return True


@hydra.main(config_path="trainer/config/envsetup", config_name="envbench-graphs", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for experiment.
    """
    # Check if we should save all checkpoints or just the latest
    save_all_checkpoints = getattr(cfg.custom, "save_all_checkpoints", False)

    if save_all_checkpoints:
        logger.info("Configured to save all checkpoints.")
        checkpoint_dirs = get_checkpoint_directories(cfg.trainer.default_local_dir)
        if not checkpoint_dirs:
            logger.error("No checkpoint directories found.")
            return None

        logger.info(f"Found {len(checkpoint_dirs)} checkpoint(s): {checkpoint_dirs}")

        for checkpoint_dir in checkpoint_dirs:
            if not process_checkpoint(cfg, checkpoint_dir):
                logger.error(f"Failed to process checkpoint {checkpoint_dir}")
    else:
        logger.info("Configured to save only the latest checkpoint.")
        # Read the latest checkpoint iteration from the file
        latest_checkpoint_file = os.path.join(cfg.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(latest_checkpoint_file) as f:
            checkpoint_iteration = f.read().strip()

        checkpoint_dir = f"global_step_{checkpoint_iteration}"
        logger.info(f"Will consider latest checkpoint at {cfg.trainer.default_local_dir}/{checkpoint_dir}.")

        if not process_checkpoint(cfg, checkpoint_dir):
            logger.error(f"Failed to process latest checkpoint {checkpoint_dir}")
            return None

    logger.info("Checkpoint processing completed successfully")
    return None


if __name__ == "__main__":
    main()
