"""
Entrypoint script for Ray jobs in the verl pipeline.
"""

import logging
import os
import subprocess
import tempfile

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from piper.config.config_resolvers import register_all_resolvers
from piper.upload_hf_checkpoint import main as upload_hf_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_entrypoint_script(cfg: DictConfig) -> str:
    """
    Generate the command to run the main_ppo.py script using a temporary config file.

    Args:
        cfg: The Hydra configuration object

    Returns:
        A shell command string to execute main_ppo.py with a temporary config file
    """
    # Start with the base command
    cmd = "python3 -m verl.trainer.main_ppo"
    if "custom" in cfg and "command" in cfg.custom and cfg.custom.command:
        cmd = cfg.custom.command

    # Convert to regular dict and remove the 'custom' section
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    if "custom" in config_dict:
        del config_dict["custom"]

    # Write the resolved config to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        yaml.dump(config_dict, temp_file, default_flow_style=False)
        temp_config_path = temp_file.name

    # Use the temporary config file instead of CLI overrides
    cmd += f" --config-path={os.path.dirname(temp_config_path)} --config-name={os.path.basename(temp_config_path)}"

    return cmd


@hydra.main(config_path="config", config_name="envbench-graphs", version_base=None)
def main(cfg: DictConfig) -> float:
    """
    Main entry point for experiment.
    """

    # Register custom resolvers before processing configuration
    register_all_resolvers()

    # Log the configuration
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # we only run data preparation if cfg.custom.run_data_preprocess=True
    if "custom" in cfg and "run_data_preprocess" in cfg.custom and cfg.custom.run_data_preprocess:
        data_preparation_cmd = f"python3 -m piper.data_preprocess.{cfg.custom.data.file} --local_dir /home/user/data/{cfg.custom.data.file}"
        if "config" in cfg.custom.data and cfg.custom.data.config:
            data_preparation_cmd += f" --config {cfg.custom.data.config}"
        logger.info(f"Running data preparation script: {data_preparation_cmd}")
        try:
            subprocess.run(data_preparation_cmd, check=True, shell=True, capture_output=False, text=True, bufsize=1)
        except subprocess.CalledProcessError as e:
            logger.error(f"Data preparation failed with return code {e.returncode}")
            logger.error(f"Command: {data_preparation_cmd}")
            raise

    # we only skip training if cfg.custom.run_training=False
    if not ("custom" in cfg and "run_training" in cfg.custom and not cfg.custom.run_training):
        # Build the command for the training script using temp config file
        training_cmd = generate_entrypoint_script(cfg)

        # Run the training script
        logger.info(f"Running training script: {training_cmd}")

        try:
            # Use Popen for real-time output streaming
            with subprocess.Popen(training_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True) as process:
                # Stream output in real-time
                for line in iter(process.stdout.readline, ""):
                    print(line, end="")

                # Wait for process to complete and get return code
                return_code = process.wait()

                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, training_cmd)

        except subprocess.CalledProcessError as e:
            logger.error(f"Training script failed with return code {e.returncode}")
            logger.error(f"Command: {training_cmd}")
            raise

    # we only upload to HF/S3 if cfg.custom.hf.upload=True or cfg.custom.s3.upload=True
    if "custom" in cfg and "hf" in cfg.custom and cfg.custom.hf.upload or "custom" in cfg and "s3" in cfg.custom and cfg.custom.s3.upload:
        upload_hf_checkpoint(cfg)

    return 0.0  # returns a scalar for sweeping. todo: get loss (?) from wandb run using wandb api


if __name__ == "__main__":
    main()
