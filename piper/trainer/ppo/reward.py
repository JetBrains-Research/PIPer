import hydra
import ray

from verl import DataProto
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.reward import get_custom_reward_fn as original_get_custom_reward_fn
from verl.trainer.ppo.reward import load_reward_manager as original_load_reward_manager


def get_custom_reward_fn(config):
    reward_fn_config = config.get("custom_reward_function")
    # If config contains a hydra config, use hydra.utils.instantiate, otherwise fallback to original_get_custom_reward_fn
    if "config" in reward_fn_config:
        reward_fn = hydra.utils.instantiate(reward_fn_config["config"])
        # Validate that reward_fn is callable
        if not callable(reward_fn):
            raise TypeError("The instantiated custom reward function is not callable.")
        return reward_fn
    else:
        return original_get_custom_reward_fn(config)



def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """Loads custom reward manager if specified in config, otherwise fallback to original verl function with predefined reward managers."""
    reward_manager_name = config.reward_model.get("reward_manager", "naive")

    if reward_manager_name == "messages_batch":
        from piper.reward_manager.messages_batch import MessagesBatchRewardManager

        reward_manager_cls = MessagesBatchRewardManager
    else:
        return original_load_reward_manager(config, tokenizer, num_examine, **reward_kwargs)

    compute_score = get_custom_reward_fn(config)
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Modified version of verl.trainer.ppo.reward.compute_reward_async that allows for custom reward managers.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)
