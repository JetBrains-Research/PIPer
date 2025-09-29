"""
Custom core algorithms for multi-turn scenarios with tool masking.
"""

import torch

import verl.utils.torch_functional as verl_F


def compute_reinforce_plus_plus_outcome_advantage_with_tool_masking(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: float):
    """
    Compute advantage for REINFORCE++ with proper handling of tool token masking.

    This implementation differs from the standard REINFORCE++ in that it only resets
    the running return for actual padding tokens (end of sequence), not for tool tokens
    in the middle of the sequence. This preserves credit assignment across tool tokens.

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for assistant tokens, 0 for tool tokens and padding
        gamma: `(float)`
            discount factor

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        bs, seq_len = token_level_rewards.shape

        # gamma where assistant, 1.0 where tool/pad
        gamma_masked = torch.where(response_mask.bool(), gamma, 1.0)

        returns = torch.zeros_like(token_level_rewards)
        running_return = torch.zeros(bs, device=token_level_rewards.device)

        for t in reversed(range(seq_len)):
            running_return = token_level_rewards[:, t] + gamma_masked[:, t] * running_return
            returns[:, t] = running_return

        # Masked whitening of returns → advantages
        if response_mask.sum() > 0:
            advantages = verl_F.masked_whiten(returns, response_mask)
            advantages = advantages * response_mask
            returns = returns * response_mask
        else:
            advantages = torch.zeros_like(returns)

    return advantages, returns
