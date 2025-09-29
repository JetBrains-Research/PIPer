# Copyright 2025 Individual Contributor: Mert Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto


class MessagesBatchRewardManager:
    """Reward manager based on verl/workers/reward_manager/batch.py, but passes raw responses (lists of messages) to underlying reward function."""

    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data):
        responses = data.non_tensor_batch["raw_responses"]
        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))

        scores = self.compute_score(
            data_sources=data_sources,
            responses=responses,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # Get response length from responses tensor
        response_length = data.batch["responses"].shape[-1]

        # Get prompt info for device and length
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        device = prompt_ids.device

        # Use loss_mask when available to find last valid token; fallback to attention_mask otherwise
        if "loss_mask" in data.batch:
            mask = data.batch["loss_mask"][:, -response_length:]
        else:
            mask = data.batch["attention_mask"][:, prompt_len:]
        positions = torch.arange(response_length, device=device)

        masked_positions = torch.where(mask.bool(), positions, -1)
        last_assistant_positions = torch.max(masked_positions, dim=-1)[0]
        last_assistant_positions = torch.clamp(last_assistant_positions, min=0)
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        scores = self.verify(data)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            last_position = last_assistant_positions[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, last_position] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine and self.tokenizer is not None:
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                response = data.non_tensor_batch["raw_responses"][i]
                
                print(f"[response] (type={type(response)})")
                response = str(response)
                print(response)
                if len(response) > 4000:
                    print('The response is too long, here are the last 1000 characters:')
                    print(response[-1000:])
                print()
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=device)
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
