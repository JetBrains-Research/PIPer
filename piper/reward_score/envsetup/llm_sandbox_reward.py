import asyncio
from typing import List

from piper.reward_score.envsetup.sandbox_like_reward.calc import LLMJudgeBatch
from piper.reward_score.envsetup.llm_reward import BaseLLMReward
from piper.reward_score.envsetup.utils import extract_bash_script, remove_reasoning
from piper.reward_score.reward_helper_fns import RewardOutput


class SandboxLLMReward(BaseLLMReward):
    def __init__(self, *args, **kwargs):
        self.judge = LLMJudgeBatch(
            *args,
            **kwargs
        )

    async def reward_fn_batched_async(self, data_sources: list[str], solution_strs: list[str], ground_truths: list[str], extra_infos: list[dict]):
        solution_strs = [remove_reasoning(s) for s in solution_strs]
        solution_strs = [extract_bash_script(s, mode='strict') for s in solution_strs]
        failed_mask = [s == "" for s in solution_strs]
        repo_names = [e['tools_kwargs']['repository'] for e in extra_infos]
        predictions = await self.judge.evaluate_batch_async(solution_strs, repo_names)
        res = []
        for i, prediction in enumerate(predictions):
            issues_count = prediction['issues_count']
            exit_code = prediction['exit_code']
            reasoning = prediction['reasoning']
            bad_format = failed_mask[i]
            if bad_format:
                score = -1
            else:
                score = max(1 - issues_count/100, 0) if exit_code == 0 else 0
            res.append({
                'score': score,
                'reasoning': reasoning,
                'issues_count': issues_count,
                'exit_code': exit_code,
                'failed_generation': exit_code == -999,
                'bad_format': bad_format
            })
        return res

    async def reward_fn_async(self, data_source: str, solution_str: str, ground_truth: str, extra_info: dict):
        return await self.reward_fn_batched_async([data_source], [solution_str], [ground_truth], [extra_info])[0]
    
    def extract_content_from_solution(self, solution_str: str):
        raise NotImplementedError('Not implemented')
    
    def get_analysis_prompt(self, content: str):
        raise NotImplementedError('Not implemented')
    
    def get_function_definition(self):
        raise NotImplementedError('Not implemented')

    async def messages_reward_fn_async(self, data_source: str, response: list[dict[str, str]], ground_truth: str, extra_info: dict | None = None):
        """Async version of LLM reward function for multi-turn setup that accepts raw conversation (list of messages)."""
        print(f"DEBUG: PackageManagerLLMReward.messages_reward_fn_async called for data_source: {data_source}")
        last_assistant_response = [msg for msg in response if msg["role"] == "assistant"][-1]["content"]
        return await self.reward_fn_async(data_source=data_source, solution_str=last_assistant_response, ground_truth=ground_truth, extra_info=extra_info)

    async def messages_sandbox_llm_reward_fn_batched_async(self, data_sources: list[str], responses: list[list[dict[str, str]]], ground_truths: list[str], extra_infos: list[dict]):
        solution_strs = []
        for response in responses:
            solution_strs.append([msg for msg in response if msg["role"] == "assistant"][-1]["content"])
        return await self.reward_fn_batched_async(data_sources=data_sources, solution_strs=solution_strs, ground_truths=ground_truths, extra_infos=extra_infos)

    async def feedback_sandbox_llm_reward_async(self, script: str, state: dict) -> str:
        solution_str = '```bash\n' + script + '\n```'
        res = await self.reward_fn_async(data_source='', solution_str=solution_str, ground_truth='', extra_info=state)
        return res['reasoning']
