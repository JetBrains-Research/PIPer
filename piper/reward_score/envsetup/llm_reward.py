import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from textwrap import dedent
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from piper.reward_score.envsetup.utils import extract_bash_script
from piper.reward_score.reward_helper_fns import RewardOutput

# Load environment variables from .env file
load_dotenv(override=True)

logger = logging.getLogger(__name__)


# Initialize OpenAI client with error handling
def _get_openai_client():
    """Initialize OpenAI client with proper error handling."""
    try:
        from openai import OpenAI

        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set")
            return None

        return OpenAI()
    except ImportError:
        logger.error("OpenAI package not installed. Install with: pip install openai")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None


def _get_async_openai_client():
    """Initialize async OpenAI client with proper error handling."""
    try:
        from openai import AsyncOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set")
            return None

        return AsyncOpenAI()
    except ImportError:
        logger.error("OpenAI package not installed. Install with: pip install openai")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize async OpenAI client: {e}")
        return None


client = _get_openai_client()
async_client = _get_async_openai_client()


class BaseLLMReward(ABC):
    """Abstract base class for LLM-based reward functions."""

    def __init__(self):
        self.client = client
        self.async_client = async_client

    @abstractmethod
    def get_function_definition(self) -> Dict[str, Any]:
        """Return the OpenAI function definition for this reward type."""
        pass

    @abstractmethod
    def get_analysis_prompt(self, content: str) -> str:
        """Return the analysis prompt for the given content."""
        pass

    @abstractmethod
    def extract_content_from_solution(self, solution_str: str) -> Optional[str]:
        """Extract the relevant content from the solution string."""
        pass

    def llm_analysis_to_reward(self, analysis_result: Optional[Dict[str, Any]]) -> float:
        """
        Convert LLM analysis results to a reward score from 0-1.

        Args:
            analysis_result: Dictionary with analysis results from LLM.

        Returns:
            float: Reward score between 0 and 1.
        """
        if analysis_result is None:
            return 0.0

        try:
            # Get the LLM's quality score (0-100) - assuming all implementations use this key
            score = analysis_result.get("environment_setup_quality", 0)

            # Ensure score is within valid range and convert to 0-1
            normalized_score = max(0, min(100, score)) / 100.0
            return normalized_score

        except Exception as e:
            logger.error(f"Error converting analysis to reward: {e}")
            return 0.0

    def analyze_with_llm(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Analyze content using OpenAI function calling.

        Args:
            content: Content to analyze.

        Returns:
            Dictionary with analysis results or None if an error occurred.
        """
        if self.client is None:
            logger.error("OpenAI client not initialized")
            return None

        function_def = self.get_function_definition()
        prompt = self.get_analysis_prompt(content)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for better function calling
                messages=[{"role": "system", "content": "You are an expert analyzer. Analyze the provided content thoroughly and return structured feedback using the provided function."}, {"role": "user", "content": prompt}],
                tools=[{"type": "function", "function": function_def}],
                tool_choice={"type": "function", "function": {"name": function_def["name"]}},
                temperature=0.0,  # Deterministic analysis
            )

            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function.name == function_def["name"]:
                    return json.loads(tool_call.function.arguments)

            logger.warning("No tool call returned from OpenAI")
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error analyzing content with LLM: {e}")
            return None

    async def analyze_with_llm_async(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Analyze content using OpenAI function calling asynchronously.

        Args:
            content: Content to analyze.

        Returns:
            Dictionary with analysis results or None if an error occurred.
        """
        if self.async_client is None:
            logger.error("Async OpenAI client not initialized")
            return None

        function_def = self.get_function_definition()
        prompt = self.get_analysis_prompt(content)

        try:
            response = await self.async_client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o for better function calling
                messages=[{"role": "system", "content": "You are an expert analyzer. Analyze the provided content thoroughly and return structured feedback using the provided function."}, {"role": "user", "content": prompt}],
                tools=[{"type": "function", "function": function_def}],
                tool_choice={"type": "function", "function": {"name": function_def["name"]}},
                temperature=0.0,  # Deterministic analysis
            )

            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function.name == function_def["name"]:
                    return json.loads(tool_call.function.arguments)

            logger.warning("No tool call returned from OpenAI")
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error analyzing content with LLM: {e}")
            return None

    def reward_fn(self, data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> RewardOutput:
        """
        Main reward function that evaluates solutions using LLM analysis.

        Args:
            data_source (str): Source of the data.
            solution_str (str): Solution string containing the content to analyze.
            ground_truth (str): Ground truth string.
            extra_info (Optional[dict], optional): Additional information. Defaults to None.

        Returns:
            RewardOutput: Dictionary containing reasoning (string) and score (0-1 float).
        """
        content = self.extract_content_from_solution(solution_str)

        if not content:
            return {"reasoning": "No relevant content found in the solution", "score": 0.0}

        analysis_result = self.analyze_with_llm(content)

        if analysis_result is None:
            return {"reasoning": "LLM analysis failed - unable to analyze the content", "score": 0.0}

        score = self.llm_analysis_to_reward(analysis_result)
        reasoning = analysis_result.get("reasoning", "Analysis completed but no reasoning provided")

        return {"reasoning": reasoning, "score": score}

    async def reward_fn_async(self, data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> RewardOutput:
        """
        Async version of the main reward function that evaluates solutions using LLM analysis.

        Args:
            data_source (str): Source of the data.
            solution_str (str): Solution string containing the content to analyze.
            ground_truth (str): Ground truth string.
            extra_info (Optional[dict], optional): Additional information. Defaults to None.

        Returns:
            RewardOutput: Dictionary containing reasoning (string) and score (0-1 float).
        """
        content = self.extract_content_from_solution(solution_str)

        if not content:
            return {"reasoning": "No relevant content found in the solution", "score": 0.0}

        analysis_result = await self.analyze_with_llm_async(content)

        if analysis_result is None:
            return {"reasoning": "LLM analysis failed - unable to analyze the content", "score": 0.0}

        score = self.llm_analysis_to_reward(analysis_result)
        reasoning = analysis_result.get("reasoning", "Analysis completed but no reasoning provided")

        return {"reasoning": reasoning, "score": score}

    def reward_fn_batched(self, data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[dict]) -> List[RewardOutput]:
        """
        Batch version of reward_fn.

        Note: This processes each script individually. For better performance,
        you could modify this to batch multiple scripts in a single OpenAI call,
        but this would require more complex prompt engineering and response parsing.
        """
        return [self.reward_fn(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info) for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos)]

    async def reward_fn_batched_async(self, data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[dict]) -> List[RewardOutput]:
        """
        Async batch version of reward_fn that processes multiple items concurrently.

        This version uses asyncio.gather to process all items concurrently,
        which can significantly improve performance when dealing with multiple API calls.
        """
        tasks = [self.reward_fn_async(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info) for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos)]
        return await asyncio.gather(*tasks)

    def create_reward_fn_batched_async_limited(self, max_concurrent: int = 10):
        """
        Factory method that creates a batched async function with a specific concurrency limit.

        Args:
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            Async function with baked-in semaphore for the specified concurrency limit
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def reward_fn_batched_async_limited(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[dict]) -> List[RewardOutput]:
            """
            Async batch version with concurrency limiting and better error handling.
            This function has a baked-in semaphore with the specified concurrency limit.

            Args:
                data_sources: List of data sources
                solution_strs: List of solution strings
                ground_truths: List of ground truth strings
                extra_infos: List of extra info dictionaries

            Returns:
                List of RewardOutput results in the same order as input
            """

            async def _process_with_semaphore(data_source: str, solution_str: str, ground_truth: str, extra_info: dict, index: int):
                """Process a single item with semaphore limiting and error handling."""
                async with semaphore:
                    try:
                        result = await self.reward_fn_async(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info)
                        return (index, result)
                    except Exception as e:
                        logger.error(f"Error processing item {index}: {e}")
                        return (index, {"reasoning": f"Error during processing: {str(e)}", "score": 0.0})

            # Create tasks with explicit indexing to maintain order
            tasks = [asyncio.create_task(_process_with_semaphore(data_source, solution_str, ground_truth, extra_info, i)) for i, (data_source, solution_str, ground_truth, extra_info) in enumerate(zip(data_sources, solution_strs, ground_truths, extra_infos))]

            # Wait for all tasks and maintain original order
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Sort by index and extract results
            ordered_results = [None] * len(data_sources)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                    continue
                index, reward_output = result
                ordered_results[index] = reward_output

            # Fill any None values with error results
            for i, result in enumerate(ordered_results):
                if result is None:
                    ordered_results[i] = {"reasoning": "Processing failed", "score": 0.0}

            return ordered_results

        return reward_fn_batched_async_limited

    def messages_reward_fn(self, data_source: str, response: List[Dict[str, str]], ground_truth: str, extra_info: Optional[dict] = None) -> RewardOutput:
        """LLM reward function for multi-turn setup that accepts raw conversation (list of messages)."""
        last_assistant_response = [msg for msg in response if msg["role"] == "assistant"][-1]["content"]
        return self.reward_fn(data_source=data_source, solution_str=last_assistant_response, ground_truth=ground_truth, extra_info=extra_info)

    async def messages_reward_fn_async(self, data_source: str, response: List[Dict[str, str]], ground_truth: str, extra_info: Optional[dict] = None) -> RewardOutput:
        """Async version of LLM reward function for multi-turn setup that accepts raw conversation (list of messages)."""
        last_assistant_response = [msg for msg in response if msg["role"] == "assistant"][-1]["content"]
        return await self.reward_fn_async(data_source=data_source, solution_str=last_assistant_response, ground_truth=ground_truth, extra_info=extra_info)

    def messages_reward_fn_batched(self, data_sources: List[str], responses: List[List[Dict[str, str]]], ground_truths: List[str], extra_infos: List[dict]) -> List[RewardOutput]:
        """Batched version of messages_reward_fn."""
        return [self.messages_reward_fn(data_source=data_source, response=response, ground_truth=ground_truth, extra_info=extra_info) for data_source, response, ground_truth, extra_info in zip(data_sources, responses, ground_truths, extra_infos)]

    async def messages_reward_fn_batched_async(self, data_sources: List[str], responses: List[List[Dict[str, str]]], ground_truths: List[str], extra_infos: List[dict]) -> List[RewardOutput]:
        """Async batched version of messages_reward_fn that processes multiple conversations concurrently."""
        tasks = [self.messages_reward_fn_async(data_source=data_source, response=response, ground_truth=ground_truth, extra_info=extra_info) for data_source, response, ground_truth, extra_info in zip(data_sources, responses, ground_truths, extra_infos)]
        return await asyncio.gather(*tasks)

    def create_messages_reward_fn_batched_async_limited(self, max_concurrent: int = 10):
        """
        Factory method that creates a batched async function for messages with a specific concurrency limit.

        Args:
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            Async function with baked-in semaphore for the specified concurrency limit
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def messages_reward_fn_batched_async_limited(data_sources: List[str], responses: List[List[Dict[str, str]]], ground_truths: List[str], extra_infos: List[dict]) -> List[RewardOutput]:
            """
            Async batch version for messages with concurrency limiting and better error handling.
            This function has a baked-in semaphore with the specified concurrency limit.

            Args:
                data_sources: List of data sources
                responses: List of conversation responses (list of messages)
                ground_truths: List of ground truth strings
                extra_infos: List of extra info dictionaries

            Returns:
                List of RewardOutput results in the same order as input
            """

            async def _process_with_semaphore(data_source: str, response: List[Dict[str, str]], ground_truth: str, extra_info: dict, index: int):
                """Process a single conversation with semaphore limiting and error handling."""
                async with semaphore:
                    try:
                        result = await self.messages_reward_fn_async(data_source=data_source, response=response, ground_truth=ground_truth, extra_info=extra_info)
                        return (index, result)
                    except Exception as e:
                        logger.error(f"Error processing conversation {index}: {e}")
                        return (index, {"reasoning": f"Error during processing: {str(e)}", "score": 0.0})

            # Create tasks with explicit indexing
            tasks = [asyncio.create_task(_process_with_semaphore(data_source, response, ground_truth, extra_info, i)) for i, (data_source, response, ground_truth, extra_info) in enumerate(zip(data_sources, responses, ground_truths, extra_infos))]

            # Wait for all tasks and maintain original order
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Sort by index and extract results
            ordered_results = [None] * len(data_sources)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                    continue
                index, reward_output = result
                ordered_results[index] = reward_output

            # Fill any None values with error results
            for i, result in enumerate(ordered_results):
                if result is None:
                    ordered_results[i] = {"reasoning": "Processing failed", "score": 0.0}

            return ordered_results

        return messages_reward_fn_batched_async_limited


class BashScriptGeneralLLMReward(BaseLLMReward):
    """Concrete implementation for bash script analysis."""

    def get_function_definition(self) -> Dict[str, Any]:
        """Define the OpenAI function schema for bash script analysis."""
        return {
            "name": "analyze_bash_script",
            "description": "Analyze a bash script for environment setup quality, syntax errors, best practices, and potential issues",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Detailed explanation of the analysis, including what makes this script good or bad for environment setup, specific issues found, and overall assessment"},
                    "environment_setup_quality": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Quality score from 0 (will definitely fail environment setup) to 100 (excellent environment setup script)"},
                },
                "required": ["reasoning", "environment_setup_quality"],
                "additionalProperties": False,
            },
        }

    def get_analysis_prompt(self, script: str) -> str:
        """Return the analysis prompt for bash script evaluation."""
        return dedent(
            """
            Analyze the following bash script specifically for ENVIRONMENT SETUP quality and correctness.

            Focus on evaluating:
            1. Will this script successfully set up an environment?
            2. Are there syntax errors that would prevent execution?
            3. Are there security vulnerabilities that could compromise the setup?
            4. Does it follow best practices for environment setup scripts?
            5. Does it properly handle errors that could occur during script execution (e.g., using error checking, set -e, trap statements, validation of commands)?
            6. Will it work reliably across different systems?

            Score from 0-100 where:
            - 100: Excellent environment setup script that will definitely work
            - 80-99: Good script with minor issues
            - 60-79: Decent script but has some problems
            - 40-59: Poor script with significant issues
            - 20-39: Bad script that will likely fail
            - 0-19: Terrible script that will definitely fail

            Provide detailed reasoning explaining your analysis and score.

            Bash script to analyze:
            ```bash
            {script}
            ```
            """
        ).format(script=script)

    def extract_content_from_solution(self, solution_str: str) -> Optional[str]:
        """Extract bash script from the solution string."""
        return extract_bash_script(solution_str)


if __name__ == "__main__":
    reward = BashScriptGeneralLLMReward()
    print(
        reward.reward_fn(
            data_source="test",
            solution_str="""
    ```bash
    #!/bin/bash
    # This is a test script
    echo "Hello, world!"
    ```
    """,
            ground_truth="",
        )
    )

    async def test_async():
        reward = BashScriptGeneralLLMReward()
        result = await reward.reward_fn_async(
            data_source="test",
            solution_str="""
        ```bash
        #!/bin/bash
        # This is a test script
        echo "Hello, world!"
        ```
        """,
            ground_truth="",
        )
        print("Async result:", result)

        # Example of batched async usage
        batch_results = await reward.reward_fn_batched_async(
            data_sources=["test1", "test2"],
            solution_strs=[
                """
                ```bash
                #!/bin/bash
                echo "Script 1"
                ```
                """,
                """
                ```bash
                #!/bin/bash
                echo "Script 2"
                ```
                """,
            ],
            ground_truths=["", ""],
            extra_infos=[{}, {}],
        )
        print("Batched async results:", batch_results)

    asyncio.run(test_async())

    async def test_limited_async():
        reward = BashScriptGeneralLLMReward()

        # Create a batch processor with concurrency limiting
        batch_processor = reward.create_reward_fn_batched_async_limited(max_concurrent=2)

        # Enhanced batch processing with concurrency limiting
        enhanced_results = await batch_processor(
            data_sources=["test1", "test2", "test3"],
            solution_strs=[
                """
                ```bash
                #!/bin/bash
                echo "Script 1"
                set -e  # Good practice
                ```
                """,
                """
                ```bash
                #!/bin/bash
                echo "Script 2"
                # This has some issues
                rm -rf /  # This would be flagged as dangerous
                ```
                """,
                """
                ```bash
                #!/bin/bash
                echo "Script 3"
                sudo apt-get update
                sudo apt-get install -y python3
                ```
                """,
            ],
            ground_truths=["", "", ""],
            extra_infos=[{}, {}, {}],
        )
        print("Limited async results:", enhanced_results)

    asyncio.run(test_limited_async())
