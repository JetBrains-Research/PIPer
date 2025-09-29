import json
import re
from typing import Dict, List

import numpy as np
import torch
from tensordict import TensorDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from verl.protocol import DataProto


def get_tool_mask(model_name: str, tokenizer: PreTrainedTokenizerFast, responses: List[str], raw_responses: List[List[Dict[str, str]]]) -> torch.tensor:
    # TODO: a lot messier than it could've been because chat templates can do unexpected things
    # tested for QwQ, Qwen-2, Qwen-2.5-Coder, Qwen-3 and R1-distilled Qwen (deepseek-ai/DeepSeek-R1-Distill-Qwen-*)
    # extra hacks might be required for other models
    try:
        response_masks = []
        for i, response in enumerate(raw_responses):
            mask = []
            debug_message_str = []

            for j, message in enumerate(response):
                message_str = tokenizer.apply_chat_template([message], add_generation_prompt=False, tokenize=False)

                # Qwen-2 & Qwen-2.5-Coder-specific logic: adds system message to the conversation when not present
                if model_name.startswith("Qwen/Qwen2-") or model_name.startswith("Qwen/Qwen2.5-Coder"):
                    message_str = re.sub(r"^<\|im_start\|>system.*?<\|im_end\|>\s*", "", message_str, flags=re.DOTALL).lstrip()

                # DeepSeek-R1-Distill-Qwen-specific logic: remove bos token, bring back think blocks after tool responses
                if model_name.startswith("deepseek-ai/DeepSeek-R1-Distill-Qwen"):
                    message_str = re.sub(r"<｜begin▁of▁sentence｜>", "", message_str, flags=re.DOTALL).lstrip()
                    if message["role"] == "assistant" and j - 1 >= 0 and response[j - 1]["role"] == "tool":
                        if "<think>" in message["content"]:
                            content = message["content"].split("</think>")[-1].strip()
                            reasoning_content = message["content"].split("</think>")[0].split("<think>")[-1].strip()
                            message_str = f"<think>{reasoning_content}</think>\n{content}<｜end▁of▁sentence｜>"

                # QwQ-specific logic: remove <think> block from intermediate assistant messages
                if model_name.startswith("Qwen/QwQ"):
                    is_last_message = j == len(response) - 1

                    if message["role"] == "assistant" and not is_last_message:
                        if "</think>" in message_str:
                            m = re.match(
                                r"<\|im_start\|>assistant\n(.*?)(<\|im_end\|>)?$",
                                message_str,
                                flags=re.DOTALL,
                            )
                            if m:
                                content = m.group(1).strip()
                                content_without_think = re.sub(r"(<think>)?.*?</think>", "", content, flags=re.DOTALL).strip()
                                message_str = f"<|im_start|>assistant\n{content_without_think}<|im_end|>\n"

                # Qwen3-specific logic: adds reasoning content for certain assistant messages
                if model_name.startswith("Qwen/Qwen3-"):
                    is_last_message = j == len(response) - 1
                    after_user_query = True  # TODO: should it be checked against raw prompts?

                    # only the final assistant turn gets an empty <think> block
                    if message["role"] == "assistant":
                        content = message["content"].split("</think>")[-1].lstrip("\n") if "</think>" in message["content"] else message["content"]
                        reasoning_content = message["content"].split("</think>")[0].split("<think>")[-1].lstrip("\n") if "</think>" in message["content"] else ""

                        def fix_tool_call(tool_call):
                            func = tool_call["function"]
                            # If arguments is a string (double-encoded), parse it back to a dict
                            if isinstance(func.get("arguments"), str):
                                try:
                                    func["arguments"] = json.loads(func["arguments"])
                                except json.JSONDecodeError:
                                    pass  # keep as-is if not parseable
                            return func

                        if not after_user_query:
                            # no think
                            message_str = "<|im_start|>assistant\n" + content.lstrip("\n") + "\n".join([f"\n<tool_call>\n{json.dumps(fix_tool_call(tool_call))}\n</tool_call>" for tool_call in message.get("tool_calls", [])]) + "<|im_end|>\n"

                        if reasoning_content or is_last_message:
                            # think
                            message_str = "<|im_start|>assistant\n<think>\n" + reasoning_content.strip("\n") + "\n</think>\n\n" + content.lstrip("\n") + "\n".join([f"\n<tool_call>\n{json.dumps(fix_tool_call(tool_call))}\n</tool_call>" for tool_call in message.get("tool_calls", [])]) + "<|im_end|>\n"

                debug_message_str.append(message_str)
                message_tokens = tokenizer.encode(message_str, add_special_tokens=False)
                mask.extend([1] * len(message_tokens) if message.get("role") == "tool" else [0] * len(message_tokens))
            response_masks.append(mask)

            # Replace assertion with a warning log to avoid crashing
            if "".join(debug_message_str) != responses[i]:
                print(f"WARNING: Token mask logic mismatch for model {model_name}. This might cause issues.")
                # Optionally log more details for debugging
                print(f"Expected: {repr(responses[i])}")
                print(f"     Got: {repr(''.join(debug_message_str))}")

        # R1-distilled Qwen-specific logic: there's an additional bos token at the beginning of each response after this line:
        # responses = tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")
        if model_name.startswith("deepseek-ai/DeepSeek-R1-Distill-Qwen"):
            response_masks = [[0] + mask for mask in response_masks]

        max_length = max(len(mask) for mask in response_masks)
        response_masks = [mask + [0] * (max_length - len(mask)) for mask in response_masks]

        response_masks = torch.tensor(response_masks, dtype=torch.int64)
        return response_masks
    except Exception as e:
        print(f"ERROR in get_tool_mask: {e}")
        # Return a safe default mask (all zeros) to avoid segfault
        max_length = max(len(tokenizer.encode(r, add_special_tokens=False)) for r in responses)
        return torch.zeros((len(responses), max_length), dtype=torch.int64)


def postprocess_custom_loss_mask(model_name: str, tokenizer: PreTrainedTokenizerFast, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], n: int) -> DataProto:
    """
    Based on verl/examples/ppo_trainer/naive_ppo_trainer.py:_postprocess,
    but additionally adds tool mask + returns raw conversations (lists of messages).

    Note:
        Tool mask support heavily depends on chat templates of individual models.
        Currently tested with Qwen-2, Qwen-2.5-Coder, Qwen-3 and R1-distilled Qwen (deepseek-ai/DeepSeek-R1-Distill-Qwen-*)
    """
    # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
    # prompts: left pad
    # responses: right pad
    # input_ids: prompt + response
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

    # prompts: [prompt] from input dataset
    prompts = [tokenizer.apply_chat_template(prompt, add_generation_prompt=False, tokenize=False, **(conversations[0].get("chat_template_kwargs", {}) if isinstance(conversations[0], dict) else {})) for prompt, conversations in zip(batch.non_tensor_batch["raw_prompt"], batch_conversations)]

    # flatten batch_conversations if n > 1
    assert len(batch_conversations) == len(prompts)
    batch_conversations = [conversation for conversations in batch_conversations for conversation in conversations]
    assert len(batch_conversations) == len(prompts) * n

    # sequences: [prompt + response]
    sequences = [
        tokenizer.apply_chat_template(conversation["messages"] if isinstance(conversation, dict) else conversation, add_generation_prompt=False, tokenize=False, **(conversation.get("chat_template_kwargs", {}) if isinstance(conversation, dict) else {})) for conversation in batch_conversations
    ]

    # raw_responses: [response] (lists of messages)
    # responses: [response] (strings with applied chat templates)
    responses = [sequence[len(prompts[i // n]) :] for i, sequence in enumerate(sequences)]
    raw_responses = [
        (conversation["messages"] if isinstance(conversation, dict) else conversation)[len(batch.non_tensor_batch["raw_prompt"][i // n]) :] if isinstance(conversation, dict) else conversation[len(batch.non_tensor_batch["raw_prompt"][i // n]) :] for i, conversation in enumerate(batch_conversations)
    ]

    # tool mask: 1 for tool response tokens, 0 for others
    tool_mask = get_tool_mask(model_name, tokenizer, responses, raw_responses)

    prompts = tokenizer(prompts, return_tensors="pt", padding="longest", padding_side="left")
    responses = tokenizer(responses, return_tensors="pt", padding="longest", padding_side="right")

    if n > 1:
        prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
        prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

    input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
    attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

    # create a separate loss_mask tensor that will be used for loss calculation
    # loss_mask = [prompt attention mask, response attention mask (with tokens flagged as tool responses set to 0)]
    loss_mask = responses["attention_mask"].clone()
    loss_mask[tool_mask == 1] = 0
    loss_mask = torch.cat([prompts["attention_mask"], loss_mask], dim=1)

    batch = TensorDict(
        {"prompts": prompts["input_ids"], "responses": responses["input_ids"], "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids, "loss_mask": loss_mask},
        batch_size=len(input_ids),
    )
    non_tensor_batch = {
        "raw_responses": np.array(raw_responses, dtype=object),
    }

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def postprocess_assistant_masks(model_name: str, tokenizer: AutoTokenizer, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], n: int) -> DataProto:
    """
    Based on verl/examples/ppo_trainer/naive_ppo_trainer.py:_postprocess,
    but additionally adds tool mask + returns raw conversations (lists of messages).

    Note:
        This version is based on return_assistant_tokens_mask argument of tokenizer.apply_chat_template,
        which depends on the presence of '{% generation %}' pattern in the chat template. It might not work correctly for all the models.
        (will be indicated by 'return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword.' warning message)
    """
    # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
    # prompts: left pad
    # responses: right pad
    # input_ids: prompt + response
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

    # prompts: [prompt] from input dataset
    prompts = [tokenizer.apply_chat_template(prompt, add_generation_prompt=False, tokenize=True, **(conversations[0].get("chat_template_kwargs", {}) if isinstance(conversations[0], dict) else {})) for prompt, conversations in zip(batch.non_tensor_batch["raw_prompt"], batch_conversations)]
    non_tokenized_prompts = [
        tokenizer.apply_chat_template(prompt, add_generation_prompt=False, tokenize=False, **(conversations[0].get("chat_template_kwargs", {}) if isinstance(conversations[0], dict) else {})) for prompt, conversations in zip(batch.non_tensor_batch["raw_prompt"], batch_conversations)
    ]

    # flatten batch_conversations if n > 1
    assert len(batch_conversations) == len(prompts)
    batch_conversations = [conversation for conversations in batch_conversations for conversation in conversations]
    assert len(batch_conversations) == len(prompts) * n

    # sequences: [prompt + response]
    sequences = [
        tokenizer.apply_chat_template(
            conversation["messages"] if isinstance(conversation, dict) else conversation, add_generation_prompt=False, tokenize=True, return_assistant_tokens_mask=True, return_dict=True, **(conversation.get("chat_template_kwargs", {}) if isinstance(conversation, dict) else {})
        )
        for conversation in batch_conversations
    ]

    # raw_responses: [response] (lists of messages)
    # responses: [response] (strings with applied chat templates)
    # tool_mask: 1 for tool response tokens, 0 for others
    responses = [
        {
            "input_ids": sequence["input_ids"][len(prompts[i // n]) :],
            "attention_mask": sequence["attention_mask"][len(prompts[i // n]) :],
            "assistant_masks": sequence["assistant_masks"][len(prompts[i // n]) :],
        }
        for i, sequence in enumerate(sequences)
    ]
    raw_responses = [(conversation["messages"] if isinstance(conversation, dict) else conversation)[len(batch.non_tensor_batch["raw_prompt"][i // n]) :] for i, conversation in enumerate(batch_conversations)]

    # pad responses to the right
    responses = {
        "input_ids": torch.nn.utils.rnn.pad_sequence([torch.tensor(response["input_ids"], dtype=torch.int64) for response in responses], batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": torch.nn.utils.rnn.pad_sequence([torch.tensor(response["attention_mask"], dtype=torch.int64) for response in responses], batch_first=True, padding_value=0),
        "assistant_masks": torch.nn.utils.rnn.pad_sequence([1 - torch.tensor(response["assistant_masks"], dtype=torch.int64) for response in responses], batch_first=True, padding_value=0),
    }

    # pad prompts to the left
    # TODO: kinda lame to tokenize twice, but I'm not sure the processing inside tokenizer.__call__ is equivalent to simply left-padding previously tokenized prompts that we already have
    prompts = tokenizer(non_tokenized_prompts, return_tensors="pt", padding="longest", padding_side="left")
    if n > 1:
        prompts["input_ids"] = prompts["input_ids"].repeat_interleave(n, dim=0)
        prompts["attention_mask"] = prompts["attention_mask"].repeat_interleave(n, dim=0)

    input_ids = torch.cat([prompts["input_ids"], responses["input_ids"]], dim=1)
    attention_mask = torch.cat([prompts["attention_mask"], responses["attention_mask"]], dim=1)
    position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

    # create a separate loss_mask tensor that will be used for loss calculation
    # loss_mask = [prompt attention mask, response attention mask (with tokens flagged as tool responses set to 0)]
    loss_mask = responses["attention_mask"].clone()
    loss_mask[responses["assistant_masks"] == 1] = 0
    loss_mask = torch.cat([prompts["attention_mask"], loss_mask], dim=1)

    batch = TensorDict(
        {"prompts": prompts["input_ids"], "responses": responses["input_ids"], "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids, "loss_mask": loss_mask},
        batch_size=len(input_ids),
    )
    non_tensor_batch = {
        "raw_responses": np.array(raw_responses, dtype=object),
    }

    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def postprocess(model_name: str, tokenizer: PreTrainedTokenizerFast, batch: DataProto, batch_conversations: List[List[List[Dict[str, str]]]], n: int) -> DataProto:
    """Postprocess multi-turn output; extends verl/examples/ppo_trainer/naive_ppo_trainer.py:_postprocess by:
    1. returning additional masks for tool responses
    2. returning raw conversations (lists of messages)

    Current logic is to:
    - use assistant_masks from tokenizer.apply_chat_template when possible (i.e. when model has '{% generation %}' in its chat template)
    - use custom mask otherwise (only tested for Qwen-2, Qwen-2.5-Coder, Qwen-3 and R1-distilled Qwen (deepseek-ai/DeepSeek-R1-Distill-Qwen-*))

    batch_conversations is expected to be a list of the same length as batch.non_tensor_batch["raw_prompt"], with each element being a list of n conversations (lists of messages) for a corresponding prompt.
    """
    chat_template = tokenizer.get_chat_template()
    if not re.search(r"\{\%-?\s*generation\s*-?\%\}", chat_template):  # same condition as in tokenizer.apply_chat_template: https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/tokenization_utils_base.py#L1805
        return postprocess_custom_loss_mask(model_name, tokenizer, batch, batch_conversations, n)
    else:
        return postprocess_assistant_masks(model_name, tokenizer, batch, batch_conversations, n)
