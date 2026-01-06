import textwrap
from config import FEW_OR_ZERO_SHOT, model_choice, USER_ROLE, AGENT_ROLE, stage_4_model_choice

def _build_messages_with_examples(FEW_OR_ZERO_SHOT, instructions, examples, user_content_builder):
    messages = [{"role": "system", "content": instructions}]

    if FEW_OR_ZERO_SHOT == "few":
        for _, example in examples.items():
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})

    messages.append({"role": "user", "content": user_content_builder()})
    return messages

def make_messages_format(FEW_OR_ZERO_SHOT, instructions, examples, conversation_history, current_turn):
    def user_content_builder():
        return f"CONVERSATION HISTORY:\n{conversation_history}\n\nTARGET TURN:\n{AGENT_ROLE}: {current_turn}"

    return _build_messages_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        user_content_builder,
    )

# ==========================
# Shared GPT-5 text builder
# ==========================

def _build_gpt5_with_examples(
    FEW_OR_ZERO_SHOT,
    instructions,
    examples,
    example_title_prefix,
    suffix_label,
    current_input_builder,
):
    instructions_text = f"INSTRUCTIONS:\n{instructions}"

    if FEW_OR_ZERO_SHOT == "few":
        few_shot_examples = ""
        count = 0
        for count, example in examples.items():
            text_example = f"*{example_title_prefix} {count}*\n\n"
            text_example += example["input"] + f"\n\n{suffix_label}:\n"
            text_example += example["output"] + "\n\n"
            few_shot_examples += text_example

        current_input = (
            f"*{example_title_prefix} {count+1}*\n\n"
            f"{current_input_builder()}\n\n"
            f"{suffix_label}:\n"
        )
        prompt = f"{instructions_text}\n\n{few_shot_examples}\n\n{current_input}"
    elif FEW_OR_ZERO_SHOT == "zero":
        prompt = (
            f"{instructions_text}\n\n"
            f"{current_input_builder()}\n\n"
            f"{suffix_label}:\n"
        )
    else:
        raise ValueError("FEW_OR_ZERO_SHOT must be 'few' or 'zero'")

    return prompt

def make_GPT5_format(FEW_OR_ZERO_SHOT, instructions, examples, conversation_history, current_turn):
    def user_content_builder():
        return (
            f"CONVERSATION HISTORY:\n{conversation_history}\n\n"
            f"TARGET TURN:\n{AGENT_ROLE}: {current_turn}"
        )

    return _build_gpt5_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        example_title_prefix="CONVERSATION",
        suffix_label="DECOMPOSITION",
        current_input_builder=user_content_builder,
    )

def _build_llama_with_examples(FEW_OR_ZERO_SHOT, instructions, examples, current_user_content_builder):
    BEGIN_OF_TEXT = "<|begin_of_text|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"
    EOT = "<|eot_id|>"

    def _header(role: str) -> str:
        return f"{START_HEADER}{role}{END_HEADER}"

    system_instructions = f"INSTRUCTIONS:\n{instructions}"

    parts = [f"{BEGIN_OF_TEXT}{_header('system')}\n\n{system_instructions}{EOT}\n"]

    if FEW_OR_ZERO_SHOT == "few":
        for _, example in examples.items():
            parts.append(f"{_header('user')}\n\n{example['input']}{EOT}\n\n")
            parts.append(f"{_header('assistant')}\n\n{example['output']}{EOT}\n\n")
    elif FEW_OR_ZERO_SHOT == "zero":
        pass
    else:
        raise ValueError("FEW_OR_ZERO_SHOT must be 'few' or 'zero'")

    parts.append(f"{_header('user')}\n\n{current_user_content_builder()}{EOT}\n\n")
    parts.append(f"{_header('assistant')}\n\n")

    return "".join(parts)

def make_llama_format(FEW_OR_ZERO_SHOT, instructions, examples, conversation_history, current_turn):
    def current_user_content_builder():
        return (
            f"CONVERSATION HISTORY:\n{conversation_history}\n\n"
            f"TARGET TURN:\n{AGENT_ROLE}: {current_turn}"
        )

    return _build_llama_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        current_user_content_builder,
    )


def make_qwen3_format(FEW_OR_ZERO_SHOT, instructions, examples, conversation_history, current_turn):
    """
    Format prompt for Qwen3 model using messages format (similar to GPT models).
    Qwen3 uses a chat template that expects messages format.
    """
    def user_content_builder():
        return f"CONVERSATION HISTORY:\n{conversation_history}\n\nTARGET TURN:\n{AGENT_ROLE}: {current_turn}"

    return _build_messages_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        user_content_builder,
    )


def make_mistral_format(FEW_OR_ZERO_SHOT, instructions, examples, conversation_history, current_turn):
    """
    Format prompt for Mistral models using messages format.
    Mistral models use a chat template that expects messages format.
    """
    def user_content_builder():
        return f"CONVERSATION HISTORY:\n{conversation_history}\n\nTARGET TURN:\n{AGENT_ROLE}: {current_turn}"

    return _build_messages_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        user_content_builder,
    )

def format_prompt(FEW_OR_ZERO_SHOT, instructions, examples, conversation_history, current_turn):
    if model_choice == "llama" or model_choice.startswith("Llama-"):
        return make_llama_format(FEW_OR_ZERO_SHOT, instructions, examples, conversation_history, current_turn)
    elif model_choice in ["gpt-5", "GPT-5"]:
        return make_GPT5_format(FEW_OR_ZERO_SHOT, instructions, examples, conversation_history, current_turn)
    elif model_choice in ["gpt-4o", "GPT-4o", "deepseek", "deepseek-reasoner", "qwen3"] or model_choice.startswith("Qwen3-"):
        return make_messages_format(FEW_OR_ZERO_SHOT, instructions, examples, conversation_history, current_turn)
    elif model_choice == "mistral" or model_choice.startswith("Mistral-") or model_choice.startswith("Mixtral-"):
        return make_mistral_format(FEW_OR_ZERO_SHOT, instructions, examples, conversation_history, current_turn)
    else:
        raise ValueError("Invalid model choice")

# ==========================
# Stage 2 prompt formatting
# ==========================

def make_stage2_messages_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    def user_content_builder():
        return (
            f"{source}\n\n"
            f"CLAIM:\n{claim}"
        )

    return _build_messages_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        user_content_builder,
    )

def make_stage2_GPT5_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    def user_content_builder():
        return (
            f"{source}\n\n"
            f"CLAIM:\n{claim}"
        )

    return _build_gpt5_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        example_title_prefix="EXAMPLE",
        suffix_label="CATEGORY",
        current_input_builder=user_content_builder,
    )

def make_stage2_llama_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    def current_user_content_builder():
        return (
            f"{source}\n\n"
            f"CLAIM:\n{claim}"
        )

    return _build_llama_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        current_user_content_builder,
    )


def make_stage2_qwen3_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    """
    Format Stage 2 prompt for Qwen3 model using messages format.
    """
    def user_content_builder():
        return (
            f"{source}\n\n"
            f"CLAIM:\n{claim}"
        )

    return _build_messages_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        user_content_builder,
    )


def make_stage2_mistral_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    """
    Format Stage 2 prompt for Mistral models using messages format.
    """
    def user_content_builder():
        return (
            f"{source}\n\n"
            f"CLAIM:\n{claim}"
        )

    return _build_messages_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        user_content_builder,
    )

def format_stage2_prompt(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    if model_choice == "llama" or model_choice.startswith("Llama-"):
        return make_stage2_llama_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source)
    elif model_choice in ["gpt-5", "GPT-5"]:
        return make_stage2_GPT5_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source)
    elif model_choice in ["gpt-4o", "GPT-4o", "deepseek", "deepseek-reasoner", "qwen3"] or model_choice.startswith("Qwen3-"):
        return make_stage2_messages_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source)
    elif model_choice == "mistral" or model_choice.startswith("Mistral-") or model_choice.startswith("Mixtral-"):
        return make_stage2_mistral_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source)
    else:
        raise ValueError("Invalid model choice")

# ==========================
# Stage 3 prompt formatting
# ==========================

def make_stage3_messages_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    def user_content_builder():
        return (
            f"{source}\n\n"
            f"CLAIM:\n{claim}"
        )

    return _build_messages_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        user_content_builder,
    )

def make_stage3_GPT5_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    def user_content_builder():
        return (
            f"{source}\n\n"
            f"CLAIM:\n{claim}"
        )

    return _build_gpt5_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        example_title_prefix="EXAMPLE",
        suffix_label="CATEGORY",
        current_input_builder=user_content_builder,
    )

def make_stage3_llama_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    def current_user_content_builder():
        return (
            f"{source}\n\n"
            f"CLAIM:\n{claim}"
        )

    return _build_llama_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        current_user_content_builder,
    )


def make_stage3_qwen3_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    """
    Format Stage 3 prompt for Qwen3 model using messages format.
    """
    def user_content_builder():
        return (
            f"{source}\n\n"
            f"CLAIM:\n{claim}"
        )

    return _build_messages_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        user_content_builder,
    )


def make_stage3_mistral_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    """
    Format Stage 3 prompt for Mistral models using messages format.
    """
    def user_content_builder():
        return (
            f"{source}\n\n"
            f"CLAIM:\n{claim}"
        )

    return _build_messages_with_examples(
        FEW_OR_ZERO_SHOT,
        instructions,
        examples,
        user_content_builder,
    )

def format_stage3_prompt(FEW_OR_ZERO_SHOT, instructions, examples, claim, source):
    if model_choice == "llama" or model_choice.startswith("Llama-"):
        return make_stage3_llama_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source)
    elif model_choice in ["gpt-5", "GPT-5"]:
        return make_stage3_GPT5_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source)
    elif model_choice in ["gpt-4o", "GPT-4o", "deepseek", "deepseek-reasoner", "qwen3"] or model_choice.startswith("Qwen3-"):
        return make_stage3_messages_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source)
    elif model_choice == "mistral" or model_choice.startswith("Mistral-") or model_choice.startswith("Mixtral-"):
        return make_stage3_mistral_format(FEW_OR_ZERO_SHOT, instructions, examples, claim, source)
    else:
        raise ValueError("Invalid model choice")


# ==========================
# Stage 4 prompt formatting
# ==========================
def make_stage4_deepseek_format(instructions: str, conversation_history: str, target_turn: str):
    """
    Format Stage 4 prompt for DeepSeek models using messages format.
    DeepSeek models use a chat template that expects messages format.
    """
    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": f"CONVERSATION HISTORY:\n{conversation_history}\n\nTARGET TURN:\n{AGENT_ROLE}: {target_turn}"
        }
    ]
    return messages

def make_stage4_GPT5_format(instructions: str, conversation_history: str, target_turn: str):
    """
    Format Stage 4 prompt for GPT-5 using text format.
    """
    instructions_text = f"INSTRUCTIONS:\n{instructions}"
    current_input = (
        f"CONVERSATION HISTORY:\n{conversation_history}\n\n"
        f"TARGET TURN:\n{AGENT_ROLE}: {target_turn}\n\n"
        f"JUDGEMENT:\n"
    )
    prompt = f"{instructions_text}\n\n{current_input}"
    return prompt

def format_stage_4_prompt(instructions: str, conversation_history: str, target_turn: str):
    if stage_4_model_choice == "deepseek" or stage_4_model_choice.startswith("deepseek-"):
        return make_stage4_deepseek_format(instructions, conversation_history, target_turn)
    elif stage_4_model_choice in ["gpt-5", "GPT-5"]:
        return make_stage4_GPT5_format(instructions, conversation_history, target_turn)
    else:
        raise ValueError(f"Invalid stage_4_model_choice: {stage_4_model_choice}. Must be 'deepseek' or 'gpt-5'")
