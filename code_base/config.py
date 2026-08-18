import os
import logging

# Suppress transformers warnings globally
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

### OPTIONS for model_choice: "GPT-4o", "GPT-5", "llama", "Llama-8B", "Llama-70B", "Llama-405B", "deepseek", "deepseek-reasoner", "gemini-*" (e.g. "gemini-3.5-flash-lite"), "qwen3", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "mistral", "Mistral-7B", "Mixtral-8x7B", "Mixtral-8x22B"

model_choice = "gemini-3.5-flash-lite"

# File paths for input and output
INPUT_FILE = "/home/lewis.2799/VISTA/data/reannotated_data/human_eval_VISTA_input.json"
OUTPUT_FILE = f"../data/VISTA_output_files/make_sure_gemini_works_{model_choice}.json"

# --- Gemini / Vertex AI ---
# Gemini models go through Vertex AI (google-genai client with vertexai=True), not a
# plain Gemini Developer API key. Set these in your environment (see README.md).
GCP_PROJECT = os.getenv("GCP_PROJECT", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "global")

# Add full dialogue history to the Stage 2 (verification) and Stage 3 (classification)
# prompts, on top of the accumulated-claims BACKGROUND KNOWLEDGE. This reproduces the
# "VISTA+ctx" setting from the contradiction-detection analysis in the paper (Section 7.2),
# which substantially improves contradiction detection. The default VISTA pipeline used
# for the main results leaves this off.
USE_DIALOGUE_CONTEXT = False

# API Keys - Load from environment variables for security
# Set these in your environment or .env file (see README.md)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Validate that API keys are set if using API-based models
if model_choice in ["gpt-4o", "gpt-5", "GPT-4o", "GPT-5", "deepseek", "deepseek-reasoner"]:
    if model_choice in ["gpt-4o", "gpt-5", "GPT-4o", "GPT-5"] and not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
    if model_choice in ["deepseek", "deepseek-reasoner"] and not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY environment variable is required for DeepSeek models")

if model_choice.startswith("gemini") and not GCP_PROJECT:
    raise ValueError("GCP_PROJECT environment variable is required for Gemini/Vertex models")

# Few shot or zero shot setting
### OPTIONS: "few" or "zero"
FEW_OR_ZERO_SHOT = "few"

### USER_ROLE and AGENT_ROLE.
USER_ROLE = 'user'
AGENT_ROLE = 'assistant'


### Base facts

NUM_BASE_FACTS = 0
BASE_FACTS = ""


# ===========================
# Optional HF lazy loaders
# ===========================
_LLAMA_MODEL = None
_LLAMA_TOKENIZER = None
_LLAMA_EOS_IDS = None

# Qwen3 cached resources
_QWEN3_MODEL = None
_QWEN3_TOKENIZER = None

# Mistral cached resources
_MISTRAL_MODEL = None
_MISTRAL_TOKENIZER = None

def get_llama_resources():
    """
    Lazily load and cache Llama model/tokenizer for any Llama variant.
    Returns a tuple: (model, tokenizer, eos_ids)
    """
    global _LLAMA_MODEL, _LLAMA_TOKENIZER, _LLAMA_EOS_IDS

    # Check if model_choice is any Llama variant
    if not (model_choice == "llama" or model_choice.startswith("Llama-")):
        raise ValueError("Llama resources requested but model_choice is not a Llama variant")

    if _LLAMA_MODEL is None or _LLAMA_TOKENIZER is None:
        # Import heavy deps lazily to avoid overhead for other models
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

        # Map model choice to actual model ID
        if model_choice == "llama":
            model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"  # Default to 70B
        elif model_choice == "Llama-8B":
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif model_choice == "Llama-70B":
            model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        elif model_choice == "Llama-405B":
            model_id = "meta-llama/Meta-Llama-3.1-405B-Instruct"
        else:
            raise ValueError(f"Unsupported Llama model size: {model_choice}")

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        _LLAMA_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )

        _LLAMA_TOKENIZER = AutoTokenizer.from_pretrained(model_id)

        eos_token_id = _LLAMA_TOKENIZER.eos_token_id
        eot_id = _LLAMA_TOKENIZER.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in _LLAMA_TOKENIZER.get_vocab() else None
        _LLAMA_EOS_IDS = [tok_id for tok_id in [eos_token_id, eot_id] if tok_id is not None]

    return _LLAMA_MODEL, _LLAMA_TOKENIZER, _LLAMA_EOS_IDS


def get_qwen3_resources():
    """
    Lazily load and cache Qwen3 model/tokenizer for any Qwen3 variant.
    Returns a tuple: (model, tokenizer)
    """
    global _QWEN3_MODEL, _QWEN3_TOKENIZER

    # Check if model_choice is any Qwen3 variant
    if not (model_choice == "qwen3" or model_choice.startswith("Qwen3-")):
        raise ValueError("Qwen3 resources requested but model_choice is not a Qwen3 variant")

    if _QWEN3_MODEL is None or _QWEN3_TOKENIZER is None:
        # Import heavy deps lazily to avoid overhead for other models
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Map model choice to actual model ID
        if model_choice == "qwen3":
            model_id = "Qwen/Qwen3-8B"  # Default to 8B
        elif model_choice == "Qwen3-8B":
            model_id = "Qwen/Qwen3-8B"
        elif model_choice == "Qwen3-14B":
            model_id = "Qwen/Qwen3-14B"
        elif model_choice == "Qwen3-32B":
            model_id = "Qwen/Qwen3-32B"
        else:
            raise ValueError(f"Unsupported Qwen3 model size: {model_choice}")

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        _QWEN3_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        _QWEN3_TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    return _QWEN3_MODEL, _QWEN3_TOKENIZER


def get_mistral_resources():
    """
    Lazily load and cache Mistral model/tokenizer for any Mistral variant.
    Returns a tuple: (model, tokenizer)
    """
    global _MISTRAL_MODEL, _MISTRAL_TOKENIZER

    # Check if model_choice is any Mistral variant
    if not (model_choice == "mistral" or model_choice.startswith("Mistral-") or model_choice.startswith("Mixtral-")):
        raise ValueError("Mistral resources requested but model_choice is not a Mistral variant")

    if _MISTRAL_MODEL is None or _MISTRAL_TOKENIZER is None:
        # Import heavy deps lazily to avoid overhead for other models
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Map model choice to actual model ID
        if model_choice == "mistral":
            model_id = "mistralai/Mistral-7B-Instruct-v0.3"  # Default to 7B
        elif model_choice == "Mistral-7B":
            model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        elif model_choice == "Mixtral-8x7B":
            model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        elif model_choice == "Mixtral-8x22B":
            model_id = "mistralai/Mixtral-8x22B-Instruct-v0.3"
        else:
            raise ValueError(f"Unsupported Mistral model size: {model_choice}")

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        _MISTRAL_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        _MISTRAL_TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    return _MISTRAL_MODEL, _MISTRAL_TOKENIZER
