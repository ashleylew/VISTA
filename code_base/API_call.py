from openai import OpenAI
import openai
import torch
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from config import DEEPSEEK_API_KEY, BASE_URL
from config import get_llama_resources, get_qwen3_resources, get_mistral_resources

# Suppress transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

def call_deepseek(messages, model_choice):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

    if model_choice == "deepseek-reasoner":
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=False
            )
    else:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False
        )

    return response.choices[0].message.content

def call_gpt4o(messages, model_choice):
    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-4o",
        messages=messages,
        stream=False
    )

    return response.choices[0].message.content

def call_gpt5(input_text, model_choice, reasoning_effort="low", text_verbosity="low"):
    """
    Call GPT-5 model with the given input text.
    
    Args:
        input_text (str): The input text to send to the model
        model_choice (str): The model choice (for consistency with other call functions)
        reasoning_effort (str): The reasoning effort level (default: "low")
        text_verbosity (str): The text verbosity level (default: "low")
    
    Returns:
        str: The model's output text
    """
    client = OpenAI()
    
    result = client.responses.create(
        model="gpt-5",
        input=input_text,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": text_verbosity},
    )
    
    return result.output_text

def call_llama(input_text, max_tokens, model_choice):
    model, tokenizer, eos_ids = get_llama_resources()

    # Tokenize and generate only up to the requested new tokens
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    # Suppress specific warnings during generation
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

    # Decode only newly generated tokens (exclude the prompt tokens)
    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_text


def call_qwen3(input_text, max_tokens, model_choice):
    """
    Call Qwen3 model with the given input text.
    
    Args:
        input_text (str): The input text to send to the model
        max_tokens (int): Maximum number of tokens to generate
    
    Returns:
        str: The model's output text
    """
    model, tokenizer = get_qwen3_resources()
    
    # Apply chat template if the input is in messages format
    if isinstance(input_text, list):
        # Input is in messages format
        text = tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Disable thinking mode for simpler output
        )
    else:
        # Input is plain text
        text = input_text
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Suppress specific warnings during generation
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    # Decode only newly generated tokens (exclude the prompt tokens)
    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return generated_text


def call_mistral(input_text, max_tokens, model_choice):
    """
    Call Mistral model with the given input text.
    
    Args:
        input_text (str): The input text to send to the model
        max_tokens (int): Maximum number of tokens to generate
    
    Returns:
        str: The model's output text
    """
    model, tokenizer = get_mistral_resources()
    
    # Apply chat template if the input is in messages format
    if isinstance(input_text, list):
        # Input is in messages format
        text = tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Input is plain text
        text = input_text
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    # Suppress specific warnings during generation
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    # Decode only newly generated tokens (exclude the prompt tokens)
    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return generated_text

def call_model(prompt, max_tokens, model_choice):
    if model_choice == "deepseek" or model_choice == "deepseek-reasoner":
        return call_deepseek(prompt, model_choice)
    elif model_choice in ["gpt-4o", "GPT-4o"]:
        return call_gpt4o(prompt, model_choice)
    elif model_choice in ["gpt-5", "GPT-5"]:
        return call_gpt5(prompt, model_choice)
    elif model_choice == "llama" or model_choice.startswith("Llama-"):
        return call_llama(prompt, max_tokens, model_choice)
    elif model_choice == "qwen3" or model_choice.startswith("Qwen3-"):
        return call_qwen3(prompt, max_tokens, model_choice)
    elif model_choice == "mistral" or model_choice.startswith("Mistral-") or model_choice.startswith("Mixtral-"):
        return call_mistral(prompt, max_tokens, model_choice)
    else:
        raise ValueError("Invalid model choice -- API_call.py")