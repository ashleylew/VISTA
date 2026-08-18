from openai import OpenAI
import openai
import torch
import os
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from config import DEEPSEEK_API_KEY, BASE_URL
from config import get_llama_resources, get_qwen3_resources, get_mistral_resources

# Suppress transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

_GEMINI_CLIENT = None

def get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        from google import genai
        from config import GCP_PROJECT, GCP_LOCATION
        _GEMINI_CLIENT = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    return _GEMINI_CLIENT


def call_gemini(messages, max_tokens, model_choice):
    """
    Call a Gemini model via Vertex AI. Converts the shared messages format
    (list of {"role", "content"} dicts) into Gemini's Content objects, folding
    any "system" message into a system_instruction.
    """
    from google.genai import types
    client = get_gemini_client()

    system_txt, contents = "", []
    for m in messages:
        if m["role"] == "system":
            system_txt += m["content"] + "\n"
        else:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

    cfg = types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=0)
    if system_txt:
        cfg.system_instruction = system_txt.strip()

    resp = None
    for attempt in range(6):
        try:
            resp = client.models.generate_content(model=model_choice, contents=contents, config=cfg)
            break
        except Exception as e:
            if "429" in str(e) and attempt < 5:
                wait = min(2 ** attempt, 30)
                logging.warning("Gemini 429 rate-limit (attempt %d/6); retrying in %ds", attempt + 1, wait)
                time.sleep(wait)
                continue
            raise

    if not resp.candidates or not resp.candidates[0].content:
        return ""
    parts = resp.candidates[0].content.parts or []
    return "".join(p.text for p in parts if getattr(p, "text", None)).strip()


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
    elif model_choice.startswith("gemini"):
        return call_gemini(prompt, max_tokens, model_choice)
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