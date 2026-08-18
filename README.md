# VISTA: Verification in Sequential Turn-based Assessment

VISTA is a multi-stage system for analyzing conversational text, extracting claims, verifying them against reference documents, and detecting contradictions. The paper can be found on [arXiv](https://arxiv.org/abs/2510.27052). 

## Overview

VISTA processes conversations through three stages:

1. **Stage 1**: Extracts atomic factual claims from target turns
2. **Stage 2**: Verifies claims against background knowledge and reference texts
3. **Stage 3**: Categorizes unverifiable claims (CONTRADICTED, OUT-OF-SCOPE, LACKING EVIDENCE, ABSTENTION)

Claims verified as true, or judged OUT-OF-SCOPE, are accumulated as BACKGROUND KNOWLEDGE for later turns, which lets VISTA catch claims that depend on information established earlier in the conversation (including contradictions with earlier turns).

Optionally, VISTA can also fold the full raw dialogue history into the Stage 2 and Stage 3 prompts (`USE_DIALOGUE_CONTEXT` in `config.py`). This reproduces the "VISTA+ctx" setting from the contradiction-detection analysis in the paper, which substantially improves contradiction detection at the cost of a longer prompt. It's off by default, matching the setting used for the main results.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for local models like Llama, Qwen3, Mistral)
- API keys for OpenAI and/or DeepSeek (for API-based models)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ashleylew/VISTA.git
cd VISTA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash

# Set your API keys as environment variables
export OPENAI_API_KEY="your-openai-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# Gemini (Vertex AI) - see note below on authentication
export GCP_PROJECT="your-gcp-project-id"
export GCP_LOCATION="global"   # optional, defaults to "global"
```

Alternatively, create a `.env` file in the root directory:
```
OPENAI_API_KEY=your-openai-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
GCP_PROJECT=your-gcp-project-id
GCP_LOCATION=global
```

Gemini models go through Vertex AI (`google-genai` client with `vertexai=True`), not the plain Gemini Developer API key. This means you also need:
- The `gcloud` CLI, and Application Default Credentials for an account with access to `GCP_PROJECT`:
  ```bash
  gcloud auth application-default login --no-browser   # headless/remote servers
  gcloud auth application-default set-quota-project "$GCP_PROJECT"
  ```
- The Vertex AI API enabled on that project:
  ```bash
  gcloud services enable aiplatform.googleapis.com --project "$GCP_PROJECT"
  ```

Then load it using `python-dotenv` (add to requirements.txt if using this method).

4. Huggingface Settings

Login to Huggingface if you're going to be using models from there. Make sure you have permissions. See individual models' HF pages for details to get access.

`hf auth login`

Your huggingface model cache is probably set by default but if you need to specify it:
`export HF_HOME="your-HF-cache"`


5. Configure your settings in `code_base/config.py`:
   - Set `model_choice` to your preferred model
   - Toggle `USE_DIALOGUE_CONTEXT` on if you want full dialogue history added to Stage 2/3 (see Overview)
   - Configure `INPUT_FILE`, `OUTPUT_FILE`, and `FEW_OR_ZERO_SHOT`
   - Adjust `USER_ROLE` and `AGENT_ROLE` for your dataset format

## Usage

### Basic Usage

Run the main evaluation script:

```bash
cd code_base
python run_metric.py
```

The script will:
- Load conversations from the configured input file
- Process each conversation through all stages
- Save results to the configured output file

### Supported Models

**API-based models:**
- `gpt-4o` - OpenAI GPT-4o
- `gpt-5` - OpenAI GPT-5
- `deepseek` - DeepSeek Chat
- `deepseek-reasoner` - DeepSeek Reasoner
- `gemini-*` - Any Gemini model available on Vertex AI, e.g. `gemini-3.5-flash-lite` (requires `GCP_PROJECT`, see Installation)

**Local models (require GPU):**
- `llama`, `Llama-8B`, `Llama-70B`, `Llama-405B` - Meta Llama3.1 models. "llama" defaults to Llama3.1-70B.
- `qwen3`, `Qwen3-8B`, `Qwen3-14B`, `Qwen3-32B` - Qwen3 models. "qwen3" defaults to Qwen3-8B
- `mistral`, `Mistral-7B`, `Mixtral-8x7B`, `Mixtral-8x22B` - Mistral models. "mistral" defaults to Mistral-7B.

### Original paper used:
    # Llama-3.1-XB-Instruct (8B, 70B)
    # Mistral-7B-Instruct-v0.3
    # Qwen3-XB (8B, 32B)
    # GPT-5 (low reasoning setting)
    # GPT-4o
    # Deepseek-V3 (deepseek-chat)


### Configuration

Key configuration options in `config.py`:

- `model_choice`: The model to use for stages 1-3
- `USE_DIALOGUE_CONTEXT`: Add full dialogue history to the Stage 2/3 prompts (reproduces "VISTA+ctx" from the paper's contradiction-detection analysis)
- `FEW_OR_ZERO_SHOT`: Use "few" or "zero" shot prompting
- `INPUT_FILE`: path to input JSON file.
- `OUTPUT_FILE`: path to output JSON file.
- `USER_ROLE` / `AGENT_ROLE`: Role names for your dataset format



### Input Format

Input JSON files should contain conversations in the following format:

```json
[
  [
    {
      "role": "user",
      "utterance": "Hello, how are you?"
    },
    {
      "role": "assistant",
      "utterance": "I'm doing well, thank you!",
      "retrieved_document": "Reference text here..."
    }
  ]
]
```

### Output Format

The output JSON file contains the same structure with additional fields:
- `claims`: List of extracted claims with their categories
- `facts`: Accumulated facts from the conversation (labeled VERIFIED or OUT-OF-SCOPE)

## Project Structure

```
VISTA/
├── code_base/
│   ├── API_call.py          # Model API calls
│   ├── config.py            # Configuration
│   ├── prompt_templates.py  # Prompt formatting for different models
│   ├── run_metric.py        # Main execution script
│   ├── stage_1.py           # Claim extraction
│   ├── stage_2.py           # Claim verification
│   └── stage_3.py           # Unverifiable claim categorization
├── data/
│   └── reannotated_data/
│       └── human_eval_VISTA_input.json  # Released human-eval dataset
├── evaluation/
│   ├── config.py             # Which VISTA output file(s) to evaluate
│   ├── normalize.py          # Cleans raw claim categories from VISTA output
│   └── metrics.py            # Claim stats, VISTA score, optional gold comparison
└── README.md                # This file
```

## License

[Add license here]

## Citation

If you use VISTA in your research, please cite:

```bibtex
@article{lewis2025vista,
  title={VISTA Score: Verification In Sequential Turn-based Assessment},
  author={Lewis, Ashley and Perrault, Andrew and Fosler-Lussier, Eric and White, Michael},
  journal={arXiv preprint arXiv:2510.27052},
  year={2025}
}
```


### Notes

Only the AGENT_ROLE turns will be checked in VISTA score.




## Evaluation

Once you've run `run_metric.py` and have a VISTA output file (or several, from
different models), `evaluation/` turns the raw per-claim categories into a
clean summary and a VISTA score.

```bash
cd evaluation
python normalize.py   # cleans raw claim categories -> normalized/{file}_NORMALIZED.json
python metrics.py     # computes stats + VISTA score  -> summaries/{file}_SUMMARY.json
```

Configure which output file(s) to process in `evaluation/config.py`, either by
listing them explicitly (`MODEL_OUTPUT_FILES`) or pointing `OUTPUT_DIR` at a
folder to process everything in it.

`metrics.py` reports, per file:
- Total claims, average claims per turn, and a breakdown by category
- The overall **VISTA score**: the proportion of all claims that are
  VISTA-safe (`VERIFIED`, `ABSTENTION`, or `OUT-OF-SCOPE`)
- A per-turn VISTA score and faithful/hallucinated call, written to
  `summaries/{file}_SUMMARY.json`
- If the data has gold labels (a `label` or `gold_label` field per turn),
  accuracy/precision/recall/F1 against them -- skipped otherwise, since new
  data you run VISTA on won't generally have ground truth

## Reannotated Data

`data/reannotated_data/human_eval_VISTA_input.json` is the human-reannotated
evaluation set from the paper: each scored turn carries `human_claims` (the
claims a human annotator extracted, with their judgement) and a `gold_label`
for the turn as a whole. It's also the default `INPUT_FILE` in
`code_base/config.py`, so it doubles as a ready-to-use example input for the
main pipeline.


