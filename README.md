# VISTA: Verification and Inference System for Text Analysis

VISTA is a multi-stage system for analyzing conversational text, extracting claims, verifying them against reference documents, and detecting contradictions.

## Overview

VISTA processes conversations through four stages:

1. **Stage 1**: Extracts atomic factual claims from target turns
2. **Stage 2**: Verifies claims against background knowledge and reference texts
3. **Stage 3**: Categorizes unverifiable claims (CONTRADICTED, OUT-OF-SCOPE, LACKING EVIDENCE, ABSTENTION)
4. **Stage 4**: Detects contradictions within conversation history (optional)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for local models like Llama, Qwen3, Mistral)
- API keys for OpenAI and/or DeepSeek (for API-based models)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
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
```

Alternatively, create a `.env` file in the root directory:
```
OPENAI_API_KEY=your-openai-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

Then load it using `python-dotenv` (add to requirements.txt if using this method).

4. Huggingface Settings

Login to Huggingface if you're going to be using models from there. Make sure you have permissions. See individual models' HF pages for details to get access.

`hf auth login`

Your huggingface model cache is probably set by default but if you need to specify it:
`export HF_HOME="your-HF-cache"`


5. Configure your settings in `code_base/config.py`:
   - Set `model_choice` to your preferred model
   - Toggle Stage 4 activation on or off, if on, select model (needs a reasoning model)
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

**Local models (require GPU):**
- `llama`, `Llama-8B`, `Llama-70B`, `Llama-405B` - Meta Llama3.1 models. "llama" defaults to Llama3.1-70B.
- `qwen3`, `Qwen3-8B`, `Qwen3-14B`, `Qwen3-32B` - Qwen3 models. "qwen3" defaults to Qwen3-8B
- `mistral`, `Mistral-7B` - Mistral models. "mistral" defaults to Mistral-7B.

### Original paper used:
    # Llama-3.1-XB-Instruct (8B, 70B)
    # Mistral-7B-Instruct-v0.3
    # Qwen3-XB (8B, 32B)
    # GPT-5 (low reasoning setting)
    # GPT-4o
    # Deepseek-V3 (deepseek-chat)
    # Deepseek-V3.2 (deepseek-reasoner) (only for stage 4)


### Configuration

Key configuration options in `config.py`:

- `model_choice`: The model to use for stages 1-3
- `stage_4_model_choice`: The model to use for stage 4 (contradiction detection)
- `stage_4_activation`: Enable/disable stage 4
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
- `contradiction_found`: Boolean (if stage 4 is enabled)
- `contradiction_explanation`: Explanation of contradiction (if stage 4 is enabled)

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
│   ├── stage_3.py           # Unverifiable claim categorization
│   └── stage_4.py           # Contradiction detection
├── data/                    # Data directory
├── evaluation/               # Evaluation scripts/outputs
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




######### EVALUATION #########

Coming soon!


######### REANNOTATED DATA #########

Coming soon!


