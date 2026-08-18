import glob
import os

# Each input file must be raw output from code_base/run_metric.py: a list of
# conversations, each a list of turns, where scored assistant turns carry a
# 'claims' field (list of [claim_text, raw_category] pairs) and, optionally,
# a gold label field (see GOLD_LABEL_FIELD below).

# --- How to find model output files: set ONE of the two options below ---

# Option A: explicit mapping from model name -> output file path. Use this
# when filenames don't already match a model name, or you only want to
# normalize a subset of what's in a directory. Takes priority over Option B
# if both are set.
MODEL_OUTPUT_FILES = {
}

# Option B: scan a directory instead. Every *.json file in OUTPUT_DIR is
# treated as one model's output, named after its filename stem (e.g.
# "gpt-4o.json" -> model name "gpt-4o"). To use this, set MODEL_OUTPUT_FILES
# above to {} and set OUTPUT_DIR below.
OUTPUT_DIR = "../data/VISTA_output_files/"  # e.g. "../data/VISTA_output_files"


def get_model_output_files():
    """Resolve the {model_name: path} mapping from whichever option is set above."""
    if MODEL_OUTPUT_FILES:
        return MODEL_OUTPUT_FILES
    if OUTPUT_DIR:
        paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json")))
        if not paths:
            raise ValueError(f"No .json files found in OUTPUT_DIR: {OUTPUT_DIR}")
        return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}
    raise ValueError("Set either MODEL_OUTPUT_FILES or OUTPUT_DIR in config.py")


# Field on each scored turn holding the ground-truth label, if present.
# Standard dataset input files use "label"; the reannotated human-eval file
# uses "gold_label"; new/unlabeled data may have neither, which is fine.
GOLD_LABEL_FIELD = "gold_label"

NORMALIZED_DIR = "normalized"


def normalized_output_path(input_path):
    """Where the normalized version of a raw VISTA output file goes.

    Named after the input file itself (not the config key it's listed
    under), so it stays traceable back to its source regardless of what
    model name it was given in MODEL_OUTPUT_FILES.
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(NORMALIZED_DIR, f"{stem}_NORMALIZED.json")


SUMMARIES_DIR = "summaries"


def summary_output_path(input_path):
    """Where the claim/VISTA-score summary for a raw VISTA output file goes."""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(SUMMARIES_DIR, f"{stem}_SUMMARY.json")
