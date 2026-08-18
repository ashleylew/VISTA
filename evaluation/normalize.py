"""
Normalizes raw VISTA output into a clean, inspectable intermediate form.

VISTA's stage 2/3 prompts ask the model to return a category name (VERIFIED,
CONTRADICTED, LACKING EVIDENCE, OUT-OF-SCOPE, ABSTENTION) at the start of its
response, but models don't always follow that format exactly -- extra
whitespace, markdown bold, a "CATEGORY:" echo, etc. This script strips that
formatting once, so every script downstream (metrics.py, plotting, ...) can
trust that claims[i]['category'] is one of the five valid categories.

Usage:
    python normalize.py

Reads MODEL_OUTPUT_FILES from config.py and writes one normalized JSON per
model to NORMALIZED_DIR, preserving the original conversation/turn shape.
"""
import json
import os
import re

from config import GOLD_LABEL_FIELD, NORMALIZED_DIR, get_model_output_files, normalized_output_path

VALID_CATEGORIES = {"VERIFIED", "CONTRADICTED", "LACKING EVIDENCE", "OUT-OF-SCOPE", "ABSTENTION"}


def clean_category(raw_category):
    """Extract the leading category label from a raw stage 2/3 model response.

    Returns None if no valid category could be recovered.
    """
    text = raw_category.strip()
    text = re.sub(r'^\d+\.\s*', '', text)      # "1. VERIFIED..."
    text = re.sub(r'^CATEGORY:\s*', '', text)  # "CATEGORY: VERIFIED..."
    text = text.replace('**', '')              # markdown bold

    for separator in ['.', ' - ', ' – ', '\n\n']:
        if separator in text:
            candidate = text.split(separator)[0].strip()
            if candidate in VALID_CATEGORIES:
                return candidate

    candidate = text.strip()
    return candidate if candidate in VALID_CATEGORIES else None


def normalize_conversations(conversations, gold_label_field, model_name):
    normalized = []
    stats = {"total_claims": 0, "unknown_claims": 0, "scored_turns": 0, "turns_missing_claims": 0}

    for conversation in conversations:
        normalized_conversation = []
        for turn in conversation:
            new_turn = dict(turn)
            has_reference = turn.get('retrieved_document', 'N/A') != 'N/A'

            if has_reference and 'claims' in turn:
                stats['scored_turns'] += 1
                clean_claims = []
                for claim_text, raw_category in turn['claims']:
                    category = clean_category(raw_category)
                    stats['total_claims'] += 1
                    if category is None:
                        stats['unknown_claims'] += 1
                    clean_claims.append({
                        'claim': claim_text,
                        'category': category,
                        'raw_category': raw_category,
                    })
                new_turn['claims'] = clean_claims
                new_turn['gold_label'] = turn.get(gold_label_field)
            elif has_reference:
                stats['turns_missing_claims'] += 1

            normalized_conversation.append(new_turn)
        normalized.append(normalized_conversation)

    if stats['unknown_claims']:
        rate = stats['unknown_claims'] / stats['total_claims'] * 100
        print(f"[{model_name}] WARNING: {stats['unknown_claims']}/{stats['total_claims']} "
              f"claims ({rate:.1f}%) had an unrecognized category -- inspect 'raw_category' "
              f"for these entries in the normalized output.")
    if stats['turns_missing_claims']:
        print(f"[{model_name}] NOTE: {stats['turns_missing_claims']} scoreable turns had no "
              f"'claims' field (the run that produced this file may be incomplete).")

    return normalized, stats


def main():
    os.makedirs(NORMALIZED_DIR, exist_ok=True)
    summary = {}

    for model_name, path in get_model_output_files().items():
        with open(path) as f:
            conversations = json.load(f)

        normalized, stats = normalize_conversations(conversations, GOLD_LABEL_FIELD, model_name)
        summary[model_name] = stats

        out_path = normalized_output_path(path)
        with open(out_path, 'w') as f:
            json.dump(normalized, f, indent=2)
        print(f"[{model_name}] wrote {out_path} "
              f"({stats['scored_turns']} scored turns, {stats['total_claims']} claims)")

    print("\nSummary:")
    for model_name, stats in summary.items():
        print(f"  {model_name:12} scored_turns={stats['scored_turns']:4} "
              f"claims={stats['total_claims']:4} unknown={stats['unknown_claims']}")


if __name__ == "__main__":
    main()
