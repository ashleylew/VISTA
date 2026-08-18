"""
Reports summary statistics and VISTA scores for each model, using the
normalized output of normalize.py. Where a turn has a gold label
(dataset-dependent -- new/unlabeled data won't), also reports
accuracy/precision/recall/F1 against it.

Run `python normalize.py` first.

A claim is "VISTA-safe" if its category is VERIFIED, ABSTENTION, or
OUT-OF-SCOPE. A turn is predicted "hallucinated" unless every one of its
claims is safe. The VISTA score of a turn is the proportion of its claims
that are safe (claims with an unrecognized category are excluded from both
the numerator and denominator, since we don't actually know if they're
safe); the overall VISTA score is that same proportion computed over every
claim across every turn. Precision/recall/F1 (where computed) treat
"hallucinated" as the positive class.

Usage:
    python metrics.py

Writes one {input_file}_SUMMARY.json per model to SUMMARIES_DIR, with the
full per-turn breakdown; prints the aggregate numbers to the terminal.
"""
import json
import os

from config import get_model_output_files, normalized_output_path, summary_output_path, SUMMARIES_DIR

VISTA_SAFE_CATEGORIES = {"VERIFIED", "ABSTENTION", "OUT-OF-SCOPE"}


def predict_label(claims):
    """Returns 'faithful', 'hallucinated', or 'unknown' for one turn."""
    categories = [c['category'] for c in claims]
    if any(category is None for category in categories):
        return 'unknown'
    if all(category in VISTA_SAFE_CATEGORIES for category in categories):
        return 'faithful'
    return 'hallucinated'


def turn_vista_score(claims):
    """Proportion of a turn's claims that are VISTA-safe.

    Returns None if the turn made no (recognizable-category) claims --
    there's nothing to compute a proportion over.
    """
    scoreable = [c for c in claims if c['category'] is not None]
    if not scoreable:
        return None
    safe = sum(1 for c in scoreable if c['category'] in VISTA_SAFE_CATEGORIES)
    return safe / len(scoreable)


def load_examples(input_path):
    """Loads a model's normalized turns, tagged with their position.

    Turns are tagged with (conversation_index, turn_index) rather than
    relying on dataset-specific id fields, since not every source dataset
    carries those.
    """
    path = normalized_output_path(input_path)
    with open(path) as f:
        conversations = json.load(f)

    examples = []
    for conversation_index, conversation in enumerate(conversations):
        for turn_index, turn in enumerate(conversation):
            if 'claims' in turn:
                examples.append({
                    'conversation_index': conversation_index,
                    'turn_index': turn_index,
                    'claims': turn['claims'],
                    'gold_label': turn.get('gold_label'),
                })
    return examples


def summarize_claims(examples):
    category_counts = {}
    total_claims = 0
    for turn in examples:
        for c in turn['claims']:
            total_claims += 1
            category_counts[c['category']] = category_counts.get(c['category'], 0) + 1

    return {
        'num_turns': len(examples),
        'total_claims': total_claims,
        'avg_claims_per_turn': total_claims / len(examples) if examples else 0.0,
        'category_counts': category_counts,
    }


def overall_vista_score(examples):
    total = safe = 0
    for turn in examples:
        for c in turn['claims']:
            if c['category'] is None:
                continue
            total += 1
            if c['category'] in VISTA_SAFE_CATEGORIES:
                safe += 1
    return safe / total if total else None


def summarize_predictions(examples):
    counts = {'faithful': 0, 'hallucinated': 0, 'unknown': 0}
    for turn in examples:
        counts[predict_label(turn['claims'])] += 1
    return counts


def evaluate_against_gold(examples):
    """Returns None if no turn in `examples` has a usable gold label."""
    tp = fp = tn = fn = 0
    for turn in examples:
        gold = turn.get('gold_label')
        if gold not in ('faithful', 'hallucinated'):
            continue
        pred = predict_label(turn['claims'])
        if pred == 'unknown':
            continue
        if gold == 'hallucinated' and pred == 'hallucinated':
            tp += 1
        elif gold == 'faithful' and pred == 'faithful':
            tn += 1
        elif gold == 'faithful' and pred == 'hallucinated':
            fp += 1
        elif gold == 'hallucinated' and pred == 'faithful':
            fn += 1

    scored = tp + fp + tn + fn
    if scored == 0:
        return None

    accuracy = (tp + tn) / scored
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {'n': scored, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def main():
    os.makedirs(SUMMARIES_DIR, exist_ok=True)

    for model_name, input_path in get_model_output_files().items():
        examples = load_examples(input_path)
        claim_summary = summarize_claims(examples)
        vista_score = overall_vista_score(examples)
        prediction_counts = summarize_predictions(examples)
        gold_metrics = evaluate_against_gold(examples)

        print(f"\n{model_name}")
        print(f"  turns={claim_summary['num_turns']} total_claims={claim_summary['total_claims']} "
              f"avg_claims_per_turn={claim_summary['avg_claims_per_turn']:.2f}")
        print(f"  claim categories: {claim_summary['category_counts']}")
        print(f"  VISTA score (safe claims / total claims): "
              f"{vista_score*100:.2f}%" if vista_score is not None else "  VISTA score: n/a (no claims)")
        print(f"  turn predictions: faithful={prediction_counts['faithful']} "
              f"hallucinated={prediction_counts['hallucinated']} unknown={prediction_counts['unknown']}")
        if gold_metrics is None:
            print("  (no gold labels found -- skipping accuracy/precision/recall/F1)")
        else:
            m = gold_metrics
            print(f"  vs. gold (n={m['n']}): accuracy={m['accuracy']*100:.2f}% "
                  f"precision={m['precision']*100:.2f}% recall={m['recall']*100:.2f}% f1={m['f1']*100:.2f}%")

        summary = {
            'model': model_name,
            'claim_summary': claim_summary,
            'overall_vista_score': vista_score,
            'prediction_counts': prediction_counts,
            'gold_metrics': gold_metrics,
            'turns': [
                {
                    'conversation_index': turn['conversation_index'],
                    'turn_index': turn['turn_index'],
                    'num_claims': len(turn['claims']),
                    'vista_score': turn_vista_score(turn['claims']),
                    'predicted_label': predict_label(turn['claims']),
                    'gold_label': turn['gold_label'],
                }
                for turn in examples
            ],
        }

        out_path = summary_output_path(input_path)
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
