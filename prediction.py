"""
prediction.py - Stage 9: use the saved model to predict a disease from symptoms

Can be used two ways:

1. From other Python code (app.py does this):
       from prediction import load_bundle, predict
       result = predict(["itching", "skin_rash", "nodal_skin_eruptions"])

2. From the terminal:
       python prediction.py itching skin_rash nodal_skin_eruptions
       python prediction.py --list          # show all symptom names
       python prediction.py                 # interactive: type symptoms

No retraining happens here - the model is loaded from models/disease_model.joblib.

EDUCATIONAL PROTOTYPE ONLY - NOT A MEDICAL DIAGNOSIS.
"""
import difflib
import sys
from functools import lru_cache

import joblib
import numpy as np

import config

DISCLAIMER = ("This is an educational machine-learning prototype trained on a small "
              "synthetic dataset. It is NOT a medical diagnosis and must not replace "
              "advice from a qualified doctor. If you feel unwell, seek medical care.")
MIN_RECOMMENDED_SYMPTOMS = 3


@lru_cache(maxsize=1)
def load_bundle(path=config.FINAL_MODEL_PATH):
    """Load the saved model bundle once and reuse it (cached)."""
    try:
        bundle = joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model not found at {path}. Run data_preprocessing.py and "
            "model_training.py first.") from None
    required = {"model", "label_encoder", "feature_columns"}
    if not required <= bundle.keys():
        raise ValueError(f"Model file is missing {required - bundle.keys()}")
    return bundle


def readable(symptom):
    """'yellowish_skin' -> 'Yellowish skin' (for display)."""
    return symptom.replace("_", " ").strip().capitalize()


def match_symptom(text, features):
    """
    Turn user text into a known symptom column name.
    Accepts 'Skin rash', 'skin_rash', 'skin rash'; tolerates small typos.
    Returns (matched_name or None, suggestions).
    """
    key = "_".join(text.strip().lower().replace("-", " ").split())
    if key in features:
        return key, []
    close = difflib.get_close_matches(key, features, n=3, cutoff=0.6)
    if close and difflib.SequenceMatcher(None, key, close[0]).ratio() >= 0.85:
        return close[0], []            # confident typo fix, e.g. 'skin_rashh'
    return None, close


def build_vector(symptoms, features):
    """0/1 vector in the exact column order the model was trained on."""
    index = {f: i for i, f in enumerate(features)}
    x = np.zeros((1, len(features)), dtype=np.int8)
    for s in symptoms:
        x[0, index[s]] = 1
    return x


def similarity_ranking(x, bundle):
    """
    For each disease, cosine similarity between the patient and the most
    similar known pattern of that disease (1.0 = identical). Used instead of
    probabilities when the model's probabilities are not reliable.
    """
    ref_X, ref_y = bundle["reference_X"].astype(float), bundle["reference_y"]
    v = x.astype(float)[0]
    sims = ref_X @ v / (np.linalg.norm(ref_X, axis=1) * np.linalg.norm(v) + 1e-12)
    n_classes = len(bundle["label_encoder"].classes_)
    best = np.zeros(n_classes)
    for cls in range(n_classes):
        best[cls] = sims[ref_y == cls].max()
    return best


def predict(symptoms, top_k=5):
    """
    Predict the disease for a list of symptom names.

    Returns a dict with:
      predicted_disease, score_type ('probability' or 'similarity'),
      ranking [(disease, score), ...], used_symptoms, unknown_symptoms,
      suggestions {unknown: [close matches]}, warnings, disclaimer
    """
    bundle = load_bundle()
    features = bundle["feature_columns"]
    encoder = bundle["label_encoder"]

    used, unknown, suggestions = [], [], {}
    for s in symptoms:
        if not str(s).strip():
            continue
        match, close = match_symptom(str(s), features)
        if match:
            if match not in used:
                used.append(match)
        else:
            unknown.append(s)
            suggestions[s] = close

    if not used:
        hints = {k: v for k, v in suggestions.items() if v}
        raise ValueError("No recognised symptoms. Use --list to see valid names."
                         + (f" Did you mean: {hints}?" if hints else ""))

    warnings = []
    if len(used) < MIN_RECOMMENDED_SYMPTOMS:
        warnings.append(f"Only {len(used)} symptom(s) given - predictions are much less "
                        f"reliable with fewer than {MIN_RECOMMENDED_SYMPTOMS}.")

    x = build_vector(used, features)
    model = bundle["model"]
    predicted = encoder.inverse_transform(model.predict(x))[0]

    if bundle.get("probabilities_reliable", False):
        scores, score_type = model.predict_proba(x)[0], "probability"
    else:
        scores, score_type = similarity_ranking(x, bundle), "similarity"

    order = np.argsort(scores)[::-1][:top_k]
    ranking = [(encoder.classes_[i], float(scores[i])) for i in order]

    # If several diseases tie for the top score, say so honestly
    ties = [d for d, s in ranking if abs(s - ranking[0][1]) < 1e-9]
    if len(ties) > 1:
        warnings.append("Several diseases match equally well: " + ", ".join(ties)
                        + ". Adding more symptoms would help tell them apart.")

    return {
        "predicted_disease": predicted,
        "model_name": bundle.get("model_name", "model"),
        "score_type": score_type,
        "ranking": ranking,
        "used_symptoms": used,
        "unknown_symptoms": unknown,
        "suggestions": suggestions,
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }


def print_result(result):
    print("\n" + "=" * 70)
    print(f"Symptoms used: {', '.join(readable(s) for s in result['used_symptoms'])}")
    for u in result["unknown_symptoms"]:
        hint = result["suggestions"].get(u)
        print(f"  Not recognised: '{u}'" + (f" - did you mean {hint}?" if hint else ""))
    print(f"\nMost likely disease ({result['model_name']}): {result['predicted_disease']}")
    label = ("Probability" if result["score_type"] == "probability"
             else "Similarity to closest known case")
    print(f"\nTop matches - {label}:")
    for disease, score in result["ranking"]:
        print(f"  {disease:45s} {score * 100:5.1f}%")
    for w in result["warnings"]:
        print(f"\nNote: {w}")
    print(f"\nDISCLAIMER: {result['disclaimer']}")
    print("=" * 70)


def main(argv):
    try:
        bundle = load_bundle()
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    if argv and argv[0] in ("--list", "-l"):
        for s in bundle["feature_columns"]:
            print(s)
        return
    if argv:
        symptoms = argv
    else:
        print("Enter symptoms separated by commas (e.g. itching, skin rash, chills).")
        print("Type --list to see all symptom names.")
        text = input("> ").strip()
        if text in ("--list", "-l"):
            print("\n".join(bundle["feature_columns"]))
            return
        symptoms = [t for t in text.split(",") if t.strip()]
    try:
        print_result(predict(symptoms))
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
