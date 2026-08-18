from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        import fitz  # PyMuPDF

        document = fitz.open(path)
        return "\n".join(page.get_text("text") for page in document)
    raise ValueError("Only .txt and .pdf files are supported.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a legal contract document.")
    parser.add_argument("file", type=Path, help="Path to a .txt or .pdf contract")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "legal_contract_classifier.joblib"
    model = joblib.load(model_path)

    text = read_document(args.file)
    predicted = model.predict([text])[0]

    scores = model.decision_function([text])[0]
    classes = model.named_steps["model"].classes_
    order = np.argsort(scores)[::-1][:3]

    print(f"Prediction: {predicted}")
    print("\nTop decision scores (not calibrated probabilities):")
    for idx in order:
        print(f"  {classes[idx]:30s} {scores[idx]: .4f}")


if __name__ == "__main__":
    main()
