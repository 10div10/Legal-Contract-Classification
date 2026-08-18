from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import requests

DATA_URL = (
    "https://huggingface.co/datasets/theatticusproject/cuad/raw/main/"
    "CUAD_v1/master_clauses.csv"
)

CLASS_PATTERNS = {
    "COLLABORATION_AGREEMENT": r"(?:collaboration agreement|strategic alliance agreement)",
    "LICENSE_AGREEMENT": (
        r"(?:content |trademark |patent |software |media |commercialization and )?"
        r"license agreement|licensing agreement"
    ),
    "SERVICE_AGREEMENT": r"(?:master )?services? agreement|servicing agreement",
    "DISTRIBUTOR_AGREEMENT": r"distributor agreement|distribution agreement",
    "MAINTENANCE_AGREEMENT": r"maintenance agreement|support and maintenance agreement",
    "SPONSORSHIP_AGREEMENT": r"sponsorship agreement",
    "DEVELOPMENT_AGREEMENT": r"development agreement|site development and hosting agreement",
}


def download_csv(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATA_URL, timeout=120)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def normalize_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text == "[]":
        return ""

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            text = " ".join(str(item) for item in parsed if item)
    except (ValueError, SyntaxError):
        pass

    text = re.sub(r"<omitted>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_document_text(row: pd.Series, context_columns: list[str]) -> str:
    parts = [normalize_cell(row[col]) for col in context_columns]
    return " ".join(part for part in parts if part)


def prepare_dataset(raw_path: Path, output_path: Path, min_chars: int = 100) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    filenames = df["Filename"].fillna("").astype(str)

    match_matrix = pd.DataFrame(
        {
            label: filenames.str.contains(pattern, case=False, regex=True)
            for label, pattern in CLASS_PATTERNS.items()
        },
        index=df.index,
    )

    match_count = match_matrix.sum(axis=1)
    selected = df.loc[match_count == 1].copy()
    selected["label"] = match_matrix.loc[match_count == 1].idxmax(axis=1)

    # Avoid direct title leakage: do not use Filename, Document Name or any Answer column.
    context_columns = [
        col
        for col in df.columns
        if col not in {"Filename", "Document Name", "Document Name-Answer"}
        and "Answer" not in col
    ]

    selected["text"] = selected.apply(
        lambda row: build_document_text(row, context_columns), axis=1
    )
    selected["n_chars"] = selected["text"].str.len()
    selected = selected.loc[selected["n_chars"] >= min_chars].copy()

    output = selected[["Filename", "label", "text", "n_chars"]].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "master_clauses.csv"
    output_path = project_root / "data" / "legal_contracts_7class.csv"

    if not raw_path.exists():
        print("Downloading CUAD master clauses CSV...")
        download_csv(raw_path)

    dataset = prepare_dataset(raw_path, output_path)
    print(f"Saved {len(dataset)} samples to {output_path}")
    print("\nClass distribution:")
    print(dataset["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
