import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = REPO_ROOT / "data" / "chchminer" / "processed"
CONFIG_FILE = PROCESSED_DIR / "config_chchminer.yml"
SAMPLE_FILE = PROCESSED_DIR / "smoke_test" / "ChChMiner_sample.csv"
SMOKE_CONFIG_FILE = PROCESSED_DIR / "smoke_test" / "config_smoke_test.yml"
SAMPLE_ROWS = 1000
EPOCHS = 3
BATCH_SIZE = 16
SEED = 1314


def main():
    with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    source_file = REPO_ROOT / config["data_dir"]
    graph_file = REPO_ROOT / config["data_bin_dir"]
    mapping_file = REPO_ROOT / config["id_to_index_dir"]
    for required_file in (source_file, graph_file, mapping_file):
        if not required_file.is_file():
            raise FileNotFoundError(f"Required input file not found: {required_file}")

    data = pd.read_csv(source_file)
    if not {"smiles_1", "smiles_2", "label"}.issubset(data.columns):
        raise ValueError("Dataset must contain smiles_1, smiles_2, and label columns")

    classes = data["label"].unique()
    if len(classes) != 2:
        raise ValueError(f"Expected a binary dataset; found labels: {classes}")

    sample_per_class = SAMPLE_ROWS // len(classes)
    parts = []
    for label in sorted(classes):
        class_rows = data[data["label"] == label]
        if len(class_rows) < sample_per_class:
            raise ValueError(
                f"Not enough rows for label {label}: "
                f"need {sample_per_class}, found {len(class_rows)}"
            )
        parts.append(
            class_rows.sample(n=sample_per_class, random_state=SEED)
        )
    sample = (
        pd.concat(parts)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_FILE, index=False)

    config["data_dir"] = SAMPLE_FILE.relative_to(REPO_ROOT).as_posix()
    config["best_model_dir"] = (
        SAMPLE_FILE.parent / "best_model_smoke_test.pth"
    ).relative_to(REPO_ROOT).as_posix()
    config["checkpoint_dir"] = (
        SAMPLE_FILE.parent / "output"
    ).relative_to(REPO_ROOT).as_posix() + "/"
    config["experiment_name"] = "chchminer_smoke_test"
    config["num_epochs"] = EPOCHS
    config["batch_size"] = BATCH_SIZE
    config["patience"] = 2
    config["seed"] = SEED

    with SMOKE_CONFIG_FILE.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)

    print(f"Sample rows: {len(sample)}")
    print(f"Sample CSV: {SAMPLE_FILE}")
    print(f"Smoke-test config: {SMOKE_CONFIG_FILE}")
    temporary_checkpoints = [
        REPO_ROOT / config["best_model_dir"],
        REPO_ROOT / config["checkpoint_dir"] / config["experiment_name"],
    ]
    try:
        subprocess.run(
            [
                sys.executable,
                "train_task.py",
                "--config_file",
                SMOKE_CONFIG_FILE.relative_to(REPO_ROOT).as_posix(),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
    finally:
        for checkpoint in temporary_checkpoints:
            checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
