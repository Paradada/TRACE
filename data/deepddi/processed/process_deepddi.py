import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parents[2] / "data" / "miracle" / "datachem"
SPLITS = ("train", "valid", "test")


def read_and_merge():
    pairs = {}
    row_count = 0
    duplicate_count = 0

    for split in SPLITS:
        source_file = SOURCE_DIR / f"DeepDDI_{split}.csv"
        with source_file.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"smiles_1", "smiles_2", "label"}
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{source_file} must contain columns {sorted(required_columns)}"
                )

            for line_number, row in enumerate(reader, start=2):
                row_count += 1
                smiles_1 = row["smiles_1"].strip()
                smiles_2 = row["smiles_2"].strip()
                if not smiles_1 or not smiles_2:
                    raise ValueError(
                        f"Empty SMILES in {source_file}, line {line_number}"
                    )

                try:
                    numeric_label = float(row["label"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid label in {source_file}, line {line_number}: "
                        f"{row['label']!r}"
                    ) from error
                if numeric_label not in (0.0, 1.0):
                    raise ValueError(
                        f"Expected binary label in {source_file}, line "
                        f"{line_number}: {row['label']!r}"
                    )

                label = int(numeric_label)
                pair_key = tuple(sorted((smiles_1, smiles_2)))
                if pair_key in pairs:
                    existing_label, _ = pairs[pair_key]
                    if existing_label != label:
                        raise ValueError(
                            f"Conflicting labels for pair {pair_key}: "
                            f"{existing_label} and {label} "
                            f"(found in {source_file}, line {line_number})"
                        )
                    duplicate_count += 1
                    continue

                pairs[pair_key] = (
                    label,
                    {
                        "smiles_1": smiles_1,
                        "smiles_2": smiles_2,
                        "label": label,
                    },
                )

    return row_count, duplicate_count, [value[1] for value in pairs.values()]


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    row_count, duplicate_count, merged_rows = read_and_merge()
    merged_file = SCRIPT_DIR / "DeepDDI.csv"
    mapping_file = SCRIPT_DIR / "drug_list_deep.csv"

    write_csv(merged_file, ["smiles_1", "smiles_2", "label"], merged_rows)

    unique_smiles = sorted(
        {
            smiles
            for row in merged_rows
            for smiles in (row["smiles_1"], row["smiles_2"])
        }
    )
    write_csv(
        mapping_file,
        ["DrugID", "Smiles"],
        ({"DrugID": smiles, "Smiles": smiles} for smiles in unique_smiles),
    )

    print(f"Source rows: {row_count}")
    print(f"Duplicate pairs removed: {duplicate_count}")
    print(f"Merged rows: {len(merged_rows)}")
    print(f"Unique SMILES: {len(unique_smiles)}")
    print(f"Dataset: {merged_file}")
    print(f"Drug mapping: {mapping_file}")


if __name__ == "__main__":
    main()
