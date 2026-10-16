import csv
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parents[2] / "data" / "miracle" / "datachem"
SPLITS = ("train", "valid", "test")


def canonical_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def load_reference_mapping():
    reference_file = SOURCE_DIR / "drug_list_zhang.csv"
    mapping = defaultdict(set)
    vocabulary = set()
    with reference_file.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not {"drugbank_id", "smiles"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected columns in {reference_file}")
        for row in reader:
            drug_id = row["drugbank_id"].strip()
            smiles = row["smiles"].strip()
            if drug_id and smiles:
                canonical = canonical_smiles(smiles)
                mapping[drug_id].add(canonical or f"!INVALID!{smiles}")
                vocabulary.add(smiles)
    return mapping, vocabulary


def read_source_rows():
    rows = []
    for split in SPLITS:
        source_file = SOURCE_DIR / f"ZhangDDI_{split}.csv"
        with source_file.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {
                "drugbank_id_1",
                "drugbank_id_2",
                "smiles_1",
                "smiles_2",
                "label",
            }
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{source_file} must contain columns "
                    f"{sorted(required_columns)}"
                )
            for line_number, source_row in enumerate(reader, start=2):
                row = {
                    key: source_row[key].strip()
                    for key in required_columns
                }
                row["source"] = source_file.name
                row["line"] = line_number
                rows.append(row)
    return rows


def write_report(report_file, lines):
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    report_file = SCRIPT_DIR / "process_zhangddi_audit.txt"
    rows = read_source_rows()
    reference, source_vocabulary = load_reference_mapping()
    observed_smiles = defaultdict(set)
    observed_canonical = defaultdict(set)
    canonical_to_ids = defaultdict(set)
    pair_groups = defaultdict(list)
    raw_label_counts = Counter()
    invalid_labels = []
    missing_values = []
    invalid_smiles = []
    vocabulary_misses = []

    for row in rows:
        id_1, id_2 = row["drugbank_id_1"], row["drugbank_id_2"]
        smiles_1, smiles_2 = row["smiles_1"], row["smiles_2"]
        if not id_1 or not id_2 or not smiles_1 or not smiles_2:
            missing_values.append((row["source"], row["line"]))
            continue
        try:
            numeric_label = float(row["label"])
            if numeric_label not in (0.0, 1.0):
                raise ValueError
            label = int(numeric_label)
        except (TypeError, ValueError):
            invalid_labels.append((row["source"], row["line"], row["label"]))
            continue
        row["label"] = label
        raw_label_counts[label] += 1
        for smiles in (smiles_1, smiles_2):
            if smiles not in source_vocabulary:
                vocabulary_misses.append((row["source"], row["line"], smiles))
        pair_groups[tuple(sorted((id_1, id_2)))].append(row)

        for drug_id, smiles in ((id_1, smiles_1), (id_2, smiles_2)):
            observed_smiles[drug_id].add(smiles)
            canonical = canonical_smiles(smiles)
            if canonical is None:
                invalid_smiles.append((row["source"], row["line"], drug_id))
                continue
            observed_canonical[drug_id].add(canonical)
            canonical_to_ids[canonical].add(drug_id)

    conflict_groups = {
        pair: pair_rows
        for pair, pair_rows in pair_groups.items()
        if len({row["label"] for row in pair_rows}) > 1
    }
    nonconflicting_groups = {
        pair: pair_rows
        for pair, pair_rows in pair_groups.items()
        if pair not in conflict_groups
    }
    duplicate_rows_removed = sum(
        len(pair_rows) - 1 for pair_rows in nonconflicting_groups.values()
    )
    clean_rows = [
        pair_rows[0] for pair_rows in nonconflicting_groups.values()
    ]
    clean_label_counts = Counter(row["label"] for row in clean_rows)

    mismatch_ids = {
        drug_id
        for drug_id, canonical_set in observed_canonical.items()
        if drug_id not in reference or canonical_set != reference[drug_id]
    }
    multi_raw_smiles_ids = {
        drug_id for drug_id, values in observed_smiles.items() if len(values) > 1
    }
    multi_canonical_smiles_ids = {
        drug_id
        for drug_id, values in observed_canonical.items()
        if len(values) > 1
    }
    shared_canonical_observed = {
        smiles: ids for smiles, ids in canonical_to_ids.items() if len(ids) > 1
    }
    reference_canonical_to_ids = defaultdict(set)
    for drug_id, smiles_set in reference.items():
        for smiles in smiles_set:
            reference_canonical_to_ids[smiles].add(drug_id)
    shared_canonical_reference = {
        smiles: ids
        for smiles, ids in reference_canonical_to_ids.items()
        if len(ids) > 1
    }

    report_lines = [
        "ZhangDDI processing audit",
        f"Source files: {', '.join(f'ZhangDDI_{split}.csv' for split in SPLITS)}",
        f"Source rows: {len(rows)}",
        "Rows by source file: "
        + ", ".join(
            f"{split}={sum(row['source'] == f'ZhangDDI_{split}.csv' for row in rows)}"
            for split in SPLITS
        ),
        f"Input label counts: negative={raw_label_counts[0]}, positive={raw_label_counts[1]}",
        f"Missing ID/SMILES records: {len(missing_values)}",
        f"Invalid label records: {len(invalid_labels)}",
        f"Invalid SMILES occurrences: {len(invalid_smiles)}",
        f"Pair SMILES assignments absent from drug_list_zhang.csv vocabulary: {len(vocabulary_misses)}",
        f"Unique unordered DrugBank-ID pairs: {len(pair_groups)}",
        f"Repeated rows removed: {duplicate_rows_removed}",
        f"Conflicting unordered ID pairs: {len(conflict_groups)}",
        f"Records in conflicting pairs (all excluded): "
        f"{sum(len(pair_rows) for pair_rows in conflict_groups.values())}",
        f"Unique rows after conflict exclusion and deduplication: {len(clean_rows)}",
        f"Remaining negative samples: {clean_label_counts[0]}",
        f"Remaining positive samples: {clean_label_counts[1]}",
        f"Observed IDs with multiple raw SMILES: {len(multi_raw_smiles_ids)}",
        f"Observed IDs with multiple canonical SMILES: {len(multi_canonical_smiles_ids)}",
        f"Observed canonical SMILES shared by multiple IDs: {len(shared_canonical_observed)}",
        f"Reference canonical SMILES shared by multiple IDs: {len(shared_canonical_reference)}",
        f"IDs in drug_list_zhang.csv: {len(reference)}",
        f"Exact SMILES vocabulary entries in drug_list_zhang.csv: {len(source_vocabulary)}",
        "IDs in drug_list_zhang.csv with multiple canonical SMILES: "
        f"{sum(len(values) > 1 for values in reference.values())}",
        f"IDs inconsistent with drug_list_zhang.csv: {len(mismatch_ids)}",
    ]
    write_report(report_file, report_lines)

    if missing_values or invalid_labels or invalid_smiles or vocabulary_misses:
        raise ValueError(f"Input validation failed; inspect {report_file}")
    if mismatch_ids or multi_canonical_smiles_ids:
        raise ValueError(
            "ZhangDDI DrugBank IDs map to inconsistent SMILES; "
            f"no final CSV was written. Inspect {report_file}"
        )
    if shared_canonical_observed or shared_canonical_reference:
        raise ValueError(
            "Different DrugBank IDs share canonical SMILES; no IDs were "
            f"merged. Inspect {report_file}"
        )

    output_rows = []
    for row in clean_rows:
        smiles_1, smiles_2 = row["smiles_1"], row["smiles_2"]
        canonical_1 = canonical_smiles(smiles_1)
        canonical_2 = canonical_smiles(smiles_2)
        if canonical_2 < canonical_1:
            smiles_1, smiles_2 = smiles_2, smiles_1
        output_rows.append(
            {"smiles_1": smiles_1, "smiles_2": smiles_2, "label": row["label"]}
        )

    canonical_pair_labels = defaultdict(set)
    for row in output_rows:
        pair = tuple(
            sorted(
                (
                    canonical_smiles(row["smiles_1"]),
                    canonical_smiles(row["smiles_2"]),
                )
            )
        )
        canonical_pair_labels[pair].add(row["label"])
    output_conflicts = sum(
        len(labels) > 1 for labels in canonical_pair_labels.values()
    )
    if output_conflicts:
        raise ValueError(
            "Final SMILES pairs have contradictory labels; no final CSV was written"
        )

    checked_output_pairs = set()
    prewrite_labels = Counter()
    prewrite_order_violations = 0
    for row in output_rows:
        if row["label"] not in (0, 1):
            raise ValueError("Invalid label found before writing final CSV")
        if not row["smiles_1"] or not row["smiles_2"]:
            raise ValueError("Missing SMILES found before writing final CSV")
        canonical_1 = canonical_smiles(row["smiles_1"])
        canonical_2 = canonical_smiles(row["smiles_2"])
        if canonical_1 is None or canonical_2 is None:
            raise ValueError("Invalid SMILES found before writing final CSV")
        if row["smiles_1"] not in source_vocabulary or row["smiles_2"] not in source_vocabulary:
            raise ValueError("Output SMILES is not present in the source vocabulary")
        if canonical_1 > canonical_2:
            prewrite_order_violations += 1
        pair = tuple(sorted((canonical_1, canonical_2)))
        if pair in checked_output_pairs:
            raise ValueError("Duplicate canonical unordered pair before writing final CSV")
        checked_output_pairs.add(pair)
        prewrite_labels[row["label"]] += 1
    if (
        prewrite_order_violations
        or len(output_rows) != len(clean_rows)
        or prewrite_labels != clean_label_counts
    ):
        raise ValueError("Pre-write output validation failed")

    output_file = SCRIPT_DIR / "ZhangDDI.csv"
    mapping_file = SCRIPT_DIR / "drug_list_zhang.csv"
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["smiles_1", "smiles_2", "label"]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    unique_smiles = sorted(
        {smiles for row in output_rows for smiles in (row["smiles_1"], row["smiles_2"])}
    )
    with mapping_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["DrugID", "Smiles"])
        writer.writeheader()
        writer.writerows({"DrugID": smiles, "Smiles": smiles} for smiles in unique_smiles)

    with output_file.open("r", newline="", encoding="utf-8") as csv_file:
        final_rows = list(csv.DictReader(csv_file))
    final_labels = Counter()
    final_pairs = set()
    final_invalid = 0
    order_violations = 0
    for row in final_rows:
        try:
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError
        except (TypeError, ValueError):
            final_invalid += 1
            continue
        if not row["smiles_1"] or not row["smiles_2"]:
            final_invalid += 1
        canonical_1 = canonical_smiles(row["smiles_1"])
        canonical_2 = canonical_smiles(row["smiles_2"])
        if canonical_1 is None or canonical_2 is None:
            final_invalid += 1
            continue
        if canonical_1 > canonical_2:
            order_violations += 1
        final_labels[label] += 1
        pair = tuple(sorted((canonical_1, canonical_2)))
        if pair in final_pairs:
            raise ValueError("Duplicate unordered canonical SMILES pair in final CSV")
        final_pairs.add(pair)
    if final_invalid or order_violations or len(final_rows) != len(output_rows):
        raise ValueError("Final CSV validation failed")

    report_lines.extend(
        [
            f"Final CSV rows: {len(final_rows)}",
            f"Final negative samples: {final_labels[0]}",
            f"Final positive samples: {final_labels[1]}",
            f"Final duplicate unordered canonical SMILES pairs: 0",
            f"Final conflicting label pairs: {output_conflicts}",
            f"Final missing/invalid records: {final_invalid}",
            f"Normalized pair order violations: {order_violations}",
            f"Final unique SMILES mapping rows: {len(unique_smiles)}",
            f"Dataset output: {output_file}",
            f"SMILES mapping output: {mapping_file}",
            f"Audit log: {report_file}",
        ]
    )
    write_report(report_file, report_lines)
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
