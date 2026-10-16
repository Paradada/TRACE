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
    reference_file = SOURCE_DIR / "drug_list_miner.csv"
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
        source_file = SOURCE_DIR / f"ChChMiner_{split}.csv"
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


def write_top_id_variants(variant_counts):
    variant_file = SCRIPT_DIR / "chchminer_id_smiles_variants_top20.csv"
    multi_smiles_ids = sorted(
        drug_id
        for drug_id, by_file in variant_counts.items()
        if len({smiles for counts in by_file.values() for smiles in counts}) > 1
    )
    top_ids = multi_smiles_ids[:20]
    source_files = [f"ChChMiner_{split}.csv" for split in SPLITS]

    with variant_file.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "drugbank_id",
            "canonical_smiles",
            "total_occurrences",
            *[f"{split}_occurrences" for split in SPLITS],
            "source_files",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for drug_id in top_ids:
            variants = sorted(
                {
                    smiles
                    for counts in variant_counts[drug_id].values()
                    for smiles in counts
                }
            )
            for smiles in variants:
                counts_by_file = {
                    source_file: variant_counts[drug_id][source_file][smiles]
                    for source_file in source_files
                }
                writer.writerow(
                    {
                        "drugbank_id": drug_id,
                        "canonical_smiles": smiles,
                        "total_occurrences": sum(counts_by_file.values()),
                        **{
                            f"{split}_occurrences": counts_by_file[
                                f"ChChMiner_{split}.csv"
                            ]
                            for split in SPLITS
                        },
                        "source_files": "|".join(
                            source_file
                            for source_file in source_files
                            if counts_by_file[source_file]
                        ),
                    }
                )
    return variant_file, top_ids


def main():
    report_file = SCRIPT_DIR / "process_chchminer_audit.txt"
    rows = read_source_rows()
    reference, source_vocabulary = load_reference_mapping()
    observed_smiles = defaultdict(set)
    observed_canonical = defaultdict(set)
    variant_counts = defaultdict(lambda: defaultdict(Counter))
    canonical_to_ids = defaultdict(set)
    id_pair_groups = defaultdict(list)
    smiles_pair_groups = defaultdict(list)
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

        id_pair_groups[tuple(sorted((id_1, id_2)))].append(row)
        canonical_pair = []
        for drug_id, smiles in ((id_1, smiles_1), (id_2, smiles_2)):
            observed_smiles[drug_id].add(smiles)
            canonical = canonical_smiles(smiles)
            if canonical is None:
                invalid_smiles.append((row["source"], row["line"], drug_id))
                continue
            observed_canonical[drug_id].add(canonical)
            variant_counts[drug_id][row["source"]][canonical] += 1
            canonical_to_ids[canonical].add(drug_id)
            canonical_pair.append(canonical)
        if len(canonical_pair) == 2:
            smiles_pair_groups[tuple(sorted(canonical_pair))].append(row)

    variant_file, top_ids = write_top_id_variants(variant_counts)
    within_file_multi_ids = {
        drug_id
        for drug_id, by_file in variant_counts.items()
        if len({smiles for counts in by_file.values() for smiles, n in counts.items() if n}) > 1
        and any(len(counts) > 1 for counts in by_file.values())
    }
    cross_file_only_multi_ids = {
        drug_id
        for drug_id, by_file in variant_counts.items()
        if len({smiles for counts in by_file.values() for smiles, n in counts.items() if n}) > 1
        and not any(len(counts) > 1 for counts in by_file.values())
    }
    multi_ids_spanning_files = {
        drug_id
        for drug_id, by_file in variant_counts.items()
        if len({smiles for counts in by_file.values() for smiles, n in counts.items() if n}) > 1
        and sum(bool(counts) for counts in by_file.values()) > 1
    }
    within_file_counts = {
        split: sum(
            len(variant_counts[drug_id].get(f"ChChMiner_{split}.csv", {})) > 1
            for drug_id in variant_counts
        )
        for split in SPLITS
    }

    id_conflict_groups = {
        pair: pair_rows
        for pair, pair_rows in id_pair_groups.items()
        if len({row["label"] for row in pair_rows}) > 1
    }
    conflict_groups = {
        pair: pair_rows
        for pair, pair_rows in smiles_pair_groups.items()
        if len({row["label"] for row in pair_rows}) > 1
    }
    nonconflicting_groups = {
        pair: pair_rows
        for pair, pair_rows in smiles_pair_groups.items()
        if pair not in conflict_groups
    }
    duplicate_rows_removed = sum(
        len(pair_rows) - 1 for pair_rows in nonconflicting_groups.values()
    )
    clean_label_counts = Counter(
        pair_rows[0]["label"] for pair_rows in nonconflicting_groups.values()
    )

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
    conflict_rows = sum(len(pair_rows) for pair_rows in conflict_groups.values())
    clean_rows = [
        pair_rows[0] for pair_rows in nonconflicting_groups.values()
    ]
    duplicate_rows_total = len(rows) - len(smiles_pair_groups)
    duplicate_rows_in_conflicts = conflict_rows - len(conflict_groups)
    nonreference_variant_count = sum(
        len(
            {
                smiles
                for counts in by_file.values()
                for smiles in counts
                if smiles not in reference.get(drug_id, set())
            }
        )
        for drug_id, by_file in variant_counts.items()
    )
    nonreference_assignment_count = sum(
        occurrences
        for drug_id, by_file in variant_counts.items()
        for counts in by_file.values()
        for smiles, occurrences in counts.items()
        if smiles not in reference.get(drug_id, set())
    )
    conflict_molecules = {
        smiles for pair in conflict_groups for smiles in pair
    }
    clean_molecules = {
        smiles for pair in nonconflicting_groups for smiles in pair
    }
    all_molecules = {
        smiles for pair in smiles_pair_groups for smiles in pair
    }

    report_lines = [
        "ChChMiner processing audit",
        f"Source files: {', '.join(f'ChChMiner_{split}.csv' for split in SPLITS)}",
        f"Source rows: {len(rows)}",
        "Rows by source file: "
        + ", ".join(
            f"{split}={sum(row['source'] == f'ChChMiner_{split}.csv' for row in rows)}"
            for split in SPLITS
        ),
        f"Input label counts: negative={raw_label_counts[0]}, positive={raw_label_counts[1]}",
        f"Missing ID/SMILES records: {len(missing_values)}",
        f"Invalid label records: {len(invalid_labels)}",
        f"Invalid SMILES occurrences: {len(invalid_smiles)}",
        f"Pair SMILES assignments absent from drug_list_miner.csv vocabulary: {len(vocabulary_misses)}",
        f"Unique unordered DrugBank-ID pairs: {len(id_pair_groups)}",
        f"ID-pair label conflicts: {len(id_conflict_groups)}",
        f"Unique canonical unordered SMILES pairs: {len(smiles_pair_groups)}",
        f"Repeated rows across canonical SMILES pairs: {duplicate_rows_total}",
        f"Repeated rows within conflicting pairs: {duplicate_rows_in_conflicts}",
        f"Repeated rows among non-conflicting pairs: {duplicate_rows_removed}",
        f"Conflicting unordered ID pairs: {len(conflict_groups)}",
        f"Records in conflicting pairs (all excluded): {conflict_rows}",
        f"Unique rows after conflict exclusion and deduplication: {len(clean_rows)}",
        f"Remaining negative samples: {clean_label_counts[0]}",
        f"Remaining positive samples: {clean_label_counts[1]}",
        f"Unique canonical molecules in source pairs: {len(all_molecules)}",
        f"Unique canonical molecules retained after conflict removal: {len(clean_molecules)}",
        f"Unique canonical molecules appearing in conflicted pairs: {len(conflict_molecules)}",
        f"Observed IDs with multiple raw SMILES: {len(multi_raw_smiles_ids)}",
        f"Observed IDs with multiple canonical SMILES: {len(multi_canonical_smiles_ids)}",
        f"One-to-many IDs with multiple variants within one file: {len(within_file_multi_ids)}",
        "One-to-many IDs with multiple variants per file: "
        + ", ".join(f"{split}={within_file_counts[split]}" for split in SPLITS),
        f"One-to-many IDs with variants only across different files: {len(cross_file_only_multi_ids)}",
        f"One-to-many IDs present in multiple files: {len(multi_ids_spanning_files)}",
        f"Observed canonical SMILES shared by multiple IDs: {len(shared_canonical_observed)}",
        f"Reference canonical SMILES shared by multiple IDs: {len(shared_canonical_reference)}",
        f"IDs in drug_list_miner.csv: {len(reference)}",
        f"Exact SMILES vocabulary entries in drug_list_miner.csv: {len(source_vocabulary)}",
        "IDs in drug_list_miner.csv with multiple canonical SMILES: "
        f"{sum(len(values) > 1 for values in reference.values())}",
        f"IDs inconsistent with drug_list_miner.csv: {len(mismatch_ids)}",
        "Distinct non-reference canonical ID/SMILES combinations: "
        f"{nonreference_variant_count}",
        "Occurrences of non-reference ID/SMILES assignments: "
        f"{nonreference_assignment_count}",
        f"First 20 one-to-many IDs (sorted): {', '.join(top_ids)}",
        f"Detailed canonical SMILES counts by file: {variant_file}",
        "DrugBank ID policy: audit only; preserve pair-file SMILES and do not reconstruct by ID",
        "Pre-write validation: in progress",
    ]
    write_report(report_file, report_lines)

    if missing_values or invalid_labels or invalid_smiles or vocabulary_misses:
        raise ValueError(
            f"Input validation failed; inspect {report_file}"
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

    output_file = SCRIPT_DIR / "ChChMiner.csv"
    mapping_file = SCRIPT_DIR / "drug_list_miner.csv"
    final_labels = Counter()
    final_pairs = set()
    order_violations = 0
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
            raise ValueError("Output SMILES is not present in the MIRACLE vocabulary")
        if canonical_1 > canonical_2:
            order_violations += 1
        canonical_pair = tuple(sorted((canonical_1, canonical_2)))
        if canonical_pair in final_pairs:
            raise ValueError("Duplicate canonical unordered pair before writing final CSV")
        final_pairs.add(canonical_pair)
        final_labels[row["label"]] += 1

    if order_violations:
        raise ValueError("Canonical SMILES pair order normalization failed")
    if len(output_rows) != len(clean_rows):
        raise ValueError("Unexpected row count before writing final CSV")

    report_lines[report_lines.index("Pre-write validation: in progress")] = (
        "Pre-write validation: passed"
    )
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["smiles_1", "smiles_2", "label"]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    unique_smiles = sorted(
        {smiles for row in output_rows for smiles in (row["smiles_1"], row["smiles_2"])}
    )
    if any(smiles not in source_vocabulary for smiles in unique_smiles):
        raise ValueError("Output mapping contains a SMILES absent from the source vocabulary")
    with mapping_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["DrugID", "Smiles"])
        writer.writeheader()
        writer.writerows({"DrugID": smiles, "Smiles": smiles} for smiles in unique_smiles)

    report_lines.extend(
        [
            "Final CSV written: yes",
            f"Final CSV rows: {len(output_rows)}",
            f"Final negative samples: {final_labels[0]}",
            f"Final positive samples: {final_labels[1]}",
            f"Final unique canonical unordered pairs: {len(final_pairs)}",
            f"Final normalized pair order violations: {order_violations}",
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
