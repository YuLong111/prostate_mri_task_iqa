import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def find_case_file(folder: Path, case_id: str) -> Path:
    patterns = [
        f"{case_id}.nii.gz",
    ]
    for pat in patterns:
        matches = list(folder.glob(pat))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            matches = sorted(matches, key=lambda p: (len(p.name), p.name))
            return matches[0]
    raise FileNotFoundError(f"Cannot find file for {case_id} in {folder}")


def main():
    excel_path = Path(r"data/labels/sampled_file_list.xlsx")  
    base_dir = Path(r"D:\1\杂物\学校\ucl\year3\project\project\OneDrive_1_2026-1-15")  
    out_dir = Path("data/splits")

    seed = 0
    train_frac = 0.7
    val_frac = 0.15

    dwi_dir = base_dir / "dwi"
    t2_dir = base_dir / "t2"
    mask_dir = base_dir / "prostate_mask"

    for p in (dwi_dir, t2_dir, mask_dir):
        if not p.exists():
            raise FileNotFoundError(f"folder not found: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(excel_path)

    df["case_id"] = df["case_id"].astype(str)
    df["label_bin"] = df["score"].astype(int)
    has_b = "b_value" in df.columns

    records = []
    missing = []

    for _, row in df.iterrows():
        cid = row["case_id"]
        try:
            dwi_path = find_case_file(dwi_dir, cid)
            t2_path = find_case_file(t2_dir, cid)
            mask_path = find_case_file(mask_dir, cid)
        except FileNotFoundError as e:
            missing.append(str(e))
            continue

        rec = {
            "id": cid,
            "dwi": str(dwi_path.resolve()),
            "t2": str(t2_path.resolve()),
            "mask": str(mask_path.resolve()),
            "label_bin": int(row["label_bin"]),
        }
        if has_b:
            rec["b_value"] = str(row["b_value"])

        records.append(rec)

    if missing:
        print("\nWARNING: missing files for some cases (first 10):")
        for m in missing[:10]:
            print(" -", m)
        print(f"Total missing: {len(missing)}")

    strata = (
        [f"{r['label_bin']}_{r['b_value']}" for r in records]
        if has_b else
        [str(r["label_bin"]) for r in records]
    )

    n = len(records)
    train_size = int(round(train_frac * n))
    test_frac_of_temp = (1.0 - train_frac - val_frac) / (1.0 - train_frac)

    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    idx_train, idx_temp = next(sss1.split(records, strata))

    temp_records = [records[i] for i in idx_temp]
    temp_strata = [strata[i] for i in idx_temp]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac_of_temp, random_state=seed)
    idx_val_rel, idx_test_rel = next(sss2.split(temp_records, temp_strata))

    train_records = [records[i] for i in idx_train]
    val_records = [temp_records[i] for i in idx_val_rel]
    test_records = [temp_records[i] for i in idx_test_rel]

    def report(name, recs):
        import pandas as _pd
        s = _pd.Series([r["label_bin"] for r in recs]).value_counts().sort_index()
        print(f"{name}: n={len(recs)} label_dist={s.to_dict()}")

    print("\nSplit summary:")
    report("TRAIN", train_records)
    report("VAL", val_records)
    report("TEST", test_records)

    (out_dir / "datalist_train.json").write_text(json.dumps(train_records, indent=2))
    (out_dir / "datalist_val.json").write_text(json.dumps(val_records, indent=2))
    (out_dir / "datalist_test.json").write_text(json.dumps(test_records, indent=2))

    print("\nWrote:")
    print(" -", out_dir / "datalist_train.json")
    print(" -", out_dir / "datalist_val.json")
    print(" -", out_dir / "datalist_test.json")


if __name__ == "__main__":
    main()
