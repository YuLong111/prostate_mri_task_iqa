# Prostate MRI task-specific IQA workflow

This project defines image quality by whether an image supports a downstream clinical task. The principal unit is one **physical 3D acquisition**, not one axial slice. A patient can therefore contribute several rows (for example distorted and undistorted high-b DWI volumes), but every row for that patient must remain in the same dataset split.

## 0. Start a PowerShell session

```powershell
Set-Location "D:\Projects\prostate_mri_task_iqa"
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Resolve-Path ".\src").Path

# Only needed when creating/updating the environment:
python -m pip install -r requirements.txt
```

Run commands from the project root. All scripts support `--help`.

## 1. Configure local paths

Edit `configs/paths.local.yaml` on each computer. This file is the only place that should contain the local OneDrive path.

```yaml
dataset_root: "C:/Users/<user>/OneDrive - <organisation>/dataset"
processed_root: "data/processed"
manifest_dir: "data/manifests"
split_dir: "data/splits"
report_dir: "reports"
run_dir: "runs"
```

Role: separates machine-specific storage from reproducible code. Opening a cloud-only NIfTI during QC or preprocessing can make OneDrive hydrate it into its normal synced path; the project does not copy it to a hidden project cache.

## 2. Build the file inventory

```powershell
python -m prostate_iqa.data.build_inventory `
  --out_csv data/manifests/file_inventory.csv
```

Role: performs a read-only recursive scan and records every supported file, its guessed patient/scan identifiers, modality, and distortion status. It recognizes the distorted/undistorted folder structure and reports counts. It does not open or modify image contents.

Output: `data/manifests/file_inventory.csv`.

Check the modality and suffix counts printed by the command. Incorrect guesses should be corrected in the inventory or naming logic before proceeding.

## 3. Build the acquisition-level master manifest

For a new corrected experiment, use versioned outputs rather than replacing an experimental locked split:

```powershell
python -m prostate_iqa.data.build_manifest `
  --inventory_csv data/manifests/file_inventory.csv `
  --out_csv data/manifests/master_manifest_acquisition_v2.csv
```

Add `--labels_csv <clinical.csv-or-xlsx>` when clinical labels are available.

Role: connects T2, DWI, ADC, prostate masks, lesion masks, and clinical labels. Distorted and undistorted volumes stay on separate rows. Multiple DWI volumes become separate acquisition rows with unique `acquisition_id` values. A single patient mask may be reused by all matching acquisitions. JSON/CSV sidecars are excluded from image columns.

Outputs:

- master manifest CSV;
- `manifest_missingness.csv`;
- `modality_summary.csv`.

Important invariant: no image field should contain semicolon-separated paths. A 4D NIfTI is not silently reduced to frame zero; split it into explicit 3D acquisitions first.

## 4. Inspect inventory and manifest summaries

Review missingness before splitting. Confirm that acquisition counts reflect the dataset design: masks can legitimately be fewer than DWI volumes because one mask is shared across several acquisitions.

Role: catches naming/matching failures before they become model errors. In particular, inspect cases marked `not_applicable` or missing DWI/T2.

## 5. Create patient-level splits once

```powershell
$SplitDir = "data/splits_acquisition_v2"

python -m prostate_iqa.data.split_patient_level `
  --manifest_csv data/manifests/master_manifest_acquisition_v2.csv `
  --out_dir $SplitDir `
  --test_size 0.15 `
  --val_size 0.15 `
  --seed 42 `
  --stratify_key pirads_ge4
```

Omit `--stratify_key` if that label is unavailable. Sparse stratification automatically falls back to a random patient-level split with a warning.

Role: prevents patient leakage. All acquisitions, distortion variants, and masks belonging to one patient are assigned together. The command writes:

- `datalist_train.json`;
- `datalist_val.json`;
- `datalist_test_locked.json`;
- `split_summary.csv`.

The command refuses to overwrite an existing locked test JSON. Use a versioned output directory for a genuinely new experiment. `--overwrite_locked_test` is deliberately explicit and should only be used before model development begins.

## 6. Perform visual and metadata QC

```powershell
python -m prostate_iqa.data.nifti_qc `
  --datalist_json "$SplitDir/datalist_train.json" `
  --out_dir reports/qc/train `
  --max_cases 50

python -m prostate_iqa.data.nifti_qc `
  --datalist_json "$SplitDir/datalist_val.json" `
  --out_dir reports/qc/val
```

Role: visually checks orientation, intensity, masks, and approximate T2–DWI correspondence without modifying data. Contact sheets sample representative axial slices for human viewing, but this sampling is only a QC display. Per-case JSON stores the full shape, spacing, affine, intensity statistics, mask size, missing modalities, and warnings.

Do not inspect the locked test set to choose preprocessing settings.

## 7. Preprocess complete 3D volumes

Choose spacing and ROI size using training/validation data, then keep them fixed.

```powershell
$ProcessedDir = "data/processed_acquisition_v2"

python -m prostate_iqa.data.preprocess_cases `
  --datalist_json "$SplitDir/datalist_train.json" `
  --out_dir $ProcessedDir `
  --spacing 0.5 0.5 1.0 `
  --roi_size 160 160 64

python -m prostate_iqa.data.preprocess_cases `
  --datalist_json "$SplitDir/datalist_val.json" `
  --out_dir $ProcessedDir `
  --spacing 0.5 0.5 1.0 `
  --roi_size 160 160 64
```

Role: loads each full 3D volume, aligns DWI/ADC/masks to the reference grid, resamples, crops around the prostate (or center-crops), robustly normalizes intensity images, and writes new NIfTIs. Masks use nearest-neighbor interpolation. Raw images are never changed.

Outputs include a cumulative `preprocessing_manifest.csv`, `preprocessing_failures.csv`, per-case summaries, and ready-to-train `datalist_train.json` / `datalist_val.json`. Distortion variants have distinct output directories.

Only after every choice is frozen should the same command be applied to `datalist_test_locked.json`.

## 8. Train the downstream clinical task

Example for PI-RADS >= 4:

```powershell
python -m prostate_iqa.training.train_binary_task `
  --train_json "$ProcessedDir/datalist_train.json" `
  --val_json "$ProcessedDir/datalist_val.json" `
  --image_keys dwi adc t2 prostate_mask `
  --target_key pirads_ge4 `
  --roi_size 160 160 64 `
  --out_dir runs/downstream_pirads_v2 `
  --epochs 100 `
  --batch_size 2 `
  --lr 0.0001 `
  --seed 42
```

Use `gleason_ge2` for the Gleason task. Image keys may be reduced to modalities actually present.

Role: establishes the task whose success/failure will define quality. It trains a 3D DenseNet121 with class balancing and selects `best.pt` by validation AUC. It also writes `val_predictions.csv` and complete binary metrics. The locked test set is rejected by training code.

## 9. Generate leakage-safe training quality targets with OOF prediction

```powershell
python -m prostate_iqa.training.run_oof_downstream `
  --train_json "$ProcessedDir/datalist_train.json" `
  --image_keys dwi adc t2 prostate_mask `
  --target_key pirads_ge4 `
  --num_folds 5 `
  --roi_size 160 160 64 `
  --out_dir runs/oof_pirads_v2 `
  --epochs 100 `
  --batch_size 2 `
  --lr 0.0001 `
  --seed 42
```

Role: ensures that each training acquisition is scored by a downstream model that did not train on that patient. This avoids turning memorized training success into an artificial quality label.

Outputs: per-fold checkpoints/predictions and `oof_predictions.csv` with one held-out prediction per training row.

## 10. Convert downstream behavior into quality labels

Training labels come from OOF predictions; validation labels come from the ordinary downstream model's validation predictions.

```powershell
python -m prostate_iqa.data.make_task_quality_labels `
  --manifest_csv "$ProcessedDir/preprocessing_manifest.csv" `
  --task_predictions_csv runs/oof_pirads_v2/oof_predictions.csv `
  --task_name pirads_ge4 `
  --split train `
  --out_csv data/manifests/pirads_quality_train_v2.csv `
  --accept_confidence 0.70 `
  --reject_confidence 0.70

python -m prostate_iqa.data.make_task_quality_labels `
  --manifest_csv "$ProcessedDir/preprocessing_manifest.csv" `
  --task_predictions_csv runs/downstream_pirads_v2/val_predictions.csv `
  --task_name pirads_ge4 `
  --split val `
  --out_csv data/manifests/pirads_quality_val_v2.csv `
  --accept_confidence 0.70 `
  --reject_confidence 0.70
```

Role: assigns reject (confident failure), caution (uncertain/mixed result), or accept (confident success). `task_quality_bin` keeps accept/reject and leaves caution null. Original clinical/direct quality labels are preserved. The script writes both CSV and same-stem JSON files; the JSON files feed IQA training directly.

Thresholds are development choices: tune them with training/validation only, never the locked test set.

For segmentation, provide `--segmentation_metrics_csv` instead. Dice and 95HD then determine accept/caution/reject.

## 11. Train IQA models

Binary baseline (caution rows are excluded because the binary target is null):

```powershell
python -m prostate_iqa.training.train_binary_iqa `
  --train_json data/manifests/pirads_quality_train_v2.json `
  --val_json data/manifests/pirads_quality_val_v2.json `
  --image_keys dwi adc t2 prostate_mask `
  --target_key task_quality_bin `
  --out_dir runs/binary_iqa_pirads_v2 `
  --epochs 100 `
  --batch_size 2 `
  --lr 0.0001 `
  --seed 42
```

Ternary reject/caution/accept model:

```powershell
python -m prostate_iqa.training.train_ternary_iqa `
  --train_json data/manifests/pirads_quality_train_v2.json `
  --val_json data/manifests/pirads_quality_val_v2.json `
  --image_keys dwi adc t2 prostate_mask `
  --target_key task_quality_ternary `
  --roi_size 160 160 64 `
  --out_dir runs/ternary_iqa_pirads_v2 `
  --epochs 100 `
  --batch_size 2 `
  --lr 0.0001 `
  --seed 42
```

Role: learns to predict whether a scan should be rejected, treated cautiously, or accepted for the selected downstream task. The ternary loss combines cross entropy with ordinal MAE, so reject-to-accept mistakes cost more than adjacent mistakes.

## 12. Evaluate a frozen model

Validation example:

```powershell
python -m prostate_iqa.evaluation.eval_model `
  --ckpt runs/ternary_iqa_pirads_v2/best.pt `
  --datalist_json data/manifests/pirads_quality_val_v2.json `
  --image_keys dwi adc t2 prostate_mask `
  --target_key task_quality_ternary `
  --num_classes 3 `
  --out_csv reports/evaluation/ternary_iqa_val_predictions.csv `
  --out_metrics_json reports/evaluation/ternary_iqa_val_metrics.json
```

Role: calculates binary or ternary metrics and saves acquisition-level probabilities, confidence, entropy, ordinal expectation, and metadata. Acquisition identity is preserved so distorted and undistorted variants cannot be cross-matched.

Final locked-test evaluation occurs once, after architecture, preprocessing, thresholds, and checkpoint selection are frozen.

## 13. Estimate novelty / out-of-distribution behavior

```powershell
python -m prostate_iqa.evaluation.eval_novelty `
  --ckpt runs/ternary_iqa_pirads_v2/best.pt `
  --train_json data/manifests/pirads_quality_train_v2.json `
  --eval_json data/manifests/pirads_quality_val_v2.json `
  --image_keys dwi adc t2 prostate_mask `
  --target_key task_quality_ternary `
  --num_classes 3 `
  --out_csv reports/evaluation/novelty_val.csv
```

Role: fits a regularized Gaussian reference distribution to training penultimate features and reports Mahalanobis distance for evaluation cases, alongside softmax confidence and entropy. Train/evaluation patient overlap is rejected.

## 14. Test whether predicted quality explains task performance

```powershell
python -m prostate_iqa.analysis.analyze_quality_vs_task `
  --quality_predictions_csv reports/evaluation/ternary_iqa_val_predictions.csv `
  --task_predictions_csv runs/downstream_pirads_v2/val_predictions.csv `
  --novelty_csv reports/evaluation/novelty_val.csv `
  --out_dir reports/analysis/pirads_val
```

Role: this is the central scientific analysis. It asks whether downstream accuracy improves from reject to caution to accept, whether expected quality correlates with downstream success/confidence, and whether novelty predicts failure. Optional segmentation metrics add Dice/IoU/ASD/95HD comparisons.

Outputs include merged case data, quality-group task summaries, novelty/failure summaries, and simple plots.

## 15. Generate report figures

```powershell
python -m prostate_iqa.analysis.make_report_figures `
  --quality_predictions_csv reports/evaluation/ternary_iqa_val_predictions.csv `
  --novelty_csv reports/evaluation/novelty_val.csv `
  --quality_task_summary_csv reports/analysis/pirads_val/quality_group_task_summary.csv `
  --out_dir reports/figures
```

Role: produces publication/report-ready confusion matrices, ROC curves when applicable, class distributions, confidence/entropy/novelty histograms, and task-performance plots using matplotlib only.

## What uses all slices?

- Inventory/manifest: record file paths only; they do not select slices.
- QC contact sheets: display a representative subset of axial slices for readability; metadata covers the complete volume.
- Preprocessing: resamples and crops the complete 3D volume to the requested ROI.
- MONAI transforms and DenseNet models: load and process the full 3D tensor (for example all `160 x 160 x 64` voxels), not only the central slice.
- A genuine 4D NIfTI is rejected rather than silently choosing one time/b-value frame. Convert it into explicit 3D acquisition files before running the workflow.

## Reproducibility and leakage rules

1. Keep every patient entirely within one split.
2. Use OOF downstream predictions for training-derived IQA labels.
3. Use validation data for model selection and threshold choices.
4. Never inspect the locked test set for preprocessing or threshold decisions.
5. Freeze the downstream model, IQA model, parameters, and quality thresholds before final test evaluation.
6. Preserve `patient_id`, `scan_id`, `distortion_status`, and `acquisition_id` in every table and merge.
7. Treat the task name as part of the quality definition: a scan may be acceptable for one downstream task and unsuitable for another.
