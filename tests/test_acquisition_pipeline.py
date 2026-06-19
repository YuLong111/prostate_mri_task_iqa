"""Regression tests for acquisition-safe manifest and preprocessing behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from prostate_iqa.analysis.analyze_quality_vs_task import (
    _standardize_quality,
    _standardize_task,
)
from prostate_iqa.analysis.make_report_figures import _merge_segmentation_metrics
from prostate_iqa.data.build_manifest import build_manifest
from prostate_iqa.data.make_task_quality_labels import (
    _classification_records,
    apply_quality_records,
)
from prostate_iqa.data.preprocess_cases import main as preprocess_main
from prostate_iqa.data.segmentation_transforms import get_segmentation_val_transforms
from prostate_iqa.data.transforms import get_val_transforms
from prostate_iqa.evaluation.eval_segmentation import _extract_unet_state_dict
from prostate_iqa.training.train_binary_task import _prepare_items
from prostate_iqa.training.train_segmentation_task import prepare_segmentation_items
from prostate_iqa.utils.segmentation import binary_segmentation_metrics


class AcquisitionPipelineTests(unittest.TestCase):
    def test_unet_checkpoint_loader_preserves_monai_model_prefix(self) -> None:
        weight = torch.ones((2, 1, 3, 3, 3))
        extracted = _extract_unet_state_dict(
            {"model_state_dict": {"model.0.conv.weight": weight}}
        )
        self.assertEqual(list(extracted), ["model.0.conv.weight"])
        wrapped = _extract_unet_state_dict(
            {"model_state_dict": {"module.model.0.conv.weight": weight}}
        )
        self.assertEqual(list(wrapped), ["model.0.conv.weight"])

    def test_segmentation_transform_keeps_mask_as_target_not_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[2:6, 4:12, 4:12] = 1
            image_path = root / "dwi.nii.gz"
            mask_path = root / "prostate_mask.nii.gz"
            sitk.WriteImage(sitk.GetImageFromArray(image), str(image_path))
            sitk.WriteImage(sitk.GetImageFromArray(mask), str(mask_path))
            transform = get_segmentation_val_transforms(
                ["dwi"], "prostate_mask", (16, 16, 8)
            )
            result = transform(
                {"dwi": str(image_path), "prostate_mask": str(mask_path)}
            )
            self.assertEqual(tuple(result["image"].shape), (1, 16, 16, 8))
            self.assertEqual(tuple(result["label"].shape), (1, 16, 16, 8))
            self.assertEqual(set(result["label"].unique().tolist()), {0, 1})

    def test_segmentation_metrics_and_target_leakage_guard(self) -> None:
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[2:6, 2:6, 2:6] = 1
        metrics = binary_segmentation_metrics(mask, mask, (1.0, 1.0, 1.0))
        self.assertEqual(metrics, {"dice": 1.0, "iou": 1.0, "asd": 0.0, "hd95": 0.0})
        with self.assertRaisesRegex(ValueError, "Target leakage"):
            _prepare_items(
                [
                    {
                        "dwi": "dwi.nii.gz",
                        "prostate_mask": "mask.nii.gz",
                        "task_quality_bin": 1,
                        "quality_target_key": "prostate_mask",
                    }
                ],
                ["dwi", "prostate_mask"],
                "task_quality_bin",
                "test",
            )
        with self.assertRaisesRegex(ValueError, "target leakage"):
            prepare_segmentation_items(
                [{"dwi": "dwi.nii.gz", "prostate_mask": "mask.nii.gz"}],
                ["dwi", "prostate_mask"],
                "prostate_mask",
                "test",
            )

    def test_training_items_drop_uncollatable_unused_null_metadata(self) -> None:
        prepared = _prepare_items(
            [
                {
                    "patient_id": "P1",
                    "scan_id": "S1",
                    "acquisition_id": "S1__distorted",
                    "distortion_status": "distorted",
                    "dwi": "dwi.nii.gz",
                    "prostate_mask": "mask.nii.gz",
                    "pirads_ge4": 1,
                    "adc": None,
                    "site": None,
                }
            ],
            ["dwi", "prostate_mask"],
            "pirads_ge4",
            "test",
        )
        self.assertEqual(len(prepared), 1)
        self.assertNotIn("adc", prepared[0])
        self.assertNotIn("site", prepared[0])
        self.assertEqual(prepared[0]["label"], 1)

    def test_validation_transform_uses_complete_3d_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
            mask = np.zeros_like(image, dtype=np.uint8)
            mask[:, 4:12, 4:12] = 1
            image_path = root / "dwi.nii.gz"
            mask_path = root / "mask.nii.gz"
            sitk.WriteImage(sitk.GetImageFromArray(image), str(image_path))
            sitk.WriteImage(sitk.GetImageFromArray(mask), str(mask_path))

            transform = get_val_transforms(
                ["dwi", "prostate_mask"], (16, 16, 8)
            )
            result = transform(
                {
                    "dwi": str(image_path),
                    "prostate_mask": str(mask_path),
                    "label": 1,
                }
            )

            self.assertEqual(tuple(result["image"].shape), (2, 16, 16, 8))
            self.assertEqual(result["label"], 1)
            self.assertGreater(float(result["image"][0, :, :, -1].sum()), 0.0)

    def test_manifest_keeps_distorted_and_undistorted_volumes_separate(self) -> None:
        rows = []
        for status in ("distorted", "undistorted"):
            for reference in ("ref01", "ref02"):
                rows.append(
                    {
                        "file_path": f"C:/dataset/{status}/DWI/patient001_{reference}.nii.gz",
                        "patient_id_guess": "patient001",
                        "scan_id_guess": reference,
                        "modality_guess": "dwi",
                        "distortion_status": status,
                    }
                )
        rows.extend(
            [
                {
                    "file_path": "C:/dataset/masks/patient001_prostate.nii.gz",
                    "patient_id_guess": "patient001",
                    "scan_id_guess": "",
                    "modality_guess": "prostate_mask",
                    "distortion_status": "not_applicable",
                },
                {
                    "file_path": "C:/dataset/distorted/DWI/patient001_ref01.json",
                    "patient_id_guess": "patient001",
                    "scan_id_guess": "ref01",
                    "modality_guess": "dwi",
                    "distortion_status": "distorted",
                },
            ]
        )

        manifest = build_manifest(pd.DataFrame(rows))

        self.assertEqual(len(manifest), 4)
        self.assertEqual(set(manifest["distortion_status"]), {"distorted", "undistorted"})
        self.assertEqual(manifest["acquisition_id"].nunique(), 4)
        self.assertTrue(manifest["prostate_mask"].str.endswith(".nii.gz").all())
        self.assertFalse(manifest["dwi"].str.contains(";").any())
        self.assertFalse(manifest["dwi"].str.endswith(".json").any())

    def test_quality_labels_match_acquisition_not_only_scan(self) -> None:
        manifest = pd.DataFrame(
            [
                {"patient_id": "P1", "scan_id": "S1", "acquisition_id": "S1__distorted"},
                {"patient_id": "P1", "scan_id": "S1", "acquisition_id": "S1__undistorted"},
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "scan_id": "S1",
                    "acquisition_id": "S1__distorted",
                    "correct": 0,
                    "confidence": 0.9,
                },
                {
                    "patient_id": "P1",
                    "scan_id": "S1",
                    "acquisition_id": "S1__undistorted",
                    "correct": 1,
                    "confidence": 0.9,
                },
            ]
        )
        records = _classification_records(predictions, "pirads", 0.7, 0.7)
        labeled = apply_quality_records(manifest, records)
        self.assertEqual(labeled["task_quality_ternary"].tolist(), [0, 2])

        quality = _standardize_quality(
            predictions.assign(pred_quality=[0, 2], expected_quality_score=[0.1, 1.9])
        )
        task = _standardize_task(
            predictions.assign(true_label=[1, 1], pred_label=[0, 1], prob_1=[0.1, 0.9])
        )
        merged = quality.merge(
            task,
            on=["_patient_key", "_scan_key", "_acquisition_key"],
            validate="one_to_one",
        )
        self.assertEqual(len(merged), 2)

        segmentation = pd.DataFrame(
            [
                {
                    "patient_id": "P1",
                    "scan_id": "S1",
                    "acquisition_id": "S1__distorted",
                    "dice": 0.4,
                },
                {
                    "patient_id": "P1",
                    "scan_id": "S1",
                    "acquisition_id": "S1__undistorted",
                    "dice": 0.9,
                },
            ]
        )
        report_table = _merge_segmentation_metrics(
            predictions.assign(pred_quality=[0, 2]), segmentation
        )
        self.assertEqual(report_table["seg_dice"].tolist(), [0.4, 0.9])

    def test_preprocessing_writes_one_directory_and_json_row_per_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask_array = np.zeros((8, 16, 16), dtype=np.uint8)
            mask_array[2:6, 4:12, 4:12] = 1
            mask_path = root / "mask.nii.gz"
            sitk.WriteImage(sitk.GetImageFromArray(mask_array), str(mask_path))

            cases = []
            for index, status in enumerate(("distorted", "undistorted"), start=1):
                image_array = np.full((8, 16, 16), index, dtype=np.float32)
                image_path = root / f"{status}.nii.gz"
                sitk.WriteImage(sitk.GetImageFromArray(image_array), str(image_path))
                cases.append(
                    {
                        "patient_id": "P1",
                        "scan_id": "S1",
                        "distortion_status": status,
                        "acquisition_id": f"S1__{status}",
                        "split": "train",
                        "dwi": str(image_path),
                        "prostate_mask": str(mask_path),
                        "pirads_ge4": 1,
                    }
                )

            datalist = root / "datalist_train.json"
            datalist.write_text(json.dumps(cases), encoding="utf-8")
            out_dir = root / "processed"
            exit_code = preprocess_main(
                [
                    "--datalist_json",
                    str(datalist),
                    "--out_dir",
                    str(out_dir),
                    "--spacing",
                    "1",
                    "1",
                    "1",
                    "--roi_size",
                    "16",
                    "16",
                    "8",
                ]
            )

            self.assertEqual(exit_code, 0)
            manifest = pd.read_csv(out_dir / "preprocessing_manifest.csv")
            self.assertEqual(len(manifest), 2)
            self.assertEqual(manifest["case_dir"].nunique(), 2)
            self.assertTrue(manifest["dwi"].map(lambda value: Path(value).is_file()).all())
            processed_json = json.loads(
                (out_dir / "datalist_train.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(processed_json), 2)
            self.assertEqual({row["pirads_ge4"] for row in processed_json}, {1})


if __name__ == "__main__":
    unittest.main()
