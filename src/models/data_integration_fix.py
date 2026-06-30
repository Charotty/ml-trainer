#!/usr/bin/env python3
"""
Интеграция источников данных в Phase 1 train/validation CSV.

Канонические входы (по умолчанию):
  - data/vybor_from_xlsx.csv                   — клинические пары supine/lateral из основного xlsx
  - data/train_displacement_dataset.csv        — legacy поднабор 50 (не используется, если есть xlsx CSV)
  - data/dicom_medical_features.csv            — опционально, без таргетов
  - data/kits19_medical_grade_features.csv     — только reference-признаки (без таргетов в train)

Режимы training_mode:
  - labeled_only (по умолчанию): train/val только Vybor; KiTS19/DICOM не в y
  - all: legacy — все источники с полными таргетами в train (включая KiTS proxy deltas)
  - harmonized_extended: Vybor (clinical) + гармонизированные KiTS19 (proxy δ) +
    DICOM с teacher pseudo-δ; val только Vybor
  - clinical_xlsx_extended: как harmonized_extended + sample_weight (clinical=1, KiTS=0.15, DICOM=0.08)
  - clinical_xlsx_kits_impute_only: Vybor + DICOM pseudo в train; KiTS19 только median imputation
  - proxy_weighted_extended: Vybor + KiTS19 proxy δ + DICOM pseudo-δ в train (weighted); val только Vybor

Выход:
  - data/integrated_master_dataset.csv
  - data/processed/train.csv, validation.csv
  - data/processed/validation_clinical.csv   — holdout для честного аудита
  - data/processed/integration_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.excel_displacement_adapter import (  # noqa: E402
    DEFAULT_EXCEL_PATH,
    load_excel_displacement_table,
)
from src.features.phase1_schema import BASE_FEATURES, TARGET_NAMES, normalize_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DICOM_PATH = REPO_ROOT / "data" / "dicom_medical_features.csv"
DEFAULT_VYBOR_PATH = REPO_ROOT / "data" / "vybor_from_xlsx.csv"
DEFAULT_EXCEL_PATH_RESOLVED = REPO_ROOT / DEFAULT_EXCEL_PATH
DEFAULT_KITS19_PATH = REPO_ROOT / "data" / "kits19_medical_grade_features.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

LABELED_SOURCES = {"Vybor", "Excel"}
# KiTS19/DICOM never provide paired supine+lateral displacement labels.
PROXY_TARGET_SOURCES = frozenset({"KiTS19", "DICOMS"})
EXTENDED_TRAIN_SOURCES = {"Vybor", "Excel", "KiTS19", "DICOMS"}
EXTENDED_MODES = frozenset({"harmonized_extended", "clinical_xlsx_extended", "clinical_xlsx_kits_impute_only"})
PROXY_TRAIN_MODES = frozenset({"proxy_weighted_extended"})
KITS_IMPUTE_ONLY_MODES = frozenset({"clinical_xlsx_kits_impute_only"})
DEPRECATED_EXTENDED_MODES = EXTENDED_MODES | frozenset({"all"})
SAMPLE_WEIGHT_BY_QUALITY = {
    "clinical": 1.0,
    "proxy_kits": 0.08,
    "pseudo_dicom": 0.06,
}
KITS_MEDIANS_PATH = PROCESSED_DIR / "kits19_feature_medians.json"


class DataIntegrationFix:
    def __init__(
        self,
        dicom_path: Path | str | None = None,
        vybor_path: Path | str | None = None,
        excel_path: Path | str | None = None,
        kits19_path: Path | str | None = DEFAULT_KITS19_PATH,
        training_mode: str = "labeled_only",
        fill_sparse_from_kits: bool = True,
    ):
        self.dicom_path = Path(dicom_path) if dicom_path else DEFAULT_DICOM_PATH
        self.vybor_path = Path(vybor_path) if vybor_path else DEFAULT_VYBOR_PATH
        self.excel_path = Path(excel_path) if excel_path else None
        self.kits19_path = Path(kits19_path) if kits19_path else None
        self.training_mode = training_mode
        self.fill_sparse_from_kits = fill_sparse_from_kits

        self.dicoms_df: Optional[pd.DataFrame] = None
        self.vybor_df: Optional[pd.DataFrame] = None
        self.excel_df: Optional[pd.DataFrame] = None
        self.kits19_df: Optional[pd.DataFrame] = None
        self.kits_feature_reference: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Загрузить доступные источники."""
        logger.info("Загрузка данных (mode=%s)...", self.training_mode)
        logger.info("  Vybor: %s", self.vybor_path)
        logger.info("  Excel: %s", self.excel_path)
        logger.info("  DICOM: %s", self.dicom_path)
        if self.kits19_path:
            logger.info("  KiTS19 (features reference): %s", self.kits19_path)

        if not self.vybor_path.exists():
            raise FileNotFoundError(
                f"Required Vybor dataset not found: {self.vybor_path}"
            )

        self.vybor_df = pd.read_csv(self.vybor_path)

        if self.excel_path and self.excel_path.exists():
            self.excel_df = load_excel_displacement_table(
                str(self.excel_path),
                vybor_df=self.vybor_df,
            )
            logger.info("Excel (unique vs Vybor): %s rows", len(self.excel_df))
        else:
            self.excel_df = pd.DataFrame()
            if self.excel_path:
                logger.warning("Excel file missing, skipping: %s", self.excel_path)

        if self.dicom_path.exists():
            self.dicoms_df = pd.read_csv(self.dicom_path)
            logger.info("DICOM: %s rows", len(self.dicoms_df))
        else:
            self.dicoms_df = pd.DataFrame()
            logger.warning("DICOM file missing, skipping: %s", self.dicom_path)

        if self.kits19_path and self.kits19_path.exists():
            kits_raw = normalize_dataframe(pd.read_csv(self.kits19_path))
            self.kits19_df = kits_raw
            base_cols = [c for c in BASE_FEATURES if c in kits_raw.columns]
            self.kits_feature_reference = kits_raw[base_cols].copy()
            logger.info(
                "KiTS19: %s rows (reference features only, targets excluded from train in labeled_only)",
                len(kits_raw),
            )
        else:
            self.kits19_df = None
            self.kits_feature_reference = None
            if self.kits19_path:
                logger.warning("KiTS19 file missing, skipping: %s", self.kits19_path)

        logger.info("Vybor: %s rows", len(self.vybor_df))

    def create_universal_keys(self) -> None:
        """Создать universal_id для каждого источника."""
        logger.info("Создание универсальных ключей...")

        if self.dicoms_df is not None and len(self.dicoms_df) > 0:
            self.dicoms_df = self.dicoms_df.copy()
            self.dicoms_df["source_id"] = range(1, len(self.dicoms_df) + 1)
            self.dicoms_df["source_name"] = "DICOMS"
            self.dicoms_df["universal_id"] = [
                f"DICOMS_{i}" for i in range(1, len(self.dicoms_df) + 1)
            ]

        self.vybor_df = self.vybor_df.copy()
        self.vybor_df["source_id"] = range(1, len(self.vybor_df) + 1)
        self.vybor_df["source_name"] = "Vybor"
        self.vybor_df["universal_id"] = [
            f"Vybor_{str(i).zfill(3)}" for i in range(1, len(self.vybor_df) + 1)
        ]

        if self.excel_df is not None and len(self.excel_df) > 0:
            self.excel_df = self.excel_df.copy()
            self.excel_df["source_id"] = range(1, len(self.excel_df) + 1)
            self.excel_df["source_name"] = "Excel"
            if "case_id" not in self.excel_df.columns:
                self.excel_df["case_id"] = [
                    f"excel_{i}" for i in range(1, len(self.excel_df) + 1)
                ]
            self.excel_df["universal_id"] = self.excel_df["case_id"]

        if self.kits19_df is not None and len(self.kits19_df) > 0:
            self.kits19_df = self.kits19_df.copy()
            self.kits19_df["source_id"] = range(1, len(self.kits19_df) + 1)
            self.kits19_df["source_name"] = "KiTS19"
            self.kits19_df["universal_id"] = [
                f"KiTS19_{str(i).zfill(5)}" for i in range(1, len(self.kits19_df) + 1)
            ]

    def _apply_kits_median_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """Заполнить пропуски в BASE_FEATURES медианами KiTS19 (анатомия, не displacement)."""
        if not self.fill_sparse_from_kits or self.kits_feature_reference is None:
            return df

        medians_path = KITS_MEDIANS_PATH
        medians: dict = {}
        if medians_path.exists():
            medians = json.loads(medians_path.read_text(encoding="utf-8"))
        else:
            for col in BASE_FEATURES:
                if col in self.kits_feature_reference.columns:
                    medians[col] = float(self.kits_feature_reference[col].median(skipna=True))

        out = df.copy()
        filled_cols = []
        for col in BASE_FEATURES:
            if col not in out.columns or col not in medians:
                continue
            missing = out[col].isna()
            if missing.any():
                out.loc[missing, col] = medians[col]
                filled_cols.append(col)
        if filled_cols:
            logger.info(
                "Filled sparse BASE_FEATURES from KiTS medians: %s",
                ", ".join(filled_cols[:8]) + ("..." if len(filled_cols) > 8 else ""),
            )
        return out

    @staticmethod
    def strip_proxy_displacement_targets(df: pd.DataFrame) -> pd.DataFrame:
        """Remove fake/proxy delta columns from KiTS19 and DICOM rows."""
        if "source" not in df.columns:
            return df
        out = df.copy()
        mask = out["source"].isin(PROXY_TARGET_SOURCES)
        if not mask.any():
            return out
        present = [c for c in TARGET_NAMES if c in out.columns]
        for col in present:
            out.loc[mask, col] = np.nan
        logger.info(
            "Stripped proxy displacement targets from %s KiTS19/DICOM rows "
            "(features kept for reference/imputation only)",
            int(mask.sum()),
        )
        return out

    def normalize_dicoms_features(self) -> pd.DataFrame:
        """Привести DICOM-таблицу к canonical schema (без синтетических таргетов)."""
        if self.dicoms_df is None or len(self.dicoms_df) == 0:
            return pd.DataFrame()

        df = normalize_dataframe(self.dicoms_df.copy())
        if "scan_position" not in df.columns and "patient_position" in df.columns:
            df["scan_position"] = df["patient_position"]

        usable = int(df[TARGET_NAMES].notna().all(axis=1).sum()) if all(
            c in df.columns for c in TARGET_NAMES
        ) else 0
        if usable == 0:
            logger.warning(
                "DICOM table has no rows with complete displacement targets; "
                "rows kept in master only."
            )
        return df

    def normalize_features(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Нормализовать все источники."""
        logger.info("Нормализация признаков (phase1_schema)...")
        dicoms_normalized = self.normalize_dicoms_features()
        vybor_normalized = normalize_dataframe(self.vybor_df.copy())
        excel_normalized = (
            self._apply_kits_median_fill(normalize_dataframe(self.excel_df.copy()))
            if self.excel_df is not None and len(self.excel_df) > 0
            else pd.DataFrame()
        )
        kits19_normalized = (
            normalize_dataframe(self.kits19_df.copy())
            if self.kits19_df is not None
            else None
        )
        return dicoms_normalized, vybor_normalized, excel_normalized, kits19_normalized

    def create_master_dataset(self) -> pd.DataFrame:
        """Объединить источники в master CSV."""
        logger.info("Создание мастер-датасета...")
        dicoms_norm, vybor_norm, excel_norm, kits19_norm = self.normalize_features()

        parts: List[pd.DataFrame] = []
        if len(dicoms_norm) > 0:
            dicom_part = dicoms_norm.assign(source="DICOMS")
            if "label_quality" not in dicom_part.columns:
                dicom_part["label_quality"] = "pseudo_dicom"
            parts.append(dicom_part)
        vybor_part = vybor_norm.assign(source="Vybor", label_quality="clinical")
        parts.append(vybor_part)
        if len(excel_norm) > 0:
            parts.append(excel_norm.assign(source="Excel", label_quality="clinical"))
        if kits19_norm is not None and len(kits19_norm) > 0:
            kits_part = kits19_norm.assign(source="KiTS19")
            if "label_quality" not in kits_part.columns:
                kits_part["label_quality"] = "proxy_kits"
            parts.append(kits_part)

        master_df = pd.concat(parts, ignore_index=True)
        master_df = normalize_dataframe(master_df)
        if self.training_mode not in PROXY_TRAIN_MODES:
            master_df = self.strip_proxy_displacement_targets(master_df)
        else:
            logger.info(
                "proxy_weighted_extended: keeping KiTS19/DICOM displacement columns in master"
            )
        master_df = master_df.sort_values(["source", "source_id"], na_position="last")

        logger.info(
            "Мастер-датасет: %s строк, источники: %s",
            len(master_df),
            master_df["source"].unique().tolist(),
        )
        return master_df

    @staticmethod
    def _filter_trainable_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Оставить только строки с полным набором таргетов."""
        present_targets = [c for c in TARGET_NAMES if c in df.columns]
        if len(present_targets) != len(TARGET_NAMES):
            missing = set(TARGET_NAMES) - set(present_targets)
            raise ValueError(f"Dataset missing target columns: {sorted(missing)}")
        before = len(df)
        out = df.dropna(subset=present_targets, how="any").copy()
        skipped = before - len(out)
        if skipped:
            logger.warning(
                "Excluded %s rows without complete targets (kept %s trainable rows)",
                skipped,
                len(out),
            )
        return out

    def _select_training_rows(self, trainable: pd.DataFrame) -> pd.DataFrame:
        if self.training_mode in PROXY_TRAIN_MODES:
            logger.info(
                "%s: %s trainable rows (clinical + KiTS proxy + DICOM pseudo)",
                self.training_mode,
                len(trainable),
            )
            return trainable.copy()

        if self.training_mode in DEPRECATED_EXTENDED_MODES - {"all"}:
            logger.warning(
                "training_mode=%s is deprecated for regression: "
                "KiTS19/DICOM proxy deltas are excluded (clinical labels only).",
                self.training_mode,
            )

        # Step 1 audit fix: regression targets only from paired clinical sources.
        if self.training_mode != "all":
            mask = trainable["source"].isin(LABELED_SOURCES)
            selected = trainable[mask].copy()
            excluded = len(trainable) - len(selected)
            if excluded:
                logger.info(
                    "%s: excluded %s non-clinical rows from train/val "
                    "(KiTS19/DICOM not used as regression targets)",
                    self.training_mode,
                    excluded,
                )
            return selected

        mask = trainable["source"].isin(LABELED_SOURCES)
        selected = trainable[mask].copy()
        logger.warning(
            "legacy mode=all: KiTS19/DICOM proxy targets stripped; "
            "using %s clinical rows only",
            len(selected),
        )
        return selected

    @staticmethod
    def _attach_sample_weights(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "label_quality" in out.columns:
            out["sample_weight"] = out["label_quality"].map(SAMPLE_WEIGHT_BY_QUALITY).fillna(0.1)
        elif "source" in out.columns:
            source_weights = {
                "Vybor": 1.0,
                "Excel": 1.0,
                "KiTS19": 0.15,
                "DICOMS": 0.08,
            }
            out["sample_weight"] = out["source"].map(source_weights).fillna(0.1)
        else:
            out["sample_weight"] = 1.0
        return out

    def _split_harmonized_extended(
        self, training_rows: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Clinical Vybor holdout for val; KiTS19 + pseudo-DICOM only in train."""
        clinical = training_rows[training_rows["source"].isin(LABELED_SOURCES)].copy()
        extra = training_rows[~training_rows["source"].isin(LABELED_SOURCES)].copy()
        if len(clinical) < 5:
            raise ValueError(
                f"harmonized_extended needs >=5 Vybor rows, got {len(clinical)}"
            )
        clin_train, val_df = self._split_clinical_labeled(clinical, test_size=test_size)
        train_df = normalize_dataframe(
            pd.concat([clin_train, extra], ignore_index=True)
        )
        val_df = normalize_dataframe(val_df)
        logger.info(
            "harmonized_extended split: train=%s (extra=%s), val Vybor=%s",
            len(train_df),
            len(extra),
            len(val_df),
        )
        return train_df, val_df

    def analyze_data_quality(self, df: pd.DataFrame) -> dict:
        """Анализ пропусков."""
        logger.info("Анализ качества данных...")
        missing_analysis = {}
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            if missing_count > 0:
                missing_analysis[col] = {
                    "count": missing_count,
                    "percentage": (missing_count / len(df)) * 100,
                }

        sorted_missing = sorted(
            missing_analysis.items(),
            key=lambda x: x[1]["percentage"],
            reverse=True,
        )
        logger.info("Топ-10 колонок с пропусками:")
        for col, stats in sorted_missing[:10]:
            logger.info("  %s: %.1f%% (%s строк)", col, stats["percentage"], stats["count"])
        return missing_analysis

    def _split_clinical_labeled(
        self, labeled_df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Patient-level split для клинических источников (Vybor + Excel)."""
        if len(labeled_df) < 5:
            raise ValueError(
                f"Too few labeled clinical rows ({len(labeled_df)}). "
                "Need Vybor and/or unique Excel patients."
            )

        if len(labeled_df) >= 10:
            train_df, val_df = train_test_split(
                labeled_df, test_size=test_size, random_state=42
            )
        else:
            # Мало строк — фиксированный holdout по case_id для воспроизводимости
            sorted_df = labeled_df.sort_values("case_id").reset_index(drop=True)
            n_val = max(1, int(round(len(sorted_df) * test_size)))
            val_df = sorted_df.iloc[:n_val]
            train_df = sorted_df.iloc[n_val:]
            if len(train_df) < 3:
                train_df, val_df = train_test_split(
                    labeled_df, test_size=test_size, random_state=42
                )

        return (
            normalize_dataframe(train_df),
            normalize_dataframe(val_df),
        )

    def save_integrated_data(self, master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Сохранить master + train/validation split."""
        logger.info("Сохранение интегрированных данных...")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        master_path = (
            REPO_ROOT / "data" / "integrated_master_dataset_harmonized.csv"
            if self.training_mode in EXTENDED_MODES | PROXY_TRAIN_MODES
            else REPO_ROOT / "data" / "integrated_master_dataset.csv"
        )
        master_df.to_csv(master_path, index=False)
        logger.info("Saved %s", master_path)

        trainable = self._filter_trainable_rows(master_df)
        training_rows = self._select_training_rows(trainable)

        if len(training_rows) < 5:
            raise ValueError(
                f"Too few training rows ({len(training_rows)}). "
                "Need Vybor (+ optional unique Excel) with complete delta targets."
            )

        if self.training_mode == "labeled_only":
            train_df, val_df = self._split_clinical_labeled(training_rows)
        elif self.training_mode in PROXY_TRAIN_MODES:
            train_df, val_df = self._split_harmonized_extended(training_rows)
            train_df = self._attach_sample_weights(train_df)
        elif self.training_mode in EXTENDED_MODES:
            train_df, val_df = self._split_harmonized_extended(training_rows)
            if self.training_mode in {"clinical_xlsx_extended", "clinical_xlsx_kits_impute_only"}:
                train_df = self._attach_sample_weights(train_df)
        else:
            train_list: List[pd.DataFrame] = []
            val_list: List[pd.DataFrame] = []
            for source in training_rows["source"].unique():
                source_data = training_rows[training_rows["source"] == source]
                if len(source_data) >= 5:
                    source_train, source_val = train_test_split(
                        source_data, test_size=0.2, random_state=42
                    )
                    train_list.append(source_train)
                    val_list.append(source_val)
                else:
                    train_list.append(source_data)
            train_df = normalize_dataframe(pd.concat(train_list, ignore_index=True))
            val_df = (
                normalize_dataframe(pd.concat(val_list, ignore_index=True))
                if val_list
                else pd.DataFrame()
            )
            if len(val_df) == 0:
                train_df, val_df = train_test_split(
                    training_rows, test_size=0.2, random_state=42
                )
                train_df = normalize_dataframe(train_df)
                val_df = normalize_dataframe(val_df)

        train_path = PROCESSED_DIR / "train.csv"
        val_path = PROCESSED_DIR / "validation.csv"
        clinical_val_path = PROCESSED_DIR / "validation_clinical.csv"
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        val_df.to_csv(clinical_val_path, index=False)

        vybor_val = val_df[val_df["source"] == "Vybor"] if "source" in val_df.columns else val_df
        if len(vybor_val) > 0:
            vybor_audit_path = PROCESSED_DIR / "validation_vybor_only.csv"
            vybor_val.to_csv(vybor_audit_path, index=False)
            logger.info("Vybor-only holdout: %s rows -> %s", len(vybor_val), vybor_audit_path)

        if self.kits_feature_reference is not None:
            ref_path = PROCESSED_DIR / "kits19_feature_reference.csv"
            self.kits_feature_reference.to_csv(ref_path, index=False)
            logger.info("KiTS feature reference (no targets): %s", ref_path)

        with open(PROCESSED_DIR / "feature_names.json", "w", encoding="utf-8") as fh:
            json.dump(list(BASE_FEATURES), fh, indent=2, ensure_ascii=False)
        with open(PROCESSED_DIR / "target_names.json", "w", encoding="utf-8") as fh:
            json.dump(list(TARGET_NAMES), fh, indent=2, ensure_ascii=False)

        kits_in_train = bool(
            self.training_mode in PROXY_TRAIN_MODES
            and "source" in train_df.columns
            and (train_df["source"] == "KiTS19").any()
        )
        dicom_in_train = bool(
            self.training_mode in PROXY_TRAIN_MODES
            and "source" in train_df.columns
            and (train_df["source"] == "DICOMS").any()
        )
        manifest = {
            "training_mode": self.training_mode,
            "labeled_sources": sorted(LABELED_SOURCES),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "train_by_source": train_df["source"].value_counts().to_dict() if "source" in train_df.columns else {},
            "val_by_source": val_df["source"].value_counts().to_dict() if "source" in val_df.columns else {},
            "master_rows_by_source": master_df["source"].value_counts().to_dict(),
            "kits_in_train": kits_in_train,
            "dicom_in_train": dicom_in_train,
            "proxy_targets_stripped": self.training_mode not in PROXY_TRAIN_MODES,
            "sample_weights": self.training_mode in PROXY_TRAIN_MODES,
            "note": (
                "proxy_weighted_extended: KiTS proxy + DICOM pseudo in train with down-weighted "
                "sample_weight; validation is clinical Vybor holdout only."
                if self.training_mode in PROXY_TRAIN_MODES
                else (
                    "Regression targets are clinical-only (Vybor/Excel paired supine+lateral). "
                    "KiTS19/DICOM rows remain in master for feature reference but proxy δ "
                    "columns are stripped and never used in train/val."
                )
            ),
        }
        manifest_path = PROCESSED_DIR / "integration_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Manifest: %s", manifest_path)

        logger.info("Train: %s rows -> %s", len(train_df), train_path)
        logger.info("Validation: %s rows -> %s", len(val_df), val_path)
        return train_df, val_df

    def run(self):
        """Полный цикл интеграции."""
        logger.info("ЗАПУСК ИНТЕГРАЦИИ ДАННЫХ (Phase 1, mode=%s)", self.training_mode)
        self.load_data()
        self.create_universal_keys()
        master_df = self.create_master_dataset()
        missing_analysis = self.analyze_data_quality(master_df)
        train_df, val_df = self.save_integrated_data(master_df)
        logger.info("ИНТЕГРАЦИЯ ЗАНЕРШЕНА")
        return master_df, train_df, val_df, missing_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 data integration")
    parser.add_argument(
        "--mode",
        choices=[
            "labeled_only",
            "all",
            "harmonized_extended",
            "clinical_xlsx_extended",
            "clinical_xlsx_kits_impute_only",
            "proxy_weighted_extended",
        ],
        default="labeled_only",
        help="labeled_only: Vybor; extended modes add DICOM pseudo; kits_impute_only excludes KiTS from train",
    )
    parser.add_argument("--no-kits-fill", action="store_true", help="Skip KiTS median imputation for Excel rows")
    parser.add_argument("--excel-path", type=Path, default=DEFAULT_EXCEL_PATH_RESOLVED)
    parser.add_argument("--vybor-path", type=Path, default=DEFAULT_VYBOR_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    fixer = DataIntegrationFix(
        vybor_path=args.vybor_path,
        excel_path=args.excel_path,
        training_mode=args.mode,
        fill_sparse_from_kits=not args.no_kits_fill,
    )
    master_df, train_df, val_df, _missing = fixer.run()

    print("\nРЕЗУЛЬТАТЫ:")
    print(f"  Режим: {args.mode}")
    print(f"  Мастер-датасет: {len(master_df)} строк")
    print(f"  Train: {len(train_df)} строк")
    print(f"  Validation: {len(val_df)} строк")
    if "source" in train_df.columns:
        print(f"  Train sources: {train_df['source'].value_counts().to_dict()}")
    print(f"  Источники master: {master_df['source'].unique().tolist()}")
    return master_df, train_df, val_df


if __name__ == "__main__":
    main()
