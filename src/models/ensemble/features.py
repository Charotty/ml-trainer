"""Feature transformer: normalize, engineer, and align the inference matrix."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.features.displacement_axis_features import (
    ANATOMICAL_FEATURES,
    DISPLACEMENT_AXIS_FEATURES,
    add_displacement_axis_features,
)
from src.features.fold_categoricals import FoldCategoricalEncoder
from src.features.leakage_safe import filter_model_features, is_leakage_feature
from src.features.na_trend_features import NaTrendStore, attach_na_trend_features
from src.features.phase1_schema import (
    BASE_FEATURES,
    CLINICAL_DEMOGRAPHIC_FEATURES,
    CROSS_FEATURES,
    ENGINEERED_FEATURES,
    TARGET_NAMES,
    encode_patient_position,
    normalize_dataframe,
)
from src.features.projection_enrichment import (
    add_projection_delta_proxies,
    attach_projection_features,
)

PACKAGE_STATUS = "production"


class FeatureTransformer:
    """Train/inference feature matrix builder. No estimators, no CV."""

    def __init__(
        self,
        *,
        enrichment_mode: str = "projection",
        na_trend_store: NaTrendStore | None = None,
        encode_categoricals: bool = True,
        add_missing_indicators: bool = True,
        drop_feature_groups: tuple[str, ...] | list[str] | None = None,
        drop_feature_prefixes: tuple[str, ...] | list[str] | None = None,
        keep_feature_names: tuple[str, ...] | list[str] | None = None,
        verbose: bool = True,
    ):
        self.enrichment_mode = enrichment_mode
        self.na_trend_store = na_trend_store
        self.encode_categoricals = bool(encode_categoricals)
        self.add_missing_indicators = bool(add_missing_indicators)
        self.drop_feature_groups = tuple(drop_feature_groups or ())
        self.drop_feature_prefixes = tuple(drop_feature_prefixes or ())
        self.keep_feature_names = tuple(keep_feature_names or ())
        self.verbose = bool(verbose)
        self.categorical_encoder_: FoldCategoricalEncoder | None = None
        self.feature_names: list[str] = []
        self.required_features = list(BASE_FEATURES)
        self.engineered_features = list(ENGINEERED_FEATURES)
        self.cross_features = list(CROSS_FEATURES)
        self.displacement_axis_features = filter_model_features(list(DISPLACEMENT_AXIS_FEATURES))
        self.anatomical_features = list(ANATOMICAL_FEATURES)
        self.clinical_features = list(CLINICAL_DEMOGRAPHIC_FEATURES)
        self.projection_features: list[str] = []
        self.na_trend_feature_cols: list[str] = []
        self.target_columns = list(TARGET_NAMES)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _apply_aux_enrichment(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df
        if self.enrichment_mode == "na_trends":
            out = attach_na_trend_features(out, self.na_trend_store)
        elif self.enrichment_mode == "projection":
            out = attach_projection_features(out)
            out = add_projection_delta_proxies(out)
        return out

    def _encode_frame(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        """One-hot sex/body_type inside the fold; never fill unknown with 0/50/25."""
        if not self.encode_categoricals:
            return df
        if fit or self.categorical_encoder_ is None:
            self.categorical_encoder_ = FoldCategoricalEncoder(
                add_missing_indicators=self.add_missing_indicators,
            )
            return self.categorical_encoder_.fit_transform(df)
        return self.categorical_encoder_.transform(df)

    def _apply_feature_group_drops(self, feature_cols: list[str]) -> list[str]:
        kept = list(feature_cols)
        dropped = set(self.drop_feature_groups)
        group_members = {
            "base": set(self.required_features),
            "demographics": set(self.clinical_features) | set(
                self.categorical_encoder_.extra_feature_names()
                if self.categorical_encoder_ is not None
                else []
            ),
            "engineered": set(self.engineered_features)
            | set(self.cross_features)
            | set(self.displacement_axis_features),
            "anatomical_extras": set(self.anatomical_features),
            "na_trends": set(),
        }
        if "base" in dropped:
            kept = [c for c in kept if c not in group_members["base"]]
        if "demographics" in dropped:
            kept = [c for c in kept if c not in group_members["demographics"]]
        if "engineered" in dropped:
            kept = [c for c in kept if c not in group_members["engineered"]]
        if "anatomical_extras" in dropped:
            kept = [c for c in kept if c not in group_members["anatomical_extras"]]
        if "na_trends" in dropped:
            kept = [
                c
                for c in kept
                if not (
                    c.startswith("na_pop_shift_")
                    or c.startswith("na_sup_")
                    or c.startswith("kits_")
                )
            ]
        for prefix in self.drop_feature_prefixes:
            kept = [c for c in kept if not str(c).startswith(prefix)]
        if self.keep_feature_names:
            allow = set(self.keep_feature_names)
            # Always keep missing-indicator companions of allowed columns when present.
            kept = [
                c
                for c in kept
                if c in allow or (c.endswith("_was_missing") and c[: -len("_was_missing")] in allow)
            ]
        if not kept:
            raise ValueError(
                "Feature ablation dropped every column; keep at least one group."
            )
        return kept

    def _build_feature_matrix(self, df):
        """Apply feature engineering and return (X, y, all_feature_cols, target_cols)."""
        df = normalize_dataframe(df)
        target_cols = [col for col in self.target_columns if col in df.columns]

        missing_base_features = [col for col in self.required_features if col not in df.columns]
        if missing_base_features:
            self._log(f"WARNING: Missing base features: {missing_base_features}")
            available_base_features = [col for col in self.required_features if col in df.columns]
            if len(available_base_features) < len(self.required_features) * 0.8:
                self._log("Too many missing features. Cannot proceed.")
                return None, None, None, None

        if len(target_cols) == 0:
            self._log("No target variables found in dataset")
            return None, None, None, None

        df_enhanced = self._create_engineered_features(df.copy())
        df_enhanced = self._create_cross_features(df_enhanced)
        df_enhanced = add_displacement_axis_features(df_enhanced)
        df_enhanced = self._apply_aux_enrichment(df_enhanced)

        base_feature_cols = [col for col in self.required_features if col in df_enhanced.columns]
        engineered_feature_cols = [col for col in self.engineered_features if col in df_enhanced.columns]
        cross_feature_cols = [col for col in self.cross_features if col in df_enhanced.columns]
        axis_feature_cols = [
            c for c in self.displacement_axis_features if c in df_enhanced.columns
        ]
        anatomical_cols = [c for c in self.anatomical_features if c in df_enhanced.columns]
        clinical_cols = [c for c in self.clinical_features if c in df_enhanced.columns]
        if self.categorical_encoder_ is not None and self.categorical_encoder_.fitted_:
            extra = [
                c
                for c in self.categorical_encoder_.extra_feature_names()
                if c in df_enhanced.columns
            ]
            clinical_cols = [c for c in clinical_cols if c not in ("sex", "body_type")] + extra
            seen: set[str] = set()
            deduped: list[str] = []
            for col in clinical_cols:
                if col not in seen:
                    seen.add(col)
                    deduped.append(col)
            clinical_cols = deduped
        projection_feature_cols = sorted(
            c for c in df_enhanced.columns
            if c.startswith("proj_") and not is_leakage_feature(c)
        )
        na_trend_feature_cols = sorted(
            c
            for c in df_enhanced.columns
            if c.startswith("na_pop_shift_")
            or c.startswith("na_sup_z_")
            or c.startswith("na_sup_pct_")
            or c.startswith("kits_z_")
            or c.startswith("kits_pct_")
            or c.startswith("kits_cohort_median_")
        )
        self.projection_features = projection_feature_cols
        self.na_trend_feature_cols = na_trend_feature_cols
        all_feature_cols = (
            base_feature_cols
            + engineered_feature_cols
            + cross_feature_cols
            + axis_feature_cols
            + anatomical_cols
            + clinical_cols
            + projection_feature_cols
            + na_trend_feature_cols
        )
        all_feature_cols = self._apply_feature_group_drops(all_feature_cols)

        X = df_enhanced[all_feature_cols].astype(float).values
        y = df_enhanced[target_cols].astype(float).values
        return X, y, all_feature_cols, target_cols

    def build_inference_matrix(self, df):
        """Apply train-time feature engineering aligned to ``self.feature_names``."""
        if not self.feature_names:
            raise RuntimeError(
                "feature_names is empty. Train (or load) the model before "
                "calling build_inference_matrix()."
            )
        df = normalize_dataframe(df)
        if self.encode_categoricals and self.categorical_encoder_ is not None and self.categorical_encoder_.fitted_:
            df = self.categorical_encoder_.transform(df)
        df_enhanced = self._create_engineered_features(df.copy())
        df_enhanced = self._create_cross_features(df_enhanced)
        df_enhanced = add_displacement_axis_features(df_enhanced)
        df_enhanced = self._apply_aux_enrichment(df_enhanced)
        for col in self.feature_names:
            if col not in df_enhanced.columns:
                df_enhanced[col] = np.nan
        return df_enhanced[self.feature_names].astype(float).values

    def _create_engineered_features(self, df):
        self._log("\n[FE] Creating engineered features...")
        if "body_width_mm" in df.columns and "body_depth_mm" in df.columns:
            df["body_ratio"] = df["body_width_mm"] / df["body_depth_mm"]
            self._log("  [OK] body_ratio created")
        if "kidney_left_center_x_rel" in df.columns and "kidney_right_center_x_rel" in df.columns:
            df["kidney_distance_lr"] = np.abs(
                df["kidney_left_center_x_rel"] - df["kidney_right_center_x_rel"]
            )
            self._log("  [OK] kidney_distance_lr created")
        if "kidney_left_volume_cm3" in df.columns and "body_width_mm" in df.columns:
            df["kidney_left_volume_norm"] = df["kidney_left_volume_cm3"] / df["body_width_mm"]
            self._log("  [OK] kidney_left_volume_norm created")
        if "kidney_right_volume_cm3" in df.columns and "body_width_mm" in df.columns:
            df["kidney_right_volume_norm"] = df["kidney_right_volume_cm3"] / df["body_width_mm"]
            self._log("  [OK] kidney_right_volume_norm created")
        if "kidney_left_length_mm" in df.columns and "body_width_mm" in df.columns:
            df["kidney_left_length_norm"] = df["kidney_left_length_mm"] / df["body_width_mm"]
            self._log("  [OK] kidney_left_length_norm created")
        if "kidney_right_length_mm" in df.columns and "body_width_mm" in df.columns:
            df["kidney_right_length_norm"] = df["kidney_right_length_mm"] / df["body_width_mm"]
            self._log("  [OK] kidney_right_length_norm created")
        if "kidney_left_volume_cm3" in df.columns and "kidney_right_volume_cm3" in df.columns:
            df["volume_asymmetry"] = df["kidney_left_volume_cm3"] - df["kidney_right_volume_cm3"]
            self._log("  [OK] volume_asymmetry created")
        if "kidney_left_length_mm" in df.columns and "kidney_right_length_mm" in df.columns:
            df["length_asymmetry"] = df["kidney_left_length_mm"] - df["kidney_right_length_mm"]
            self._log("  [OK] length_asymmetry created")
        if "kidney_left_to_spine_distance" in df.columns and "kidney_right_to_spine_distance" in df.columns:
            df["spine_distance_asymmetry"] = (
                df["kidney_left_to_spine_distance"] - df["kidney_right_to_spine_distance"]
            )
            self._log("  [OK] spine_distance_asymmetry created")
        if (
            "kidney_left_to_body_center_distance" in df.columns
            and "kidney_right_to_body_center_distance" in df.columns
        ):
            df["body_center_asymmetry"] = (
                df["kidney_left_to_body_center_distance"]
                - df["kidney_right_to_body_center_distance"]
            )
            self._log("  [OK] body_center_asymmetry created")
        if "kidney_left_to_spine_distance" in df.columns and "body_width_mm" in df.columns:
            df["kidney_left_to_spine_ratio"] = df["kidney_left_to_spine_distance"] / df["body_width_mm"]
            self._log("  [OK] kidney_left_to_spine_ratio created")
        if "kidney_right_to_spine_distance" in df.columns and "body_width_mm" in df.columns:
            df["kidney_right_to_spine_ratio"] = df["kidney_right_to_spine_distance"] / df["body_width_mm"]
            self._log("  [OK] kidney_right_to_spine_ratio created")
        if "patient_position_encoded" not in df.columns or df["patient_position_encoded"].isna().all():
            pos_col = None
            for candidate in ("scan_position", "patient_position"):
                if candidate in df.columns:
                    pos_col = candidate
                    break
            if pos_col is not None:
                df["patient_position_encoded"] = df[pos_col].map(encode_patient_position)
                self._log(f"  [OK] patient_position_encoded from {pos_col}")
            else:
                df["patient_position_encoded"] = 1
                self._log("  [OK] patient_position_encoded set to default (supine=1)")
        self._log(f"[FE] Engineered features creation completed. New shape: {df.shape}")
        return df

    def _create_cross_features(self, df):
        self._log("\n[FE] Creating cross-features...")
        if (
            "body_width_mm" in df.columns
            and "body_depth_mm" in df.columns
            and "kidney_left_length_mm" in df.columns
        ):
            avg_kidney_height = (
                df["kidney_left_length_mm"]
                + df.get("kidney_right_length_mm", df["kidney_left_length_mm"])
            ) / 2
            df["body_volume_estimated"] = (
                df["body_width_mm"] * df["body_depth_mm"] * avg_kidney_height / 1000
            )
            self._log("  [OK] body_volume_estimated created")
        if "kidney_left_volume_cm3" in df.columns and "kidney_left_length_mm" in df.columns:
            df["kidney_left_density_ratio"] = df["kidney_left_volume_cm3"] / df["kidney_left_length_mm"]
            self._log("  [OK] kidney_left_density_ratio created")
        if "kidney_right_volume_cm3" in df.columns and "kidney_right_length_mm" in df.columns:
            df["kidney_right_density_ratio"] = (
                df["kidney_right_volume_cm3"] / df["kidney_right_length_mm"]
            )
            self._log("  [OK] kidney_right_density_ratio created")
        if "spine_center_x" in df.columns and "body_width_mm" in df.columns:
            df["spine_to_body_ratio_x"] = df["spine_center_x"] / df["body_width_mm"]
            self._log("  [OK] spine_to_body_ratio_x created")
        if "spine_center_y" in df.columns and "body_depth_mm" in df.columns:
            df["spine_to_body_ratio_y"] = df["spine_center_y"] / df["body_depth_mm"]
            self._log("  [OK] spine_to_body_ratio_y created")
        if all(col in df.columns for col in ["body_com_x", "body_com_y", "spine_center_x", "spine_center_y"]):
            df["body_com_to_spine_distance"] = np.sqrt(
                (df["body_com_x"] - df["spine_center_x"]) ** 2
                + (df["body_com_y"] - df["spine_center_y"]) ** 2
            )
            self._log("  [OK] body_com_to_spine_distance created")
        if "kidney_left_to_spine_distance" in df.columns and "kidney_left_volume_cm3" in df.columns:
            df["kidney_left_spine_interaction"] = (
                df["kidney_left_to_spine_distance"] * df["kidney_left_volume_cm3"]
            )
            self._log("  [OK] kidney_left_spine_interaction created")
        if "kidney_right_to_spine_distance" in df.columns and "kidney_right_volume_cm3" in df.columns:
            df["kidney_right_spine_interaction"] = (
                df["kidney_right_to_spine_distance"] * df["kidney_right_volume_cm3"]
            )
            self._log("  [OK] kidney_right_spine_interaction created")
        if "body_width_mm" in df.columns and "body_depth_mm" in df.columns:
            bw = pd.to_numeric(df["body_width_mm"], errors="coerce")
            bd = pd.to_numeric(df["body_depth_mm"], errors="coerce")
            df["body_size_index"] = np.sqrt(bw**2 + bd**2)
            self._log("  [OK] body_size_index created")
        if all(
            col in df.columns
            for col in ["kidney_left_center_x_rel", "kidney_left_center_y_rel", "kidney_left_center_z_rel"]
        ):
            lx = pd.to_numeric(df["kidney_left_center_x_rel"], errors="coerce")
            ly = pd.to_numeric(df["kidney_left_center_y_rel"], errors="coerce")
            lz = pd.to_numeric(df["kidney_left_center_z_rel"], errors="coerce")
            df["kidney_position_index_left"] = np.sqrt(lx**2 + ly**2 + lz**2)
            self._log("  [OK] kidney_position_index_left created")
        if all(
            col in df.columns
            for col in ["kidney_right_center_x_rel", "kidney_right_center_y_rel", "kidney_right_center_z_rel"]
        ):
            rx = pd.to_numeric(df["kidney_right_center_x_rel"], errors="coerce")
            ry = pd.to_numeric(df["kidney_right_center_y_rel"], errors="coerce")
            rz = pd.to_numeric(df["kidney_right_center_z_rel"], errors="coerce")
            df["kidney_position_index_right"] = np.sqrt(rx**2 + ry**2 + rz**2)
            self._log("  [OK] kidney_position_index_right created")
        if "kidney_left_volume_cm3" in df.columns and "body_area_mm2" in df.columns:
            df["volume_to_area_ratio_left"] = df["kidney_left_volume_cm3"] / (df["body_area_mm2"] / 100)
            self._log("  [OK] volume_to_area_ratio_left created")
        if "kidney_right_volume_cm3" in df.columns and "body_area_mm2" in df.columns:
            df["volume_to_area_ratio_right"] = df["kidney_right_volume_cm3"] / (
                df["body_area_mm2"] / 100
            )
            self._log("  [OK] volume_to_area_ratio_right created")
        if (
            "kidney_left_volume_cm3" in df.columns
            and "kidney_right_volume_cm3" in df.columns
            and "body_width_mm" in df.columns
        ):
            df["relative_volume_sum"] = (
                df["kidney_left_volume_cm3"] + df["kidney_right_volume_cm3"]
            ) / df["body_width_mm"]
            self._log("  [OK] relative_volume_sum created")
        if all(
            col in df.columns
            for col in [
                "kidney_left_center_x_rel",
                "kidney_right_center_x_rel",
                "kidney_left_center_y_rel",
                "kidney_right_center_y_rel",
            ]
        ):
            lx = pd.to_numeric(df["kidney_left_center_x_rel"], errors="coerce")
            ly = pd.to_numeric(df["kidney_left_center_y_rel"], errors="coerce")
            rx = pd.to_numeric(df["kidney_right_center_x_rel"], errors="coerce")
            ry = pd.to_numeric(df["kidney_right_center_y_rel"], errors="coerce")
            left_vector = np.column_stack([lx, ly])
            right_vector = np.column_stack([rx, ry])
            dot_product = np.sum(left_vector * right_vector, axis=1)
            norm_left = np.sqrt(np.sum(left_vector**2, axis=1))
            norm_right = np.sqrt(np.sum(right_vector**2, axis=1))
            cos_angle = np.where(
                (norm_left > 0) & (norm_right > 0),
                dot_product / (norm_left * norm_right),
                0,
            )
            df["kidney_separation_angle"] = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
            self._log("  [OK] kidney_separation_angle created")
        self._log(f"[FE] Cross-features creation completed. New shape: {df.shape}")
        return df


def inference_transformer_from_payload(payload: Mapping[str, Any]) -> FeatureTransformer:
    """Restore a FeatureTransformer for inference from a joblib payload."""
    from src.features.na_trend_features import NaTrendStore

    store_payload = payload.get("na_trend_store")
    store = NaTrendStore.from_dict(store_payload) if store_payload else None
    transformer = FeatureTransformer(
        enrichment_mode=payload.get("enrichment_mode", "projection"),
        na_trend_store=store,
        encode_categoricals=bool(payload.get("encode_categoricals", False)),
        verbose=False,
    )
    transformer.feature_names = list(payload["feature_names"])
    encoder = payload.get("categorical_encoder")
    if encoder is not None:
        transformer.categorical_encoder_ = encoder
        transformer.encode_categoricals = True
    return transformer
