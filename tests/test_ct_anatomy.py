"""Synthetic-volume tests for the CT anatomy extractor. No DICOM required."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.ct_anatomy.abdomen import (
    assign_clinical_body_size,
    clinical_body_size_at,
    kidney_level_z_from_row,
    measure_abdomen,
    widest_body_z,
)
from src.features.ct_anatomy.calibration import apply_feature_calibration, suggest_calibration
from src.features.ct_anatomy.distances import measure_kidney_distances
from src.features.ct_anatomy.kidney_shape import measure_kidney_shape
from src.features.ct_anatomy.perirenal import map_points, measure_perirenal
from src.features.ct_anatomy.psoas import measure_psoas
from src.features.ct_anatomy.qc import QC_MANUAL, QC_OUT_OF_RANGE, body_type_from_bmi, finalize_qc
from src.features.ct_anatomy.segmentation import run_anatomy_segmentation
from src.features.ct_anatomy.spine import measure_spine
from src.features.ct_anatomy.vertebral_axes import score_variant_pairs
from src.features.ct_anatomy.vertebral_frame import measure_vertebral_frame, publish_vertebral_frame
from src.features.ct_anatomy.volume import AnatomyVolume
from scripts.validation.validate_extractor_anatomy import agreement_row, compare_tables


def _volume(shape, masks, hu=None) -> AnatomyVolume:
    return AnatomyVolume(affine=np.eye(4), masks=masks, hu=hu)


def _tilted_slab(mask, x0, x1, y0, y1, z_at, thickness=6):
    for y in range(y0, y1):
        z_top = int(round(z_at(y)))
        z_bot = max(0, z_top - thickness)
        mask[x0:x1, y, z_bot:z_top] = True


def test_body_type_uses_excel_codes():
    assert body_type_from_bmi(17.0) == 1.0
    assert body_type_from_bmi(22.0) == 0.0
    assert body_type_from_bmi(31.0) == 2.0
    assert body_type_from_bmi(None) is None


def test_out_of_range_is_cleared_and_approx_is_kept():
    cleared = finalize_qc({"lumbar_lordosis_deg": 200.0, "lumbar_lordosis_deg_qc": "ok"})
    assert cleared["lumbar_lordosis_deg"] is None
    assert cleared["lumbar_lordosis_deg_qc"] == QC_OUT_OF_RANGE

    kept = finalize_qc(
        {
            "kidney_left_upper_pole_to_diaphragm_mm": 40.0,
            "kidney_left_upper_pole_to_diaphragm_mm_qc": "approx",
        }
    )
    assert kept["kidney_left_upper_pole_to_diaphragm_mm"] == 40.0
    assert kept["kidney_left_upper_pole_to_diaphragm_mm_qc"] == "approx"


def test_spine_levels_lordosis_and_disc_height():
    shape = (80, 80, 180)
    masks = {name: np.zeros(shape, dtype=bool) for name in ("vertebrae_L1", "vertebrae_L3", "vertebrae_L4", "vertebrae_S1")}
    masks["vertebrae_L1"][40:60, 20:40, 150:156] = True
    masks["vertebrae_L3"][40:60, 20:40, 120:130] = True
    masks["vertebrae_L4"][40:60, 20:40, 100:110] = True
    theta = np.deg2rad(30.0)
    _tilted_slab(
        masks["vertebrae_S1"],
        40,
        60,
        20,
        40,
        lambda y: 40 + (y - 20) * np.tan(theta),
    )
    measured = measure_spine(_volume(shape, masks))
    assert measured["vertebrae_L3_z"] == pytest.approx(124.5)
    assert measured["l3_l4_z"] == pytest.approx(0.5 * (124.5 + 104.5))
    assert measured["lumbar_lordosis_deg"] == pytest.approx(30.0, abs=5.0)
    assert measured["s1_plate_tilt_deg"] == pytest.approx(-30.0, abs=5.0)
    assert measured["disc_l3l4_anterior_height_mm"] == pytest.approx(11.0)
    assert measured["disc_l3l4_posterior_height_mm"] == pytest.approx(11.0)


def test_abdomen_diameters_and_wall_at_given_slice():
    shape = (160, 120, 40)
    body = np.zeros(shape, dtype=bool)
    cavity = np.zeros(shape, dtype=bool)
    body[10:110, 5:85, 34] = True
    cavity[20:100, 20:80, 34] = True
    measured = measure_abdomen(_volume(shape, {"body_trunc": body, "abdominal_cavity": cavity}), 34.0)
    assert measured["abd_width_l3l4_mm"] == pytest.approx(99.0)
    assert measured["abd_depth_l3l4_mm"] == pytest.approx(79.0)
    assert measured["abd_wall_thickness_mm"] == pytest.approx(15.0)
    assert measured["abd_wall_thickness_mm_qc"] == "ok"


def _torso(width_vox: int, depth_vox: int, z: int) -> AnatomyVolume:
    shape = (width_vox + 40, depth_vox + 40, z + 5)
    body = np.zeros(shape, dtype=bool)
    body[20 : 20 + width_vox, 20 : 20 + depth_vox, z] = True
    return _volume(shape, {"body_trunc": body})


def test_body_mask_size_replaces_cropped_hu_blob():
    volume = _torso(301, 201, 12)
    measured = clinical_body_size_at(volume, 12.0, source="body_mask_kidney_z")
    assert measured["body_mask_width_mm"] == pytest.approx(300.0)
    assert measured["body_mask_depth_mm"] == pytest.approx(200.0)
    assert measured["body_mask_qc"] == "approx"

    row = {
        "body_width_mm": 121.0,
        "body_depth_mm": 123.0,
        "body_area_mm2": 7844.0,
        **measured,
    }
    assigned = assign_clinical_body_size(row)
    assert assigned["body_width_mm"] == pytest.approx(300.0)
    assert assigned["body_depth_mm"] == pytest.approx(200.0)
    assert assigned["body_area_mm2"] == pytest.approx(300.0 * 200.0)
    assert assigned["body_size_source"] == "body_mask_kidney_z"


def test_cropped_body_mask_is_not_written_into_model_columns():
    tiny = clinical_body_size_at(_torso(100, 80, 4), 4.0, source="body_mask_kidney_z")
    assert tiny["body_mask_qc"] == "out_of_range"
    assert assign_clinical_body_size({"body_width_mm": 121.0, **tiny}) == {}


def test_one_kidney_still_selects_a_body_slice():
    assert kidney_level_z_from_row({"kidney_right_center_z": 40.0, "kidney_left_center_z": None}) == pytest.approx(40.0)
    assert kidney_level_z_from_row({"kidney_left_center_z": 10.0, "kidney_right_center_z": 30.0}) == pytest.approx(20.0)

    shape = (80, 80, 30)
    body = np.zeros(shape, dtype=bool)
    body[10:20, 10:20, 5] = True
    body[5:70, 5:60, 18] = True
    assert widest_body_z(_volume(shape, {"body_trunc": body})) == pytest.approx(18.0)


def test_skin_mask_without_vertebrae_still_fills_body_size():
    from src.features.ct_anatomy.extract import extract_anatomy_features

    shape = (360, 260, 40)
    body = np.zeros(shape, dtype=bool)
    body[20:321, 20:221, 20] = True
    row = extract_anatomy_features(
        _volume(shape, {"body_trunc": body}),
        apply_ranges=False,
        calibration={},
    )
    assert row["body_width_mm"] == pytest.approx(300.0)
    assert row["body_depth_mm"] == pytest.approx(200.0)
    assert row["body_area_mm2"] == pytest.approx(60000.0)
    assert row["body_size_source"] == "body_mask_widest_slice"


def test_kidney_landmark_distances():
    shape = (90, 80, 100)
    kidney = np.zeros(shape, dtype=bool)
    left = np.zeros(shape, dtype=bool)
    rib = np.zeros(shape, dtype=bool)
    hip = np.zeros(shape, dtype=bool)
    liver = np.zeros(shape, dtype=bool)
    lung = np.zeros(shape, dtype=bool)
    body = np.zeros(shape, dtype=bool)
    process = np.zeros(shape, dtype=bool)
    t11 = np.zeros(shape, dtype=bool)
    kidney[10:20, 30:40, 30:50] = True
    left[60:70, 30:40, 30:50] = True
    # Medial rib end next to T11, plus a low anterior tip that must be ignored.
    rib[38:42, 30:40, 68:72] = True
    rib[4:8, 30:40, 54:57] = True
    t11[42:52, 30:40, 65:75] = True
    # Crest is offset in Y; only the vertical gap counts. z max of the crest is 14.
    hip[10:20, 60:70, 10:15] = True
    liver[10:20, 30:40, 70:76] = True
    lung[60:70, 30:40, 80:88] = True
    body[35:45, 30:40, 30:50] = True
    process[25:36, 30:40, 30:50] = True
    measured = measure_kidney_distances(
        _volume(
            shape,
            {
                "kidney_right": kidney,
                "kidney_left": left,
                "rib_right_11": rib,
                "vertebrae_T11": t11,
                "hip_right": hip,
                "liver": liver,
                "lung_lower_lobe_left": lung,
                "vertebrae_L3": body | process,
                "vertebrae_body": body,
            },
        )
    )
    # Upper pole z=49, medial rib end z=69.5. The anterior tip at z=54 is ignored.
    assert measured["kidney_right_upper_pole_to_rib11_mm"] == pytest.approx(20.5, abs=1.0)
    # Lower pole z=30, crest z=14.
    assert measured["kidney_right_lower_pole_to_iliac_crest_mm"] == pytest.approx(16.0, abs=0.6)
    # Underside of the liver in the column above the pole, not the liver top.
    assert measured["kidney_right_upper_pole_to_diaphragm_mm"] == pytest.approx(21.0, abs=0.6)
    assert measured["kidney_right_upper_pole_to_diaphragm_mm_qc"] == "approx"
    assert measured["kidney_left_upper_pole_to_diaphragm_mm"] == pytest.approx(31.0, abs=0.6)
    # Vertebral body starts at x=35. The transverse process at x=25 must not close the gap.
    assert measured["kidney_right_medial_to_spine_mm"] == pytest.approx(16.0, abs=0.6)
    assert measured["kidney_right_medial_to_spine_mm_qc"] == "ok"


def test_iliac_distance_is_negative_when_the_crest_is_above_the_pole():
    shape = (40, 40, 60)
    kidney = np.zeros(shape, dtype=bool)
    hip = np.zeros(shape, dtype=bool)
    kidney[10:20, 10:20, 10:20] = True
    hip[10:20, 10:20, 40:50] = True
    measured = measure_kidney_distances(_volume(shape, {"kidney_right": kidney, "hip_right": hip}))
    assert measured["kidney_right_lower_pole_to_iliac_crest_mm"] == pytest.approx(-39.0, abs=0.6)


def test_perirenal_ray_and_map_bins():
    shape = (80, 80, 5)
    kidney = np.zeros(shape, dtype=bool)
    kidney[30:50, 30:50, 2] = True
    hu = np.full(shape, 40.0)
    hu[30:50, 50:58, 2] = -120.0
    measured = measure_perirenal(_volume(shape, {"kidney_left": kidney}, hu=hu))
    assert measured["kidney_left_perirenal_dorsal_mm"] == pytest.approx(8.0)
    assert measured["kidney_left_map_score"] == 0.0
    assert map_points(25.0, 0.30) == 5.0
    assert map_points(15.0, 0.10) == 3.0
    assert map_points(5.0, 0.0) == 0.0


def test_perirenal_ray_skips_partial_volume_rim():
    shape = (40, 80, 5)
    kidney = np.zeros(shape, dtype=bool)
    kidney[10:20, 20:40, 2] = True
    hu = np.full(shape, 40.0)
    hu[10:20, 40:42, 2] = 30.0
    hu[10:20, 42:50, 2] = -110.0
    measured = measure_perirenal(_volume(shape, {"kidney_left": kidney}, hu=hu))
    assert measured["kidney_left_perirenal_dorsal_mm"] == pytest.approx(8.0, abs=1.0)


def test_lateral_fat_stops_at_the_organ_not_a_sideways_ray():
    shape = (80, 80, 12)
    kidney = np.zeros(shape, dtype=bool)
    liver = np.zeros(shape, dtype=bool)
    spleen = np.zeros(shape, dtype=bool)
    kidney[40:55, 30:45, 4:8] = True
    liver[28:35, 30:45, 4:8] = True
    left = np.zeros(shape, dtype=bool)
    left[10:25, 30:45, 4:8] = True
    spleen[32:40, 30:45, 4:8] = True
    hu = np.full(shape, 40.0)
    measured = measure_perirenal(
        _volume(shape, {"kidney_right": kidney, "kidney_left": left, "liver": liver, "spleen": spleen}, hu=hu)
    )
    assert measured["kidney_right_perirenal_lateral_mm"] == pytest.approx(6.0, abs=1.0)
    assert measured["kidney_right_perirenal_lateral_mm_qc"] == "ok"
    assert measured["kidney_left_perirenal_lateral_mm"] == pytest.approx(8.0, abs=1.0)
    assert measured["kidney_left_perirenal_lateral_mm_qc"] == "ok"


def test_perirenal_density_uses_the_posterior_pad():
    shape = (40, 80, 5)
    kidney = np.zeros(shape, dtype=bool)
    kidney[10:30, 20:40, 2] = True
    hu = np.full(shape, 400.0)
    hu[10:30, 41:49, 2] = -80.0
    hu[10:30, 5:12, 2] = -150.0
    measured = measure_perirenal(_volume(shape, {"kidney_left": kidney}, hu=hu))
    assert measured["kidney_left_perirenal_hu"] == pytest.approx(-80.0, abs=15.0)


def test_psoas_area_on_the_requested_slice():
    shape = (80, 80, 20)
    psoas = np.zeros(shape, dtype=bool)
    psoas[40:50, 40:50, 8] = True
    measured = measure_psoas(_volume(shape, {"psoas_major_right": psoas}), 8.0)
    assert measured["kidney_right_psoas_area_cm2"] == pytest.approx(1.0)
    assert measured["kidney_right_psoas_thickness_mm"] == pytest.approx(9.0)
    assert measured["kidney_right_psoas_area_cm2_qc"] == "ok"


def test_psoas_level_follows_each_kidney():
    shape = (40, 40, 30)
    right = np.zeros(shape, dtype=bool)
    left = np.zeros(shape, dtype=bool)
    right[10:20, 10:16, 5] = True
    left[10:20, 10:22, 20] = True
    measured = measure_psoas(
        _volume(shape, {"psoas_major_right": right, "psoas_major_left": left}),
        z_by_side={"right": 5.0, "left": 20.0},
    )
    assert measured["kidney_right_psoas_thickness_mm"] == pytest.approx(5.0)
    assert measured["kidney_left_psoas_thickness_mm"] == pytest.approx(11.0)
    assert measured["kidney_right_psoas_area_cm2"] is not None
    missing = measure_psoas(
        _volume(shape, {"psoas_major_right": right}),
        z_by_side={"right": None, "left": None},
    )
    assert missing["kidney_right_psoas_thickness_mm"] is None


def test_kidney_thirds_rotation_and_contrast_pedicle():
    shape = (80, 80, 50)
    kidney = np.zeros(shape, dtype=bool)
    kidney[50:58, 40:48, 5:35] = True
    aorta = np.zeros(shape, dtype=bool)
    aorta[30:36, 40:48, 15:25] = True
    hu = np.zeros(shape, dtype=float)
    hu[30:58, 40:48, 15:25] = 300.0
    measured = measure_kidney_shape(_volume(shape, {"kidney_left": kidney, "aorta": aorta}, hu=hu))
    upper = measured["kidney_left_third_upper_z"]
    middle = measured["kidney_left_third_middle_z"]
    lower = measured["kidney_left_third_lower_z"]
    assert upper > middle > lower
    assert measured["kidney_left_rotation_a_deg"] == pytest.approx(0.0, abs=8.0)
    assert measured["kidney_left_rotation_a_deg_qc"] == "approx"
    # Left-kidney hilum opens toward the midline (−X), so cos(angle) is negative.
    assert np.cos(np.deg2rad(measured["kidney_left_rotation_c_deg"])) < 0
    assert measured["kidney_left_pedicle_status"] == "approx"
    assert measured["kidney_left_pedicle_length_mm"] > 0

    dark = measure_kidney_shape(_volume(shape, {"kidney_left": kidney, "aorta": aorta}, hu=np.zeros(shape)))
    assert dark["kidney_left_pedicle_status"] == "no_contrast"
    assert dark["kidney_left_pedicle_length_mm"] is None


def test_vertebral_frame_uses_the_body_at_the_kidney_not_a_side_bone():
    shape = (80, 40, 50)
    kidney = np.zeros(shape, dtype=bool)
    body = np.zeros(shape, dtype=bool)
    process = np.zeros(shape, dtype=bool)
    kidney[60:70, 20:30, 10:40] = True
    body[40:50, 20:30, 10:40] = True
    process[5:15, 20:30, 10:40] = True
    measured = measure_vertebral_frame(
        _volume(shape, {"kidney_left": kidney, "vertebrae_body": body, "vertebrae_L3": process})
    )
    assert measured["kidney_left_middle_x_vert"] == pytest.approx(20.0, abs=0.6)
    assert measured["kidney_left_middle_y_vert"] == pytest.approx(0.0, abs=0.6)
    assert measured["kidney_left_middle_z_vert"] == pytest.approx(0.0, abs=1.0)
    assert measured["vert_origin_x"] == pytest.approx(44.5, abs=0.6)


def test_skin_size_replaces_the_cropped_body_median():
    published = publish_vertebral_frame(
        {
            "vert_origin_x": 10.0,
            "vert_origin_y": 20.0,
            "vert_origin_z": 30.0,
            "body_width_mm": 320.0,
            "body_depth_mm": 210.0,
            "body_width_mm_median": 110.0,
            "body_depth_mm_median": 95.0,
            "body_size_source": "body_mask_l3l4",
        }
    )
    assert published["spine_center_z_mm"] == pytest.approx(30.0)
    assert published["spine_center_source"] == "vertebral_body_at_kidney"
    assert published["body_width_mm_median"] == pytest.approx(320.0)
    assert published["body_depth_mm_median"] == pytest.approx(210.0)
    assert publish_vertebral_frame({"body_width_mm": 110.0, "body_depth_mm": 90.0}) == {}


def test_manual_fields_are_not_invented():
    shape = (8, 8, 8)
    kidney = np.zeros(shape, dtype=bool)
    kidney[2:5, 2:5, 2:5] = True
    from src.features.ct_anatomy.extract import extract_anatomy_features

    row = extract_anatomy_features(_volume(shape, {"kidney_left": kidney}), apply_ranges=False, calibration={})
    assert row["diagnosis"] is None
    assert row["has_previous_surgery_qc"] == QC_MANUAL
    assert row["anatomy_feature_schema"] == "ct_anatomy_v1"


def test_full_profile_reuses_cached_task(tmp_path: Path):
    seg = tmp_path / "seg"
    (seg / "body").mkdir(parents=True)
    (seg / "body" / "body_trunc.nii.gz").write_bytes(b"cached")
    calls = []

    def runner(**kwargs):
        calls.append(kwargs["task"])
        out = Path(kwargs["output"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "marker.nii.gz").write_bytes(b"x")
        return True

    first = run_anatomy_segmentation(tmp_path / "ct.nii.gz", seg, "fast", runner=runner)
    assert calls == []
    assert first["anatomy_profile"] == "fast"

    full = run_anatomy_segmentation(tmp_path / "ct.nii.gz", seg, "full", runner=runner, reuse=True)
    assert "body" not in calls
    assert "total" in calls
    assert full["anatomy_seg_status_body"] == "cached"
    assert full["anatomy_seg_status_total"] == "ok"


def test_linear_calibration_roundtrip():
    auto = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    manual = 2.0 * auto + 3.0
    coeff = suggest_calibration(manual, auto)
    assert coeff is not None
    assert coeff["slope"] == pytest.approx(2.0)
    assert coeff["intercept"] == pytest.approx(3.0)
    corrected = apply_feature_calibration(
        {"abd_wall_thickness_mm": 4.0},
        {"abd_wall_thickness_mm": coeff},
    )
    assert corrected["abd_wall_thickness_mm"] == pytest.approx(11.0)
    assert corrected["abd_wall_thickness_mm_calibrated"] is True
    assert suggest_calibration(manual[:3], auto[:3]) is None


def test_agreement_metrics_and_unpaired_report():
    manual = np.array([10.0, 12.0, 14.0, 16.0])
    auto = manual + 2.0
    stats = agreement_row("abd_wall_thickness_mm", manual, auto)
    assert stats["n_paired"] == 4
    assert stats["mae"] == pytest.approx(2.0)
    assert stats["bias"] == pytest.approx(2.0)
    assert stats["icc"] == pytest.approx(0.8, abs=0.15)

    workbook = pd.DataFrame(
        {
            "name_key": ["ivanov"],
            "abd_wall_thickness_mm": [12.0],
        }
    )
    report = compare_tables(workbook, extracted=None)
    assert report.loc[0, "status"] == "no_extraction"
    assert int(report.loc[0, "n_manual"]) == 1


def _box(shape, x0, x1, y0, y1, z0, z1):
    mask = np.zeros(shape, dtype=bool)
    mask[x0:x1, y0:y1, z0:z1] = True
    return mask


def test_lyashchenko_axes_follow_a_rotated_vertebra():
    """On the side, scanner X is the front-back axis. Lateral distance follows the vertebra."""
    shape = (80, 80, 40)
    body = _box(shape, 18, 28, 36, 46, 18, 24)
    arch = _box(shape, 34, 44, 38, 44, 19, 23)
    kidney = _box(shape, 8, 16, 18, 28, 10, 34)
    psoas = _box(shape, 30, 50, 20, 24, 18, 24)
    hu = np.zeros(shape, dtype=float)
    hu[17:25, 18:28, 18:24] = -80.0
    measured = measure_vertebral_frame(
        _volume(
            shape,
            {
                "kidney_left": kidney,
                "vertebrae_body": body,
                "vertebrae_L2": body | arch,
                "iliopsoas_left": psoas,
            },
            hu=hu,
        )
    )
    # Scanner X difference is the front-back gap, about 11 mm. The sheet's X is lateral.
    assert measured["kidney_left_middle_x_vert"] == pytest.approx(11.0, abs=2.0)
    assert measured["kidney_left_middle_x_body_com"] == pytest.approx(18.0, abs=2.5)
    assert measured["kidney_left_middle_x_body_com"] > measured["kidney_left_middle_x_vert"] + 4.0
    assert measured["kidney_left_middle_y_canal_com"] > measured["kidney_left_middle_y_body_com"] + 8.0
    assert measured["kidney_left_middle_x_body_near"] < measured["kidney_left_middle_x_body_com"] - 3.0
    assert measured["kidney_left_middle_vert_level"] == "L2"
    assert measured["kidney_left_perirenal_dorsal_vert_mm"] > 4.0
    assert measured["kidney_left_perirenal_ventral_vert_mm"] == pytest.approx(0.0, abs=0.1)
    assert measured["kidney_left_psoas_thickness_vert_mm"] > 15.0
    assert measured["kidney_left_medial_to_spine_vert_mm"] > 0.0


def test_z_is_a_projection_onto_named_vertebrae():
    shape = (40, 40, 50)
    low = _box(shape, 16, 24, 16, 24, 8, 14)
    high = _box(shape, 16, 24, 16, 24, 28, 34)
    arch_low = _box(shape, 18, 22, 26, 32, 9, 13)
    arch_high = _box(shape, 18, 22, 26, 32, 29, 33)
    kidney = _box(shape, 28, 34, 10, 16, 6, 40)
    measured = measure_vertebral_frame(
        _volume(
            shape,
            {
                "kidney_left": kidney,
                "vertebrae_body": low | high,
                "vertebrae_L3": low | arch_low,
                "vertebrae_L2": high | arch_high,
            },
        )
    )
    assert measured["kidney_left_lower_vert_level"] == "L3"
    assert measured["kidney_left_upper_vert_level"] == "L2"
    assert measured["kidney_left_lower_z_from_l3"] < measured["kidney_left_upper_z_from_l3"]
    assert measured["kidney_left_upper_z_from_l2"] == pytest.approx(0.0, abs=8.0)


def test_variant_score_keeps_the_definition_that_matches_the_sheet():
    excel = {
        "right_supine_middle_x": 20.0,
        "right_supine_middle_y": 30.0,
        "right_supine_middle_z": -40.0,
    }
    ct = {
        "kidney_right_middle_x_body_com": 21.0,
        "kidney_right_middle_y_body_com": 29.0,
        "kidney_right_middle_z_body_com": 1.0,
        "kidney_right_middle_x_canal_near": 8.0,
        "kidney_right_middle_y_canal_near": 4.0,
        "kidney_right_middle_z_canal_near": 1.0,
        "kidney_right_middle_z_from_l2": -41.0,
    }
    # The other thirds and the left side stay empty, so only the filled cells count.
    scored = score_variant_pairs([(excel, ct)], "supine")
    by_key = {(row["variant"], row["axis"], row["third"]): row for row in scored}
    body = by_key[("body_com", "x", "middle")]
    canal = by_key[("canal_near", "x", "middle")]
    assert body["mae"] < canal["mae"]
    local_z = by_key[("body_com", "z", "middle")]
    level_z = by_key[("centroid_from_l2", "z", "middle")]
    assert level_z["mae"] < local_z["mae"]
