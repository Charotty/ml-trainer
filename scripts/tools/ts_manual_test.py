#!/usr/bin/env python3
from pathlib import Path


def main() -> int:
    from totalsegmentator.python_api import totalsegmentator

    inp = Path("/tmp/ml_trainer_dicom/nifti_case_0001_0345bc6294b9/Abdomen_Native_3.nii.gz")
    out = Path("/tmp/ml_trainer_dicom/seg_manual_test")
    out.mkdir(parents=True, exist_ok=True)
    print("input exists", inp.exists(), inp)
    totalsegmentator(
        input=str(inp),
        output=str(out),
        roi_subset=["kidney_left", "kidney_right"],
        device="gpu",
        fast=True,
        quiet=False,
        nr_thr_resamp=1,
        nr_thr_saving=1,
    )
    print("output files:", [p.name for p in out.glob("*")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
