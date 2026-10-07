"""Surname pairing for the CT versus Excel displacement report."""

from scripts.validation.compare_ct_excel_displacement import build_report, match_tables


def test_duplicate_surnames_are_not_paired_on_the_surname_alone():
    excel = [
        {"label": "Гусев", "surname": "гусев"},
        {"label": "Иванов", "surname": "иванов"},
    ]
    ct = [
        {"label": "Гусев Г.Г. - 21.10.2025", "surname": "гусев", "source": "boku"},
        {"label": "Гусев Станислав Иванович 09.07.2025", "surname": "гусев", "source": "boku"},
        {"label": "Иванов И.О", "surname": "иванов", "source": "boku"},
    ]
    pairs, skipped = match_tables(excel, ct)
    assert [pair[0]["label"] for pair in pairs] == ["Иванов"]
    assert {row["label"] for row in skipped} == {
        "Гусев",
        "Гусев Г.Г. - 21.10.2025",
        "Гусев Станислав Иванович 09.07.2025",
    }


def test_displacement_is_lateral_minus_supine_without_a_constant():
    excel = [{"label": "Шевченко", "surname": "шевченко", "left_delta_middle_z": -10.0, "left_lateral_middle_z": -50.0, "left_supine_middle_z": -40.0}]
    spine = [{"label": "Шевченко А.Н.", "surname": "шевченко", "source": "spine", "kidney_left_middle_z_vert": 5.0}]
    boku = [{"label": "Шевченко А.Н.", "surname": "шевченко", "source": "boku", "kidney_left_middle_z_vert": -8.0}]
    summary, patients, _duplicates = build_report(excel, __import__("pandas").DataFrame(), spine, boku)
    row = summary.set_index("metric").loc["delta/left_middle_z"]
    assert row["n"] == 1
    assert row["bias_ct_minus_excel"] == -3.0
    assert patients.loc[0, "match_status"] == "excel_and_both_ct"
