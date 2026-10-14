"""The retrieval evaluation: Hit Rate@k, MRR and equipment isolation, at the strict and the area level (no index needed)."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def re_():
    spec = importlib.util.spec_from_file_location("retrieval_eval", ROOT / "eval" / "retrieval_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(title, equipment="thermal_station"):
    return SimpleNamespace(section_title=title, equipment_model=equipment)


# --- matching a passage to the judged section ---


@pytest.mark.parametrize(
    "title, relevant, area, expected",
    [
        ("6.2.3 Power OFF Procedure", "6.2.3 Power OFF Procedure", False, True),
        ("6.2.3.1 Detail", "6.2.3 Power OFF Procedure", False, True),  # inside the judged section
        ("6.2 Machine Operating Procedure", "6.2.3 Power OFF Procedure", False, False),  # the parent: not strict
        ("6.2 Machine Operating Procedure", "6.2.3 Power OFF Procedure", True, True),  # the parent: area
        ("6. Operation", "6.2.3 Power OFF Procedure", True, False),  # a whole chapter is too coarse
        ("6.20 Other", "6.2.3 Power OFF Procedure", True, False),  # 6.20 is not 6.2
        ("6.3 Something", "6.2.3 Power OFF Procedure", True, False),
        ("2.5.4 Draining of EGW Line", "2.5.4 Draining of EGW Line", False, True),
        ("Unknown", "2.5.4 Draining of EGW Line", True, False),
        (None, "2.5.4 Draining of EGW Line", True, False),
        ("", "2.5.4", False, False),
        ("Recipe Screen overview", "Recipe Screen", False, True),  # a judged section without a number: titles
        ("Safety", "Recipe Screen", False, False),
    ],
)
def test_a_passage_matches_the_judged_section(re_, title, relevant, area, expected):
    assert re_.section_matches(title, relevant, area) is expected


def test_the_rank_is_the_first_passage_that_matches(re_):
    chunks = [chunk("1. Introduction"), chunk("6.2 Machine Operating Procedure"), chunk("6.2.3 Power OFF Procedure")]
    assert re_.rank_of_first_hit(chunks, ["6.2.3 Power OFF Procedure"]) == 3
    assert re_.rank_of_first_hit(chunks, ["6.2.3 Power OFF Procedure"], area=True) == 2
    assert re_.rank_of_first_hit(chunks, ["9.9 Nothing"]) is None
    assert re_.rank_of_first_hit(chunks, ["9.9 Nothing", "6.2.3 Power OFF Procedure"]) == 3  # any judged section counts
    assert re_.rank_of_first_hit([{"section_title": "6.2.3 Power OFF Procedure"}], ["6.2.3"]) == 1  # plain dictionaries


# --- the metrics ---

CASES = [
    {
        "id": "a",
        "equipment_model": "thermal_station",
        "query": "qa",
        "relevant_sections": ["6.2.3 Power OFF"],
        "category": "procedure",
    },
    {
        "id": "b",
        "equipment_model": "thermal_station",
        "query": "qb",
        "relevant_sections": ["3.1.1 Electrical"],
        "category": "spec",
    },
    {
        "id": "c",
        "equipment_model": "thermal_retrofit_1kw",
        "query": "qc",
        "relevant_sections": ["5.3.5 Recipe"],
        "category": "gui",
    },
    {
        "id": "d",
        "equipment_model": "thermal_station",
        "query": "qd",
        "relevant_sections": ["2.5.6 Check"],
        "category": "safety",
    },
]
RESULTS = {
    "qa": [chunk("6.2.3 Power OFF Procedure")],  # exact, rank 1
    "qb": [chunk("1. Intro"), chunk("2. Safety"), chunk("3.1 Facility Requirements")],  # parent at rank 3: area only
    "qc": [
        chunk("9. A"),
        chunk("9.1 B"),
        chunk("9.2 C"),
        chunk("9.3 D"),
        chunk("9.4 E"),
        chunk("5.3.5 Recipe Screen", "thermal_retrofit_1kw"),
    ],
    "qd": [chunk("1. A"), chunk("2. B")],  # not found
}


def search(query, equipment, k):
    return RESULTS[query][:k]


def test_hit_rate_mrr_and_isolation(re_):
    report = re_.evaluate(CASES, search)
    assert report["questions"] == 4
    # strict: a hits at 1, c at 6 (outside the top 5), b and d miss
    assert report["hit_at_1"] == 0.25 and report["hit_at_3"] == 0.25 and report["hit_at_5"] == 0.25
    assert report["mrr"] == round((1 + 0 + 1 / 6 + 0) / 4, 3)
    # area: a at 1, b (parent) at 3, c at 6, d none
    assert report["area_hit_at_1"] == 0.25 and report["area_hit_at_3"] == 0.5 and report["area_hit_at_5"] == 0.5
    assert report["area_mrr"] == round((1 + 1 / 3 + 1 / 6 + 0) / 4, 3)
    assert report["isolation"] == 0.75  # query c returned passages of the other equipment
    assert [r["id"] for r in report["misses"]] == ["c", "d"]


def test_the_results_are_split_by_equipment_and_category(re_):
    report = re_.evaluate(CASES, search)
    assert report["by_equipment"]["thermal_station"]["questions"] == 3
    assert report["by_equipment"]["thermal_retrofit_1kw"]["questions"] == 1
    assert set(report["by_category"]) == {"procedure", "spec", "gui", "safety"}


def test_a_search_that_finds_nothing_scores_zero(re_):
    report = re_.evaluate(CASES[:1], lambda query, equipment, k: [])
    assert report["hit_at_5"] == 0.0 and report["mrr"] == 0.0 and report["isolation"] == 1.0  # nothing leaked


def test_no_cases_do_not_divide_by_zero(re_):
    assert re_.evaluate([], search)["mrr"] == 0.0


def test_the_retriever_is_asked_for_the_top_ten_of_the_selected_equipment(re_):
    seen = []

    def spy(query, equipment, k):
        seen.append((equipment, k))
        return []

    re_.evaluate(CASES, spy)
    assert {k for _, k in seen} == {re_.MAX_K}
    assert [e for e, _ in seen] == [c["equipment_model"] for c in CASES]


# --- the dataset and the runner ---


def test_the_judgements_cover_the_gold_set(re_):
    cases = re_.load_cases()
    gold = [
        json.loads(line)
        for line in (ROOT / "eval" / "datasets" / "gold_set.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [c["id"] for c in cases] == [g["id"] for g in gold]
    for case, g in zip(cases, gold):
        assert case["query"] == g["question"] and case["relevant_sections"] == [g["expected_section"]]
        assert case["equipment_model"] == g["equipment_model"]


def test_without_an_index_the_suite_is_skipped_not_failed(re_, monkeypatch):
    monkeypatch.setattr(re_, "make_search", lambda: None)
    report = re_.run()
    assert report["available"] is False and "index" in report["reason"]


def test_the_report_prints_for_both_a_run_and_a_skip(re_, capsys):
    re_.print_report(re_.evaluate(CASES, search))
    out = capsys.readouterr().out
    assert "strict" in out and "area" in out and "isolation" in out
    re_.print_report({"available": False, "reason": "no index"})
    assert "skipped" in capsys.readouterr().out


def test_the_targets_skip_an_unavailable_suite():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)
    targets = json.loads((ROOT / "eval" / "thresholds.json").read_text(encoding="utf-8"))
    assert set(targets["retrieval"]) == {"isolation", "area_hit_at_5", "area_mrr"}
    rows = run_eval.check_targets(
        {"retrieval": {"available": False, "reason": "no index"}}, {"retrieval": targets["retrieval"]}
    )
    assert rows == []
    rows = run_eval.check_targets(
        {"retrieval": {"available": True, "isolation": 1.0, "area_hit_at_5": 0.9, "area_mrr": 0.6}},
        {"retrieval": targets["retrieval"]},
    )
    assert {r["metric"]: r["met"] for r in rows} == {"isolation": True, "area_hit_at_5": True, "area_mrr": False}


def test_without_an_index_folder_nothing_is_loaded_or_created(re_, tmp_path, monkeypatch):
    """CI and a fresh clone have no index: the suite is skipped without the embedding model or an empty index."""
    monkeypatch.chdir(tmp_path)
    assert re_.make_search() is None
    assert not (tmp_path / "chroma_store").exists()
