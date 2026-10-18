# Copyright 2026 zndx
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from aegir.ontology.session_stream import concat, qualify, select_admissions, stable_indexes


def test_one_window_is_offsets_not_a_copy_of_the_text() -> None:
    text = "xxxxHELLO"

    def windows(body: str):
        assert body == text
        return [(4, 9)]

    def query(segment: str):
        assert segment == "HELLO"
        return [
            {"score": 0.9, "code": "9", "iri": "https://signals.zndx.org/sdg#LIMS", "pref_label": "LIMS"},
            {"score": 0.4, "code": "2", "iri": "https://signals.zndx.org/sdg#OTHER", "pref_label": "Other"},
        ]

    kept = qualify(text, windows=windows, query=query, exclude_codes={"ROOT"})
    assert kept == [
        {
            "start": 4,
            "end": 9,
            "genus_iri": "https://signals.zndx.org/sdg#LIMS",
            "genus_code": "9",
            "genus_label": "LIMS",
            "rel_margin_h": 0.5556,
            "tau": 0.10,
        }
    ]
    assert "text" not in kept[0]


def test_a_root_hit_is_not_returned() -> None:
    def windows(_body: str):
        return [(0, 4)]

    def query(_segment: str):
        return [{"score": 0.9, "code": "ROOT", "iri": "https://signals.zndx.org/sdg#ROOT", "pref_label": "Root"}]

    assert qualify("xxxx", windows=windows, query=query, exclude_codes={"ROOT"}) == []


def test_a_full_neighbor_makes_the_earlier_window_stable() -> None:
    assert stable_indexes(767, 2, flush=False) == set()
    assert stable_indexes(768, 2, flush=False) == {0}
    assert stable_indexes(1024, 2, flush=False) == {0, 1}
    assert stable_indexes(1, 2, flush=True) == {0, 1}


def test_flush_keeps_the_higher_margin_overlap() -> None:
    text = "AAAAABBBBB"
    spans = [(0, 8), (4, 10)]

    def query(segment: str):
        if segment == text[0:8]:
            return [
                {"score": 0.5, "code": "A", "iri": "iri:A", "pref_label": "A"},
                {"score": 0.4, "code": "B", "iri": "iri:B", "pref_label": "B"},
            ]
        return [
            {"score": 0.9, "code": "B", "iri": "iri:B", "pref_label": "B"},
            {"score": 0.2, "code": "A", "iri": "iri:A", "pref_label": "A"},
        ]

    assert select_admissions(
        text, spans, n_tokens=10, flush=False, query=query, exclude_codes=set(),
    ) == []
    flushed = select_admissions(
        text, spans, n_tokens=10, flush=True, query=query, exclude_codes=set(),
    )
    assert [(row["start"], row["end"], row["genus_iri"]) for row in flushed] == [(4, 10, "iri:B")]
    assert "text" not in flushed[0]


def test_flush_drops_a_root_only_window() -> None:
    def query(_segment: str):
        return [{"score": 0.9, "code": "ROOT", "iri": "iri:ROOT", "pref_label": "Root"}]

    assert select_admissions(
        "xxxx", [(0, 4)], n_tokens=4, flush=True, query=query, exclude_codes={"ROOT"},
    ) == []


def test_concat_appends_in_seq_order() -> None:
    frames = [
        SimpleNamespace(seq=1, text="xxxx"),
        SimpleNamespace(seq=2, text="HELLO"),
    ]
    body, last = concat(frames)
    assert body == "xxxxHELLO"
    assert last == 2
