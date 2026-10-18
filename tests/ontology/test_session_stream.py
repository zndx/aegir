# Copyright 2026 zndx
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from aegir.ontology.session_stream import concat, qualify


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


def test_concat_appends_in_seq_order() -> None:
    frames = [
        SimpleNamespace(seq=1, text="xxxx"),
        SimpleNamespace(seq=2, text="HELLO"),
    ]
    body, last = concat(frames)
    assert body == "xxxxHELLO"
    assert last == 2
