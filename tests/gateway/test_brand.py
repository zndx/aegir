"""Deployment brand packs: Weathership / Cloudera / Custom."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from aegir.gateway import brand as B


def test_list_packs_order_and_shipped_logos() -> None:
    packs = {p["id"]: p for p in B.list_packs()}
    assert [p["id"] for p in B.list_packs()] == ["weathership", "cloudera", "custom"]
    assert packs["weathership"]["has_logo"]
    assert packs["cloudera"]["has_logo"]
    assert packs["weathership"]["logo_href"] == "/brand/weathership/logo.svg"
    assert packs["cloudera"]["logo_href"] == "/brand/cloudera/logo.svg"


def test_settings_bootstrap_weathership(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_UI_CONFIG", str(tmp_path / "aegir-ui.json"))
    monkeypatch.delenv("AEGIR_UI_BRAND", raising=False)
    payload = B.settings_payload()
    assert payload["brand_id"] == "weathership"
    assert payload["brand"]["has_logo"]


def test_apply_brand_persists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_UI_CONFIG", str(tmp_path / "aegir-ui.json"))
    out = B.apply_brand("cloudera")
    assert out["brand_id"] == "cloudera"
    saved = json.loads((tmp_path / "aegir-ui.json").read_text())
    assert saved["brand_id"] == "cloudera"
    assert B.load_settings()["brand_id"] == "cloudera"


def test_apply_empty_custom_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_UI_CONFIG", str(tmp_path / "aegir-ui.json"))
    monkeypatch.setenv("AEGIR_UI_BRAND_CUSTOM", str(tmp_path / "custom"))
    try:
        B.apply_brand("custom")
        raise AssertionError("empty custom must fail")
    except ValueError as e:
        assert "empty" in str(e).lower()


def test_ingest_flat_tgz(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AEGIR_UI_CONFIG", str(tmp_path / "aegir-ui.json"))
    dest = tmp_path / "custom"
    monkeypatch.setenv("AEGIR_UI_BRAND_CUSTOM", str(dest))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        logo = tmp_path / "logo.svg"
        logo.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        archive.add(logo, arcname="logo.svg")
        fav = tmp_path / "favicon.svg"
        fav.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
        archive.add(fav, arcname="favicon.svg")
    payload = B.ingest_tgz(buf.getvalue(), dest)
    assert payload["brand_id"] == "custom"
    assert dest.joinpath("logo.svg").is_file()
    assert dest.joinpath("favicon.svg").is_file()
    assert B.describe_pack("custom")["has_logo"]
