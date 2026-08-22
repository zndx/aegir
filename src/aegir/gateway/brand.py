"""Deployment-wide logo packs (Signals/Gaius branding, ported).

Theme stays Keiretsu. Product name stays Ægir. Packs only swap logo / favicon.
Settings persist under ``AEGIR_UI_CONFIG`` (default ``build/config/aegir-ui.json``).
"""
from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
from pathlib import Path

BRAND_WEATHERSHIP = "weathership"
BRAND_CLOUDERA = "cloudera"
BRAND_CUSTOM = "custom"
PACK_ORDER = (BRAND_WEATHERSHIP, BRAND_CLOUDERA, BRAND_CUSTOM)
MAX_TGZ_BYTES = 4 * 1024 * 1024
ALLOWED_SUFFIX = (".svg", ".json", ".md", ".png", ".css")


def repo_root() -> Path:
    raw = (os.environ.get("AEGIR_REPO_ROOT") or os.environ.get("DEVENV_ROOT") or "").strip()
    if raw:
        return Path(raw)
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / ".git").exists() and (cand / "src" / "aegir").exists():
            return cand
    return Path.cwd()


def shipped_brand_root() -> Path:
    env = (os.environ.get("AEGIR_UI_BRAND_ASSETS") or "").strip()
    if env:
        return Path(env)
    return repo_root() / "ui" / "public" / "brand"


def custom_dir() -> Path:
    env = (os.environ.get("AEGIR_UI_BRAND_CUSTOM") or "").strip()
    if env:
        return Path(env)
    return repo_root() / "build" / "dev" / ".aegir-ui-brand" / "custom"


def config_path() -> Path:
    env = (os.environ.get("AEGIR_UI_CONFIG") or "").strip()
    if env:
        return Path(env)
    return repo_root() / "build" / "config" / "aegir-ui.json"


def pack_dir(brand_id: str) -> Path:
    if brand_id == BRAND_CUSTOM:
        live = custom_dir()
        if (live / "logo.svg").is_file():
            return live
    return shipped_brand_root() / brand_id


def _read_manifest(directory: Path) -> dict:
    path = directory / "brand.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "id": directory.name,
        "display_name": "Custom" if directory.name == BRAND_CUSTOM else directory.name,
        "logo_file": "logo.svg",
        "logo_alt": "Organization",
    }


def describe_pack(brand_id: str) -> dict:
    directory = pack_dir(brand_id)
    manifest = _read_manifest(directory) if directory.is_dir() else {
        "id": brand_id,
        "display_name": brand_id.title(),
        "logo_file": "logo.svg",
        "logo_alt": brand_id,
    }
    logo_file = manifest.get("logo_file") or "logo.svg"
    has_logo = (directory / logo_file).is_file()
    has_light = (directory / "logo-on-light.svg").is_file()
    has_favicon = (directory / "favicon.svg").is_file()
    display = manifest.get("display_name") or brand_id.title()
    alt = manifest.get("logo_alt") or display
    href = f"/brand/{brand_id}/{logo_file}" if has_logo else ""
    return {
        "id": brand_id,
        "display_name": display,
        "logo_href": href,
        "logo_href_light": f"/brand/{brand_id}/logo-on-light.svg" if has_light else href,
        "favicon_href": f"/brand/{brand_id}/favicon.svg" if has_favicon else href,
        "logo_alt": alt,
        "has_logo": has_logo,
        "source": manifest.get("source") or "",
    }


def list_packs() -> list[dict]:
    out = []
    for brand_id in PACK_ORDER:
        pack = describe_pack(brand_id)
        if brand_id == BRAND_CUSTOM or pack["has_logo"] or (shipped_brand_root() / brand_id).is_dir():
            out.append(pack)
    return out


def load_settings() -> dict:
    path = config_path()
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            brand_id = str(data.get("brand_id") or "").strip()
            if brand_id:
                return {"brand_id": brand_id}
        except (OSError, json.JSONDecodeError):
            pass
    bootstrap = (os.environ.get("AEGIR_UI_BRAND") or BRAND_WEATHERSHIP).strip() or BRAND_WEATHERSHIP
    return {"brand_id": bootstrap}


def save_settings(brand_id: str) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"brand_id": brand_id}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def settings_payload() -> dict:
    settings = load_settings()
    packs = list_packs()
    selected = settings["brand_id"]
    by_id = {p["id"]: p for p in packs}
    current = by_id.get(selected)
    if current is None or (selected == BRAND_CUSTOM and not current["has_logo"]):
        selected = BRAND_WEATHERSHIP
        current = by_id.get(selected) or packs[0]
    return {
        "brand_id": selected,
        "brand": current,
        "brands": packs,
        "config_path": str(config_path()),
    }


def apply_brand(brand_id: str) -> dict:
    brand_id = (brand_id or "").strip()
    if brand_id not in PACK_ORDER:
        raise ValueError(f"unknown brand pack: {brand_id}")
    pack = describe_pack(brand_id)
    if brand_id == BRAND_CUSTOM and not pack["has_logo"]:
        raise ValueError("Custom pack is empty. Upload a .tgz first.")
    if brand_id != BRAND_CUSTOM and not pack["has_logo"]:
        raise ValueError(f"brand pack missing logo: {brand_id}")
    save_settings(brand_id)
    return settings_payload()


def _allowed_name(name: str) -> bool:
    lower = name.lower()
    if ".." in lower or lower.startswith("/"):
        return False
    return lower.endswith(ALLOWED_SUFFIX)


def _first_existing(root: Path, rels: tuple[str, ...]) -> Path | None:
    for rel in rels:
        path = root / rel
        if path.is_file():
            return path
    return None


def _locate_kit_root(staging: Path) -> Path | None:
    def is_kit(path: Path) -> bool:
        return (
            (path / "logo.svg").is_file()
            or (path / "logo" / "lockup" / "lockup-mono-white.svg").is_file()
            or (path / "brand.json").is_file()
        )

    if is_kit(staging):
        return staging
    try:
        children = list(staging.iterdir())
    except OSError:
        return None
    for child in children:
        if child.is_dir() and is_kit(child):
            return child
        brand = child / "brand"
        if brand.is_dir() and is_kit(brand):
            return brand
    return None


def ingest_tgz(payload: bytes, dest: Path | None = None) -> dict:
    if len(payload) > MAX_TGZ_BYTES:
        raise ValueError(f"pack larger than {MAX_TGZ_BYTES} bytes")
    dest = dest or custom_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aegir-brand-") as tmp:
        staging = Path(tmp) / "in"
        staging.mkdir()
        try:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    name = member.name
                    if not _allowed_name(name):
                        continue
                    target = staging / name
                    if not str(target.resolve()).startswith(str(staging.resolve())):
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    extracted = archive.extractfile(member)
                    if extracted is None:
                        continue
                    target.write_bytes(extracted.read())
        except tarfile.TarError as exc:
            raise ValueError(f"invalid brand pack: {exc}") from exc

        kit = _locate_kit_root(staging)
        if kit is None:
            raise ValueError(
                "archive is not a brand pack. Need logo.svg + favicon.svg, or a "
                "Weathership kit (logo/lockup/lockup-mono-white.svg + favicon/favicon.svg)"
            )
        logo_src = _first_existing(
            kit,
            ("logo.svg", "logo/lockup/lockup-mono-white.svg", "logo/lockup/lockup-horizontal.svg"),
        )
        if logo_src is None:
            raise ValueError("brand pack missing logo.svg")
        out = Path(tmp) / "normalized"
        out.mkdir()
        (out / "logo.svg").write_bytes(logo_src.read_bytes())
        fav = _first_existing(kit, ("favicon.svg", "favicon/favicon.svg"))
        (out / "favicon.svg").write_bytes((fav or logo_src).read_bytes())
        mark = _first_existing(kit, ("mark.svg", "logo/mark/mark-mono-white.svg", "logo/mark/mark.svg"))
        if mark is not None:
            (out / "mark.svg").write_bytes(mark.read_bytes())
        if (kit / "brand.json").is_file():
            (out / "brand.json").write_bytes((kit / "brand.json").read_bytes())
        else:
            (out / "brand.json").write_text(
                json.dumps({
                    "id": "custom",
                    "display_name": "Custom",
                    "logo_file": "logo.svg",
                    "logo_alt": "Organization",
                    "source": "uploaded tgz",
                })
                + "\n",
                encoding="utf-8",
            )
        if dest.exists():
            import shutil
            shutil.rmtree(dest)
        import shutil
        shutil.copytree(out, dest)
    save_settings(BRAND_CUSTOM)
    return settings_payload()


def custom_file(name: str) -> Path | None:
    safe = Path(name).name
    if safe != name or ".." in name:
        return None
    path = custom_dir() / safe
    return path if path.is_file() else None
