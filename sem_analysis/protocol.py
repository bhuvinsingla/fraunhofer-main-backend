"""Measurement protocol helpers (client-approved physical constants)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_PROTOCOL = {
    "approved": False,
    "approved_by": None,
    "notes": (
        "Proposed defaults pending client approval of Method 1 l, "
        "Method 2 fit band, and Method 3 circle diameter D."
    ),
    "method1_distances_nm": [25, 50, 100, 200],
    "method1_primary_nm": 100,
    "method2_fit_band_nm": [50, 200],
    "method3_circle_diameter_nm": 100,
}


def get_protocol(config: dict | None = None) -> dict[str, Any]:
    """Return merged protocol block from config."""
    proto = {**DEFAULT_PROTOCOL, **((config or {}).get("protocol") or {})}
    proto["method1_distances_nm"] = list(proto.get("method1_distances_nm") or [25, 50, 100, 200])
    proto["method2_fit_band_nm"] = list(proto.get("method2_fit_band_nm") or [50, 200])
    proto["method1_primary_nm"] = float(proto.get("method1_primary_nm", 100))
    proto["method3_circle_diameter_nm"] = float(proto.get("method3_circle_diameter_nm", 100))
    proto["approved"] = bool(proto.get("approved", False))
    return proto


def apply_protocol_overrides(config: dict, overrides: dict | None) -> dict:
    """
    Return a deep-copied config with protocol overrides applied.

    Supported keys: method1_distances_nm, method1_primary_nm,
    method2_fit_band_nm, method3_circle_diameter_nm, approved, approved_by.
    """
    cfg = deepcopy(config)
    if not overrides:
        return cfg

    proto = get_protocol(cfg)
    if "method1_distances_nm" in overrides and overrides["method1_distances_nm"] is not None:
        proto["method1_distances_nm"] = [float(x) for x in overrides["method1_distances_nm"]]
    if "method1_primary_nm" in overrides and overrides["method1_primary_nm"] is not None:
        proto["method1_primary_nm"] = float(overrides["method1_primary_nm"])
    if "method2_fit_band_nm" in overrides and overrides["method2_fit_band_nm"] is not None:
        band = list(overrides["method2_fit_band_nm"])
        proto["method2_fit_band_nm"] = [float(band[0]), float(band[1])]
    if "method3_circle_diameter_nm" in overrides and overrides["method3_circle_diameter_nm"] is not None:
        proto["method3_circle_diameter_nm"] = float(overrides["method3_circle_diameter_nm"])
    if "approved" in overrides and overrides["approved"] is not None:
        proto["approved"] = bool(overrides["approved"])
    if "approved_by" in overrides:
        proto["approved_by"] = overrides["approved_by"]

    cfg["protocol"] = proto
    # Keep measurement_methods in sync
    mm = cfg.setdefault("measurement_methods", {})
    mm.setdefault("fixed_distance_circle", {})["distances_nm"] = proto["method1_distances_nm"]
    mm.setdefault("fixed_distance_circle", {})["primary_nm"] = proto["method1_primary_nm"]
    mm.setdefault("projected_tip_distance", {})["fit_band_nm"] = proto["method2_fit_band_nm"]
    mm.setdefault("inscribed_angle", {})["circle_diameter_nm"] = proto["method3_circle_diameter_nm"]
    return cfg


def parse_protocol_form(
    *,
    method1_distances_nm: str | None = None,
    method1_primary_nm: float | None = None,
    method2_fit_lo_nm: float | None = None,
    method2_fit_hi_nm: float | None = None,
    method3_circle_diameter_nm: float | None = None,
    protocol_approved: bool | None = None,
    protocol_approved_by: str | None = None,
) -> dict:
    """Parse optional multipart form fields into a protocol overrides dict."""
    overrides: dict[str, Any] = {}
    if method1_distances_nm:
        parts = [p.strip() for p in method1_distances_nm.replace(";", ",").split(",") if p.strip()]
        overrides["method1_distances_nm"] = [float(p) for p in parts]
    if method1_primary_nm is not None:
        overrides["method1_primary_nm"] = float(method1_primary_nm)
    if method2_fit_lo_nm is not None and method2_fit_hi_nm is not None:
        overrides["method2_fit_band_nm"] = [float(method2_fit_lo_nm), float(method2_fit_hi_nm)]
    if method3_circle_diameter_nm is not None:
        overrides["method3_circle_diameter_nm"] = float(method3_circle_diameter_nm)
    if protocol_approved is not None:
        overrides["approved"] = bool(protocol_approved)
    if protocol_approved_by is not None and protocol_approved_by.strip():
        overrides["approved_by"] = protocol_approved_by.strip()
    return overrides
