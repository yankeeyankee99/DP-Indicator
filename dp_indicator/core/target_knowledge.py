from __future__ import annotations
import yaml
from pathlib import Path
from typing import Optional

# ── Fallback profile for targets with no YAML file ──
# Target-specific knowledge lives in config/profiles/*.yaml. This module supplies
# only the generic default used when no target profile exists.
TARGET_PROFILES = {
    "default": {
        "official_name": "Unknown target",
        "cell_type_expression": {},
        "functional_chain": [],
        "therapeutic_window": {"suppressed": [], "spared": []},
        "disease_categories": {},
        "excluded_indications": [],
        "key_publications": [],
    }
}

# ── YAML profile cache ──
_yaml_cache: dict[str, dict] = {}
_profiles_dir: Path | None = None


def _get_profiles_dir() -> Path:
    global _profiles_dir
    if _profiles_dir is None:
        _profiles_dir = Path(__file__).parent.parent / "config" / "profiles"
    return _profiles_dir


def _normalize_target_key(target: str) -> str:
    """Normalize target name for file lookup."""
    return target.lower().replace(".", "").replace("-", "").replace(" ", "_")


def _load_yaml_profile(target: str) -> Optional[dict]:
    """Load target profile from YAML file (hot-load). Returns None if not found."""
    global _yaml_cache
    if target in _yaml_cache:
        return _yaml_cache[target]

    profiles_dir = _get_profiles_dir()
    if not profiles_dir.exists():
        return None

    # Try exact match, then normalized match
    candidates = [
        profiles_dir / f"{target}.yaml",
        profiles_dir / f"{target}.yml",
        profiles_dir / f"{target.upper()}.yaml",
        profiles_dir / f"{_normalize_target_key(target)}.yaml",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    _yaml_cache[target] = data
                    return data
            except Exception:
                continue
    return None


def clear_yaml_cache():
    """Clear YAML cache to force re-load (useful after file edits)."""
    global _yaml_cache
    _yaml_cache = {}


def get_target_profile(target: str) -> dict:
    """Get target profile: YAML is the sole source of target-specific mechanism
    knowledge. If no YAML file matches `target`, this returns the fully generic
    "default" profile."""
    yaml_profile = _load_yaml_profile(target)
    if yaml_profile:
        return yaml_profile
    return TARGET_PROFILES.get(target, TARGET_PROFILES["default"])


def build_biological_context(target: str) -> str:
    profile = get_target_profile(target)
    if not profile or profile.get("official_name") == "Unknown target":
        return ""

    lines = [
        f"## Target Biological Profile: {target}",
        f"Official name: {profile.get('official_name', '')}",
        "",
        "### Differential Expression (Therapeutic Window Foundation)",
    ]
    for cell, info in profile.get("cell_type_expression", {}).items():
        lines.append(f"- {cell}: {info['level']} ({info.get('channels_per_cell', 'N/A')} channels/cell, {info['dependency']})")

    lines.extend([
        "",
        "### Functional Mechanism Chain",
        " → ".join(profile.get("functional_chain", [])),
        "",
        "### Therapeutic Selectivity",
        f"- Suppressed (pathogenic): {', '.join(profile.get('therapeutic_window', {}).get('suppressed', []))}",
        f"- Spared (protective): {', '.join(profile.get('therapeutic_window', {}).get('spared', []))}",
    ])

    disease_categories = profile.get("disease_categories", {})
    if disease_categories:
        lines.extend(["", "### Disease Category Mapping (Based on Target Biology)"])
        for category, info in disease_categories.items():
            lines.append(f"- {category}: {info.get('rationale', '')}")

    excluded = profile.get("excluded_indications", [])
    if excluded:
        lines.extend(["", "### Known/Established Indications (EXCLUDE from novel suggestions)", f"- {', '.join(excluded)}"])

    mechanism_scope = profile.get("mechanism_scope", {})
    if mechanism_scope.get("description"):
        lines.extend(["", "### Mechanism Scope Boundary (STRICT)", mechanism_scope["description"]])

    # Extended mechanistic bridges (from YAML or hardcoded extensions)
    bridges = profile.get("mechanistic_bridges", [])
    if bridges:
        lines.extend(["", "### Extended Mechanistic Bridges (Target-Specific)"])
        for bridge in bridges:
            # Render the internal snake_case axis identifier as a readable phrase.
            axis = str(bridge.get("axis", "unknown")).replace("_", " ").strip() or "unknown"
            mechanism = bridge.get("mechanism", "")
            cells = bridge.get("cell_types", bridge.get("cell_types_involved", []))
            lines.append(f"- **{axis}** ({', '.join(cells)}): {mechanism}")

    intersection_guidance = get_intersection_guidance(target)
    if intersection_guidance:
        lines.extend(["", "### Intersection Analysis Guidance", intersection_guidance])

    return "\n".join(lines)


def _load_target_extensions(target: str) -> dict:
    """Load optional target-specific extension knowledge."""
    profile = get_target_profile(target)
    if profile and profile.get("official_name") != "Unknown target":
        # If YAML profile contains extension fields, return them directly
        extensions = {}
        if "mechanistic_bridges" in profile:
            extensions["mechanistic_bridges"] = profile["mechanistic_bridges"]
        if "intersection_guidance" in profile:
            extensions["intersection_guidance"] = profile["intersection_guidance"]
        if "reasoning_guidance" in profile:
            extensions["reasoning_guidance"] = profile["reasoning_guidance"]
        if "mechanism_scope" in profile:
            extensions["mechanism_scope"] = profile["mechanism_scope"]
        if extensions:
            return extensions

    # Load an optional Python extension when one is installed.
    try:
        target_key = target.lower().replace(".", "").replace("-", "")
        if target_key == "kv13":
            from dp_indicator.data.kv13_mechanistic_context import get_kv13_extensions
            return get_kv13_extensions()
    except Exception:
        pass
    return {}


def build_mechanistic_bridge_context(target: str) -> str:
    extensions = _load_target_extensions(target)
    if not extensions:
        return ""
    lines = []
    bridges = extensions.get("mechanistic_bridges", [])
    if bridges:
        lines.append("")
        lines.append("### Extended Mechanistic Bridges (Target-Specific)")
        for bridge in bridges:
            axis = bridge.get("axis", "unknown")
            mechanism = bridge.get("mechanism", "")
            cells = bridge.get("cell_types_involved", bridge.get("cell_types", []))
            lines.append(f"- **{axis}**: {mechanism}")
    return "\n".join(lines)


def _procedural_bridge_search_terms(target: str, profile: dict) -> list[str]:
    """Build deterministic L6 bridge-layer queries from `mechanistic_bridges`.

    Queries are kept short because PubMed and Europe PMC treat multi-word input as
    an implicit AND. The generic templates are:
      (1) `{target} {axis}`                 — one query per axis
      (2) `{target} {single_cell_type}`     — one query per cell type in the axis
    Results are de-duplicated while preserving order. Returns [] when the profile
    has no mechanistic bridges.
    """
    terms: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        term = " ".join(term.split())  # collapse whitespace
        key = term.lower()
        if term and key not in seen:
            seen.add(key)
            terms.append(term)

    for bridge in profile.get("mechanistic_bridges", []):
        axis_words = str(bridge.get("axis", "")).replace("_", " ").strip()
        cells = bridge.get("cell_types", bridge.get("cell_types_involved", []))
        if axis_words:
            _add(f"{target} {axis_words}")
        for cell in cells:
            cell_words = str(cell).replace("_", " ").strip()
            if cell_words:
                _add(f"{target} {cell_words}")
    return terms


def get_bridge_search_terms(target: str) -> list[str]:
    """L6 bridge-layer retrieval queries. Procedurally derived from this target's
    `mechanistic_bridges` structure (see `_procedural_bridge_search_terms`) — no
    hand-authored `bridge_search_terms` field is read from YAML anymore, so there is
    no free-text list for a curator to skew toward a particular disease's vocabulary."""
    profile = get_target_profile(target)
    return _procedural_bridge_search_terms(target, profile)


def get_intersection_guidance(target: str) -> str:
    extensions = _load_target_extensions(target)
    return extensions.get("intersection_guidance", "")


def get_reasoning_guidance(target: str) -> list[str]:
    extensions = _load_target_extensions(target)
    return extensions.get("reasoning_guidance", [])


def get_out_of_scope_keywords(target: str) -> list[str]:
    """Return keywords for biological roles outside the configured mechanism scope.

    These keep associated-disease lookups focused on the role represented by the
    profile's cell-type expression and functional chain.
    """
    extensions = _load_target_extensions(target)
    scope = extensions.get("mechanism_scope", {})
    return scope.get("out_of_scope_keywords", [])
