import hashlib
import importlib
import importlib.util
import json
import re
from pathlib import Path

import dp_indicator
from dp_indicator.reporting.generator import ReportGenerator


ROOT = Path(__file__).resolve().parents[1]

FIXED_QUERY = "Explore new therapeutic indications for Kv1.3 pathway inhibitors"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_tree_digest(entries: list[dict]) -> str:
    lines = [
        f"{entry['path']}\t{entry['sha256'].lower()}"
        for entry in sorted(entries, key=lambda item: item["path"])
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


CURRENT_ENTRY_SCRIPTS = (
    "run_pipeline.py",
    "run_evidence_integrity_benchmark.py",
)
CURRENT_SOURCE_SUFFIXES = {".py", ".md", ".yaml", ".yml"}

# Match the legacy product name only; do not flag agent class names such as
# HypothesisCritic.
_LEGACY_PRODUCT_NAME = re.compile(r"Hypothesis\s+Explorer")
_HISTORY_LABEL_PATTERNS = (
    re.compile(r"\bv6\."),
    re.compile(r"\bv7(?:\.|\b)"),
    re.compile(r"\b7\.23\b"),
    re.compile(r"\b8\.3\b"),
    re.compile(r"\bP\d+\s+fix\b"),
    re.compile(r"\bTask\s+2\b"),
    re.compile(r"\bqwen3\.6-plus\b"),
    re.compile(r"\bkimi-k2\.5\b"),
)


def _current_source_paths() -> list[Path]:
    package_root = ROOT / "dp_indicator"
    assert package_root.is_dir(), "dp_indicator package directory must exist"
    paths = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix.lower() in CURRENT_SOURCE_SUFFIXES
    )
    for script_name in CURRENT_ENTRY_SCRIPTS:
        script_path = ROOT / script_name
        assert script_path.is_file(), f"missing current entry script: {script_name}"
        paths.append(script_path)
    readme_path = ROOT / "README.md"
    assert readme_path.is_file(), "missing public README.md"
    paths.append(readme_path)
    return paths


def test_public_package_identity_is_fix10():
    assert dp_indicator.__version__ == "fix10"
    assert importlib.util.find_spec("hypothesis_explorer") is None


def test_public_current_sources_drop_legacy_brand_and_history_labels():
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in _current_source_paths()
    )
    assert _LEGACY_PRODUCT_NAME.search(text) is None, (
        "user-visible legacy product name must be removed from current source"
    )
    for pattern in _HISTORY_LABEL_PATTERNS:
        match = pattern.search(text)
        assert match is None, f"historical version/fix label found: {match.group(0)!r}"


def test_report_generator_uses_current_brand_and_version(tmp_path):
    generator = ReportGenerator(output_dir=str(tmp_path))
    paths = generator.generate(
        hypotheses=[
            {
                "hypothesis_id": "H1",
                "indication": "Example Disease",
                "rank": 1,
                "overall_score": 0.5,
                "scores": {},
            }
        ],
        experiments=[],
        query={"target": "Kv1.3"},
    )
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "# DP-Indicator 报告" in markdown
    assert "*报告由 DP-Indicator fix10 生成*" in markdown

    html = Path(paths["html"]).read_text(encoding="utf-8")
    assert "<title>DP-Indicator Report</title>" in html
    assert '<h1 id="summary">DP-Indicator 报告</h1>' in html
    assert "DP-Indicator fix10" in html


def test_public_fixed_query_parser_is_packaged_and_usable():
    parser_path = ROOT / "dp_indicator" / "core" / "intent_parser.py"
    assert parser_path.is_file()

    parser = importlib.import_module("dp_indicator.core.intent_parser")
    query = parser.parse_query(FIXED_QUERY)

    assert query["raw_input"] == FIXED_QUERY
    assert query["target"] == "Kv1.3"
    assert query["direction"] == "target_to_indication"
    assert query["exploration_mode"] == "target_to_indication"
    assert query["needs_clarification"] is False


def test_public_manifest_covers_every_recorded_file_and_intent_parser():
    manifest_path = ROOT / "PUBLIC_RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    by_path = {entry["path"]: entry for entry in entries}

    assert "dp_indicator/core/intent_parser.py" in by_path
    for relative_path, entry in by_path.items():
        file_path = ROOT / Path(relative_path)
        assert file_path.is_file(), relative_path
        assert entry["byte_size"] == file_path.stat().st_size
        assert entry["sha256"] == _sha256_file(file_path)
        if entry["classification"] == "SOURCE_COPY":
            assert entry["source_sha256"] == entry["sha256"]

    tree = manifest["current_tree_digest"]
    assert tree["entry_count"] == len(entries)
    assert tree["sha256"] == _manifest_tree_digest(entries)
