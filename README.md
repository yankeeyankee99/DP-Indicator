# DP-Indicator: multi-agent target–indication exploration

DP-Indicator is a multi-agent workflow for literature-grounded
target–indication exploration. It organizes mechanistic hypotheses as
multi-axis L1–L5 causal chains, assigns node-level evidence states
(`supported` / `inferred` / `hypothesized`), triggers targeted secondary
retrieval at weak nodes, and ranks candidates with G1–G4 scores.

Importable Python package name: `dp_indicator`.

## Paper

Manuscript (working title): *Can Multi-Agent LLMs Identify Drug Indications?
A Dry-Wet Combined Evaluation from Kv1.3 to IgA Nephropathy*
(Kai Yan, Wei Guo; China Pharmaceutical University).

This public release corresponds to the software described in that manuscript.

## Cite

Release tag: **v1.0.0**

- GitHub: https://github.com/yankeeyankee99/DP-Indicator
- Release: https://github.com/yankeeyankee99/DP-Indicator/releases/tag/v1.0.0
- Zenodo DOI: *(add after this GitHub release is archived to Zenodo)*
- Preferred citation metadata: see [`CITATION.cff`](CITATION.cff)

Please cite both the paper (when available) and this software release (DOI).

## Install

Python 3.12 or later is required. From the repository root:

```text
python -m pip install -r requirements-lock.txt
```

## Quick start

Set a valid Bohrium OpenAPI credential:

```text
# Windows PowerShell
$env:BH_API_KEY = "YOUR_KEY"

# bash
export BH_API_KEY=YOUR_KEY
```

Then run the fixed-query entry point:

```text
python run_pipeline.py
```

The entry point always submits this exact English query (do not change it if
you intend to reproduce the manuscript entry condition):

```text
Explore new therapeutic indications for Kv1.3 pathway inhibitors
```

A complete run requires outbound network access and paid external model/API
calls. It may take substantial time and incur provider charges.

Equivalent CLI form:

```text
python -m dp_indicator run-all "Explore new therapeutic indications for Kv1.3 pathway inhibitors"
```

## What a run produces

- Stage checkpoints: `checkpoints/stages/`
- Final reports: `reports/report.json`, `reports/report.md`, `reports/report.html`

These output directories are git-ignored; they are created locally at runtime.

## Offline tests

```text
python -m pytest
```

Tests do not require `BH_API_KEY` and do not intentionally make paid model
calls or network requests.

## Architecture (short)

The Orchestrator runs:

`parse_intent` → `explore` → `hypothesize` → `design` → `generate_report`

Thirteen specialized agents and ten biomedical resource clients are included
under `dp_indicator/`. Reference defaults for model routing, G1–G4 weights,
and database endpoints are documented in `dp_indicator/config/settings.yaml`.
Runtime routing is implemented in `dp_indicator/core/model_router.py`.

Prompt files under `dp_indicator/prompts/` include `reasoner.md` (loaded by
ReasonerAgent) and `critic.md` (GRADE rubric documentation). Other agent
system prompts are embedded in the Python sources.

## Scope / not included

Included: source code, fixed-query entry point, tests, dependency locks,
reference settings, version-controlled prompt templates, and frozen benchmark
derivatives listed in `PUBLIC_RELEASE_MANIFEST.json`.

Not redistributed: raw paid request payloads, provider-side prompts, private
run logs, unpublished checkpoints, copyrighted publisher full text, and
provider-internal model revisions.

Do not commit `BH_API_KEY`, local checkpoints, paid run logs, or full-text
caches.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
