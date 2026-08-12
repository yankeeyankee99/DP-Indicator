"""Schema type hints for DP-Indicator fix10.

Dataclasses are retained for documentation and static type hints.
The runtime pipeline uses dicts for evidence/hypothesis/experiment data
(for JSON serialization and performance). These classes are not instantiated.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
class EvidenceType(str, Enum):
    RCT_HUMAN = "RCT_human"
    CLINICAL_TRIAL = "clinical_trial"
    COHORT = "cohort"
    CASE_CONTROL = "case_control"
    GWAS = "gwas"
    EXPERT_CURATION = "expert_curation"
    ANIMAL = "animal"
    IN_VITRO = "in_vitro"
    DATABASE_ASSOCIATION = "database_association"
    LITERATURE = "literature"
    PREPRINT = "preprint"
    REVIEW = "review"
@dataclass
class Evidence:
    evidence_id: str
    source_db: str
    evidence_type: EvidenceType
    title: str
    abstract_snippet: str = ""
    publication_date: str = ""
    url: str = ""
    grade_score: int = 2
    grade_rating: str = "⊕⊕○○"
    relevance_to_target: str = "medium"
    inclusion: bool = True
    independence_group: str = "ungrouped"
    source_client: str = ""
    query_params: dict = field(default_factory=dict)
    raw_id: str = ""
    source_url: str = ""
    raw_metadata: dict = field(default_factory=dict)
@dataclass
class CausalLink:
    level: int
    from_node: str
    to_node: str
    relationship: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence_rationale: str = ""
    status: str = "inferred"
    source_text: str = ""
@dataclass
class CausalChain:
    indication: str
    links: list[CausalLink] = field(default_factory=list)
    overall_score: float = 0.0
@dataclass
class Hypothesis:
    hypothesis_id: str
    indication: str
    statement: str
    causal_chain: CausalChain
    evidence_ids: list[str] = field(default_factory=list)
    falsifiable_prediction: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    feasibility_score: float = 0.5
    feasibility_rationale: str = ""
    validation_stage: str = ""
    n_evidence: int = 0
    n_contradicting: int = 0
    n_missing_links: int = 0
    evidence_chain_trace: list[dict] = field(default_factory=list)
@dataclass
class Darkroom:
    darkroom_id: str
    hypothesis_id: str
    missing_link: str
    evidence_type_needed: str
    suggested_search: str = ""
@dataclass
class ExperimentProposal:
    experiment_id: str
    hypothesis_id: str
    title: str
    method: str
    expected_outcome: str
    falsifies_what: str = ""
    cost: str = ""
    duration: str = ""
    priority: str = "medium"
@dataclass
class AuditEvent:
    timestamp: str
    agent: str
    phase: str
    event_type: str
    payload: dict = field(default_factory=dict)
@dataclass
class RunMetadata:
    target: str
    direction: str
    wall_clock_seconds: float = 0.0
    cost_usd: float = 0.0
    per_model_cost_usd: dict = field(default_factory=dict)
    per_model_tokens: dict = field(default_factory=dict)
@dataclass
class ExplorationResult:
    query: dict
    hypotheses: list[Hypothesis] = field(default_factory=list)
    darkrooms: list[Darkroom] = field(default_factory=list)
    experiments: list[ExperimentProposal] = field(default_factory=list)
    audit_trail: list[AuditEvent] = field(default_factory=list)
    metadata: Optional[RunMetadata] = None
