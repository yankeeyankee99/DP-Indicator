import json
import re
import tempfile
import unittest
from pathlib import Path

from dp_indicator.reporting.generator import ReportGenerator


class VerificationReportingTests(unittest.TestCase):
    def test_report_preserves_raw_final_and_sanitization_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
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
                query={
                    "target": "Kv1.3",
                    "_verification_report": {"v1_id_existence": {}},
                    "_verification_report_raw": {
                        "v1_id_existence": {
                            "PMID:bad": {"exists": False}
                        }
                    },
                    "_verification_adjustments": {"H1": -0.15},
                    "_verification_sanitization": [
                        {
                            "hypothesis_id": "H1",
                            "action": "remove_citation",
                            "evidence_id": "PMID:bad",
                        }
                    ],
                },
            )

            report = json.loads(
                Path(paths["json"]).read_text(encoding="utf-8")
            )

        self.assertEqual(
            report["verification_report"],
            {"v1_id_existence": {}},
        )
        self.assertEqual(
            report["verification_report_raw"]["v1_id_existence"][
                "PMID:bad"
            ]["exists"],
            False,
        )
        self.assertEqual(
            report["verification_adjustments"],
            {"H1": -0.15},
        )
        self.assertEqual(
            report["verification_sanitization"][0]["action"],
            "remove_citation",
        )
        self.assertNotIn("_verification_report_raw", report["query"])

    def test_markdown_prefers_verified_span_over_generated_source_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Example Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "verification": {
                            "claim_grounding": [
                                {
                                    "claim_id": "CLM-1",
                                    "verdict": "supported",
                                }
                            ]
                        },
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "primary",
                                    "steps": [
                                        {
                                            "layer": "L1",
                                            "mechanism": "A mechanistic claim.",
                                            "status": "supported",
                                            "evidence_ids": ["PMID:1"],
                                            "source_text": (
                                                "model-generated paraphrase"
                                            ),
                                            "verified_spans": [
                                                {
                                                    "evidence_id": "PMID:1",
                                                    "quote": (
                                                        "exact abstract words"
                                                    ),
                                                    "quote_verified": True,
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ],
                experiments=[],
                query={"target": "Kv1.3"},
            )

            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")

        self.assertIn("核验原文", markdown)
        self.assertIn("exact abstract words", markdown)
        self.assertIn("Claim核验", markdown)
        self.assertIn("supported=1", markdown)
        self.assertNotIn("原文: \"model-generated paraphrase", markdown)
        self.assertIn("核验原文", html)
        self.assertIn("exact abstract words", html)
        self.assertIn("Claim核验", html)
        self.assertNotIn("原文: \"model-generated paraphrase", html)

    def test_bridge_evidence_labels_and_gap_summary_in_markdown_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Example Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "primary",
                                    "steps": [
                                        {
                                            "layer": "L3",
                                            "mechanism": "Bridge-supported step.",
                                            "status": "supported",
                                            "sources": [
                                                {
                                                    "evidence_id": "PMID:direct",
                                                    "first_author": "Direct",
                                                    "journal_short": "Direct J",
                                                    "year": 2020,
                                                    "pmid": "999",
                                                },
                                                {
                                                    "evidence_id": "PMID:1",
                                                    "evidence_role": "bridge_evidence",
                                                    "retrieval_reason": (
                                                        "uncited_causal_gap"
                                                    ),
                                                    "first_author": "Bridge",
                                                    "journal_short": "Bridge J",
                                                    "year": 2021,
                                                    "pmid": "1",
                                                },
                                            ],
                                            "verified_spans": [
                                                {
                                                    "evidence_id": "PMID:1",
                                                    "quote": "bridge quote text",
                                                    "quote_verified": True,
                                                    "evidence_role": (
                                                        "bridge_evidence"
                                                    ),
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={
                    "target": "Kv1.3",
                    "_verification_summary": "verification complete",
                    "_verification_report": {
                        "gap_retrieval": {
                            "groups": [],
                            "evidence": [],
                            "summary": {
                                "gap_parent_steps": 19,
                                "searched_parent_steps": 19,
                                "resolved_parent_steps": 8,
                                "unresolved_atomic_claims": 52,
                                "errors": [],
                            },
                        },
                    },
                },
            )

            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")

        self.assertIn("补充桥接证据", markdown)
        self.assertIn("PMID:1", markdown)
        self.assertIn("缺口步骤: 19；已补证: 8", markdown)
        self.assertIn("已检索: 19", markdown)
        self.assertIn("未解决原子Claim: 52", markdown)
        self.assertIn('class="bridge-evidence"', html)
        self.assertIn("PMID:1", html)
        self.assertIn("缺口步骤: 19；已补证: 8", html)
        self.assertNotIn("PMID:999（补充桥接证据）", markdown)
        self.assertNotIn(
            'PMID:999"><span class="bridge-evidence">',
            html,
        )
        direct_md_line = next(
            line
            for line in markdown.splitlines()
            if "PMID:999" in line and "**来源**" in line
        )
        self.assertNotIn("补充桥接证据", direct_md_line)

    def test_gap_summary_shows_zero_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
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
                query={
                    "target": "Kv1.3",
                    "_verification_summary": "verification complete",
                    "_verification_report": {
                        "gap_retrieval": {
                            "groups": [],
                            "evidence": [],
                            "summary": {
                                "gap_parent_steps": 0,
                                "searched_parent_steps": 0,
                                "resolved_parent_steps": 0,
                                "unresolved_atomic_claims": 0,
                                "errors": [],
                            },
                        },
                    },
                },
            )

            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")

        self.assertIn("缺口步骤: 0", markdown)
        self.assertIn("已补证: 0", markdown)
        self.assertIn("已检索: 0", markdown)
        self.assertIn("未解决原子Claim: 0", markdown)
        self.assertIn("缺口步骤: 0", html)
        self.assertIn("已补证: 0", html)

    def test_gap_acceptance_metrics_report_empirical_shortfall(self):
        summary = {
            "gap_parent_steps": 1,
            "searched_parent_steps": 1,
            "resolved_parent_steps": 1,
            "unresolved_atomic_claims": 12,
            "uncited_atomic_before": 13,
            "supported_or_partial_after": 1,
            "coverage_gain": 1 / 13,
            "acceptance_threshold": 0.30,
            "acceptance_passed": False,
            "errors": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = ReportGenerator(output_dir=tmp).generate(
                hypotheses=[{
                    "hypothesis_id": "H1",
                    "indication": "Example Disease",
                    "rank": 1,
                    "overall_score": 0.5,
                    "scores": {},
                }],
                experiments=[],
                query={
                    "target": "Kv1.3",
                    "_verification_report": {
                        "gap_retrieval": {"summary": summary}
                    },
                },
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")

        for content in (markdown, html):
            self.assertIn("uncited_atomic_before: 13", content)
            self.assertIn("supported_or_partial_after: 1", content)
            self.assertIn("coverage_gain: 7.7%", content)
            self.assertIn("acceptance_threshold: 30.0%", content)
            self.assertIn("实证覆盖不足", content)

    def test_direct_source_line_is_not_labeled_bridge_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Example Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "primary",
                                    "steps": [
                                        {
                                            "layer": "L3",
                                            "mechanism": "Mixed evidence step.",
                                            "status": "supported",
                                            "sources": [
                                                {
                                                    "evidence_id": "PMID:direct",
                                                    "first_author": "Direct",
                                                    "journal_short": "Direct J",
                                                    "year": 2020,
                                                    "pmid": "999",
                                                },
                                                {
                                                    "evidence_id": "PMID:1",
                                                    "evidence_role": "bridge_evidence",
                                                    "pmid": "1",
                                                },
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={"target": "Kv1.3"},
            )

            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")

        direct_md_line = next(
            line
            for line in markdown.splitlines()
            if "PMID:999" in line and "**来源**" in line
        )
        bridge_md_line = next(
            line
            for line in markdown.splitlines()
            if "[PMID:1]" in line and "补充桥接证据" in line
        )
        self.assertNotIn("补充桥接证据", direct_md_line)
        self.assertIn("补充桥接证据", bridge_md_line)

        direct_html_line = next(
            line
            for line in html.splitlines()
            if "PMID:999" in line
        )
        bridge_html_line = next(
            line
            for line in html.splitlines()
            if "PMID:1" in line and "bridge-evidence" in line
        )
        self.assertNotIn("bridge-evidence", direct_html_line)
        self.assertIn('class="bridge-evidence"', bridge_html_line)

    def test_pmid_only_source_renders_in_markdown_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Example Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "primary",
                                    "steps": [
                                        {
                                            "layer": "L4",
                                            "mechanism": "Uncited gap step.",
                                            "status": "inferred",
                                            "sources": [
                                                {
                                                    "evidence_id": "PMID:42",
                                                    "evidence_role": "bridge_evidence",
                                                    "pmid": "42",
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={"target": "Kv1.3"},
            )

            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")

        self.assertIn("[PMID:42]", markdown)
        self.assertIn("补充桥接证据", markdown)
        self.assertIn("[PMID:42]", html)
        self.assertIn('class="bridge-evidence"', html)

    def test_html_escapes_malicious_dynamic_content(self):
        payload = '<script>alert("x")</script>'
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Example Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "primary",
                                    "steps": [
                                        {
                                            "layer": "L1",
                                            "mechanism": payload,
                                            "status": "supported",
                                            "sources": [
                                                {
                                                    "evidence_id": "PMID:1",
                                                    "first_author": payload,
                                                    "journal_short": payload,
                                                    "year": 2020,
                                                    "pmid": "1",
                                                }
                                            ],
                                            "verified_spans": [
                                                {
                                                    "evidence_id": "PMID:1",
                                                    "quote": payload,
                                                    "quote_verified": True,
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={
                    "target": "Kv1.3",
                    "_verification_summary": payload,
                    "_verification_report": {
                        "gap_retrieval": {
                            "summary": {
                                "gap_parent_steps": 1,
                                "searched_parent_steps": 1,
                                "resolved_parent_steps": 0,
                                "unresolved_atomic_claims": 0,
                            },
                        },
                    },
                },
            )

            html = Path(paths["html"]).read_text(encoding="utf-8")

        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;script&gt;alert", html)

    def test_markdown_safely_renders_verification_summary_and_inline_text(self):
        summary = "safe line\n```\n## injected heading\n**bold**"
        quote = "line1\nline2 `code`"
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Example Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "primary",
                                    "steps": [
                                        {
                                            "layer": "L1",
                                            "mechanism": "**not a heading**",
                                            "status": "supported",
                                            "verified_spans": [
                                                {
                                                    "evidence_id": "PMID:1",
                                                    "quote": quote,
                                                    "quote_verified": True,
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={
                    "target": "Kv1.3",
                    "_verification_summary": summary,
                },
            )

            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        verification_block_start = markdown.index("## 证据验证报告")
        verification_block = markdown[verification_block_start:]
        self.assertNotIn("\n```\n", verification_block)
        self.assertIn("safe line", verification_block)
        self.assertIn("injected heading", verification_block)
        self.assertIn("            line1", markdown)
        self.assertIn("            line2 `code`", markdown)
        self.assertIn("    ```", verification_block)
        self.assertNotIn("line1 line2", markdown)
        self.assertNotRegex(
            markdown,
            r"(?m)^ {0,11}<script>",
        )
        self.assertIn("*报告由 DP-Indicator fix10 生成*", markdown)

    def test_verified_quote_uses_twelve_space_nested_code_block(self):
        quote = "intro\n<script>alert(1)</script>\n"
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Example Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "primary",
                                    "steps": [
                                        {
                                            "layer": "L1",
                                            "mechanism": "Step",
                                            "status": "supported",
                                            "verified_spans": [
                                                {
                                                    "evidence_id": "PMID:1",
                                                    "quote": quote,
                                                    "quote_verified": True,
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={"target": "Kv1.3"},
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertRegex(
            markdown,
            r"(?m)^ {12}<script>alert\(1\)</script>$",
        )
        self.assertRegex(markdown, r"(?m)^ {12}intro$")
        self.assertNotRegex(markdown, r"(?m)^ {0,11}<script>")

    def test_doi_source_id_renders_safely_in_markdown_and_html(self):
        doi = "10.1000/x"
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Example Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "primary",
                                    "steps": [
                                        {
                                            "layer": "L1",
                                            "mechanism": "DOI source step.",
                                            "status": "supported",
                                            "sources": [
                                                {
                                                    "evidence_id": f"DOI:{doi}",
                                                    "doi": doi,
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={"target": "Kv1.3"},
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")

        self.assertIn("[DOI:10\\.1000/x]", markdown)
        self.assertIn("10.1000/x", html)
        self.assertNotIn("10\\.1000", html)
        self.assertIn("[DOI:10.1000/x]", html)

    def test_empty_verification_report_dict_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
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
                query={
                    "target": "Kv1.3",
                    "_verification_report": {},
                },
            )
            report = json.loads(
                Path(paths["json"]).read_text(encoding="utf-8")
            )

        self.assertIn("verification_report", report)
        self.assertEqual(report["verification_report"], {})

    def test_invalid_gap_summary_values_are_not_rendered(self):
        base = {
            "gap_parent_steps": 1,
            "searched_parent_steps": 1,
            "resolved_parent_steps": 0,
            "unresolved_atomic_claims": 0,
        }
        invalid_cases = [
            {**base, "gap_parent_steps": -1},
            {**base, "gap_parent_steps": "1"},
            {**base, "gap_parent_steps": True},
            {**base, "gap_parent_steps": None},
            {**base, "resolved_parent_steps": 1.5},
        ]
        for summary in invalid_cases:
            with self.subTest(summary=summary):
                with tempfile.TemporaryDirectory() as tmp:
                    generator = ReportGenerator(output_dir=tmp)
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
                        query={
                            "target": "Kv1.3",
                            "_verification_summary": "verification complete",
                            "_verification_report": {
                                "gap_retrieval": {"summary": summary},
                            },
                        },
                    )
                    markdown = Path(paths["markdown"]).read_text(
                        encoding="utf-8"
                    )
                    html = Path(paths["html"]).read_text(encoding="utf-8")

                self.assertNotIn("缺口步骤:", markdown)
                self.assertNotIn("缺口证据检索", markdown)
                self.assertNotIn("缺口步骤:", html)

    def test_html_escapes_all_dynamic_report_paths(self):
        payload = '<img src=x onerror=alert(1)>'
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": payload,
                        "indication": payload,
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "verification": {
                            "claim_grounding": [
                                {
                                    "claim_id": "CLM-1",
                                    "verdict": payload,
                                }
                            ]
                        },
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": payload,
                                    "steps": [
                                        {
                                            "layer": payload,
                                            "mechanism": payload,
                                            "status": "supported",
                                            "verified_spans": [
                                                {
                                                    "evidence_id": "PMID:1",
                                                    "quote": payload,
                                                    "quote_verified": True,
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                            "cross_talk": [
                                {"description": payload},
                            ],
                        },
                    }
                ],
                experiments=[],
                query={
                    "target": payload,
                    "_degradation_notes": [payload],
                    "_verification_summary": payload,
                },
            )
            html = Path(paths["html"]).read_text(encoding="utf-8")

        self.assertNotIn("<img src=x onerror=alert(1)>", html)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", html)
        self.assertNotIn(payload, html)

    def test_html_escapes_legacy_chain_and_cross_talk(self):
        payload = '<script>legacy</script>'
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H-legacy",
                        "indication": "Legacy Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": {
                            "links": [
                                {
                                    "level": "3",
                                    "from_node": payload,
                                    "to_node": payload,
                                    "relationship": payload,
                                    "status": "inferred",
                                    "evidence_ids": [payload],
                                    "source_text": payload,
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={"target": "Kv1.3"},
            )
            html = Path(paths["html"]).read_text(encoding="utf-8")

        self.assertNotIn("<script>legacy</script>", html)
        self.assertIn("&lt;script&gt;legacy&lt;/script&gt;", html)

    def test_mermaid_uses_strict_security_and_sanitizes_labels(self):
        injection = '"]; A-->B["injected'
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Example Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "primary",
                                    "steps": [
                                        {
                                            "layer": "L1",
                                            "mechanism": injection,
                                            "status": "supported",
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={"target": "Kv1.3"},
            )
            html = Path(paths["html"]).read_text(encoding="utf-8")

        self.assertIn("securityLevel: 'strict'", html)
        mermaid_block = re.search(
            r'<pre class="mermaid">\n(.*?)\n</pre>',
            html,
            re.DOTALL,
        )
        self.assertIsNotNone(mermaid_block)
        diagram = mermaid_block.group(1)
        self.assertNotIn('A-->B["injected', diagram)
        self.assertNotIn("<script>", diagram.lower())

    def test_markdown_inline_escapes_html_and_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Example Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "primary",
                                    "steps": [
                                        {
                                            "layer": "L1",
                                            "mechanism": (
                                                "## heading & <tag>"
                                            ),
                                            "status": "supported",
                                            "sources": [
                                                {
                                                    "evidence_id": "PMID:1",
                                                    "first_author": "A&B",
                                                    "journal_short": "J",
                                                    "year": 2020,
                                                    "pmid": "1",
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={"target": "Kv1.3"},
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertIn("&lt;tag&gt;", markdown)
        self.assertIn("A&amp;B", markdown)
        self.assertNotIn("<tag>", markdown)
        self.assertIn("\\#\\# heading", markdown)

    def test_markdown_sanitizes_all_major_dynamic_paths(self):
        html_payload = '<script>alert(1)</script>'
        md_payload = (
            '## injected\n[evil](http://evil.com) `code` **bold** _x_'
        )
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": md_payload,
                        "indication": md_payload,
                        "rank": md_payload,
                        "overall_score": 0.5,
                        "feasibility_score": 0.5,
                        "statement": md_payload,
                        "falsifiable_prediction": md_payload,
                        "scores": {"G1": md_payload, "_error": md_payload},
                        "scoring_method": "failed",
                        "supporting_evidence": [
                            {
                                "evidence_type": "literature",
                                "source_db": md_payload,
                                "grade_score": 1,
                                "title": md_payload,
                            }
                        ],
                        "verification": {
                            "claim_grounding": [
                                {
                                    "claim_id": "CLM-1",
                                    "verdict": md_payload,
                                }
                            ],
                            "v2_citation_issues": [
                                {
                                    "severity": "high",
                                    "evidence_id": md_payload,
                                    "issue": md_payload,
                                }
                            ],
                            "score_adjustment": -0.1,
                        },
                        "critic_review": {
                            "fatal_weakness": {"reasoning": md_payload},
                            "suggested_fix": md_payload,
                        },
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": md_payload,
                                    "steps": [
                                        {
                                            "layer": md_payload,
                                            "mechanism": md_payload,
                                            "status": md_payload,
                                            "source_text": md_payload,
                                        }
                                    ],
                                }
                            ],
                            "cross_talk": [
                                {"description": md_payload},
                            ],
                        },
                        "meta_reflection": {
                            "uncertainty_ranking": {
                                "hypothesis": md_payload,
                                "reason": md_payload,
                            },
                            "evidence_gap": {
                                "what": md_payload,
                                "why": md_payload,
                            },
                            "interconnections": md_payload,
                            "potential_biases": [md_payload],
                            "counter_intuitive": {
                                "finding": md_payload,
                                "assessment": md_payload,
                            },
                            "recommended_next_step": md_payload,
                            "overall_confidence": md_payload,
                        },
                    }
                ],
                experiments=[
                    {
                        "experiment_id": md_payload,
                        "title": md_payload,
                        "model": md_payload,
                        "intervention": md_payload,
                        "readout": md_payload,
                        "control": md_payload,
                        "priority": md_payload,
                        "expected_supporting": md_payload,
                        "expected_refuting": md_payload,
                        "timeline": md_payload,
                        "rationale": md_payload,
                    }
                ],
                query={
                    "target": html_payload,
                    "direction": md_payload,
                    "_degradation_notes": [html_payload],
                    "_retrieval_log": [
                        {
                            "type": "focus",
                            "db": md_payload,
                            "term": md_payload,
                            "n_results": 1,
                        }
                    ],
                    "_verification_summary": "summary block preserved",
                },
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertNotIn("<script>", markdown)
        self.assertNotIn("[evil](http://evil.com)", markdown)
        self.assertNotIn("\n## injected\n", markdown)
        self.assertIn("&lt;script&gt;alert\\(1\\)&lt;/script&gt;", markdown)
        self.assertIn("summary block preserved", markdown)

    def test_markdown_sanitizes_legacy_chain_fields(self):
        payload = "## legacy *bold* [x](http://evil)"
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Legacy Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": [
                            {
                                "layer": payload,
                                "mechanism": payload,
                                "status": payload,
                            }
                        ],
                    }
                ],
                experiments=[],
                query={"target": "Kv1.3"},
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertNotIn("[x](http://evil)", markdown)
        self.assertIn("\\#\\# legacy", markdown)

    def test_markdown_preserves_readable_chinese_and_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "银屑病",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "statement": "Kv1.3 抑制 T 细胞（α-干扰素通路）",
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "主路径",
                                    "steps": [
                                        {
                                            "layer": "L1",
                                            "mechanism": "p<0.05 显著",
                                            "status": "supported",
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={"target": "Kv1.3"},
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertIn("银屑病", markdown)
        self.assertIn("Kv1\\.3", markdown)
        self.assertIn("干扰素", markdown)
        self.assertIn("α\\-干扰素", markdown)
        self.assertIn("p&lt;0\\.05", markdown)

    def test_verified_quote_preserves_multiline_content_in_markdown(self):
        quote = "alpha\nbeta `keep`\n<script>no exec</script>"
        with tempfile.TemporaryDirectory() as tmp:
            generator = ReportGenerator(output_dir=tmp)
            paths = generator.generate(
                hypotheses=[
                    {
                        "hypothesis_id": "H1",
                        "indication": "Example Disease",
                        "rank": 1,
                        "overall_score": 0.5,
                        "scores": {},
                        "causal_chain": {
                            "mechanism_axes": [
                                {
                                    "axis_name": "primary",
                                    "steps": [
                                        {
                                            "layer": "L1",
                                            "mechanism": "Step",
                                            "status": "supported",
                                            "verified_spans": [
                                                {
                                                    "evidence_id": "PMID:1",
                                                    "quote": quote,
                                                    "quote_verified": True,
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
                experiments=[],
                query={"target": "Kv1.3"},
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")

        self.assertIn("            alpha", markdown)
        self.assertIn("            beta `keep`", markdown)
        self.assertIn("            <script>no exec</script>", markdown)
        self.assertNotIn("alpha beta", markdown)
        self.assertNotRegex(markdown, r"(?m)^ {0,11}<script>")

    def test_incomplete_or_empty_gap_summary_is_not_rendered(self):
        cases = [
            {},
            {"gap_parent_steps": 3},
            {
                "gap_parent_steps": 3,
                "searched_parent_steps": 2,
            },
        ]
        for summary in cases:
            with self.subTest(summary=summary):
                with tempfile.TemporaryDirectory() as tmp:
                    generator = ReportGenerator(output_dir=tmp)
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
                        query={
                            "target": "Kv1.3",
                            "_verification_summary": "verification complete",
                            "_verification_report": {
                                "gap_retrieval": {"summary": summary},
                            },
                        },
                    )
                    markdown = Path(paths["markdown"]).read_text(
                        encoding="utf-8"
                    )
                    html = Path(paths["html"]).read_text(encoding="utf-8")

                self.assertNotIn("缺口步骤:", markdown)
                self.assertNotIn("缺口证据检索", markdown)
                self.assertNotIn("缺口步骤:", html)
                self.assertNotIn("缺口证据检索", html)
                self.assertIn("verification complete", markdown)


if __name__ == "__main__":
    unittest.main()
