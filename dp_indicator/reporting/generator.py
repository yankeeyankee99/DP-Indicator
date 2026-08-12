"""报告生成 — JSON + Markdown + HTML"""
from __future__ import annotations
import html
import json
import re
import time
from pathlib import Path

_GAP_SUMMARY_KEYS = (
    "gap_parent_steps",
    "searched_parent_steps",
    "resolved_parent_steps",
    "unresolved_atomic_claims",
)


_MARKDOWN_INLINE_ESCAPE_CHARS = (
    "`", "*", "_", "{", "}", "[", "]", "<", ">", "(", ")", "#",
    "+", "-", ".", "!", "|",
)


class ReportGenerator:
    """生成三种格式报告"""
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    @staticmethod
    def _safe_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)

    @staticmethod
    def _html_escape(value) -> str:
        return html.escape(ReportGenerator._safe_text(value), quote=True)

    @staticmethod
    def _safe_markdown_inline(value) -> str:
        text = ReportGenerator._safe_text(value)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\n", " ")
        text = html.escape(text, quote=False)
        text = text.replace("\\", "\\\\")
        for char in _MARKDOWN_INLINE_ESCAPE_CHARS:
            text = text.replace(char, f"\\{char}")
        return text

    @classmethod
    def _markdown_indented_block_lines(
        cls,
        text: object,
        indent: str = "        ",
    ) -> list[str]:
        normalized = cls._safe_text(text).replace("\r\n", "\n").replace("\r", "\n")
        return [f"{indent}{line}" for line in normalized.split("\n")]

    @classmethod
    def _markdown_verification_summary_lines(cls, summary: object) -> list[str]:
        return cls._markdown_indented_block_lines(summary, indent="    ")

    @staticmethod
    def _is_non_negative_int(value) -> bool:
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        )

    @staticmethod
    def _gap_summary_complete(summary: dict) -> bool:
        if not isinstance(summary, dict):
            return False
        return all(
            ReportGenerator._is_non_negative_int(summary.get(key))
            for key in _GAP_SUMMARY_KEYS
        )

    @staticmethod
    def _safe_mermaid_label(value) -> str:
        text = ReportGenerator._safe_text(value)
        text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        text = re.sub(r"<[^>]*>", "", text)
        text = text.replace("\\", "\\\\")
        text = text.replace('"', "'")
        text = text.replace("[", "(").replace("]", ")")
        text = text.replace("{", "(").replace("}", ")")
        text = text.replace("|", "/")
        text = text.replace("#", "")
        text = text.replace("<", "").replace(">", "")
        text = text.replace("&", "and")
        return text[:60]

    @staticmethod
    def _safe_mermaid_node_id(value, fallback: str = "node") -> str:
        text = re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            ReportGenerator._safe_text(value),
        )
        if not text:
            text = fallback
        if text[0].isdigit():
            text = f"n_{text}"
        return text

    @staticmethod
    def _safe_html_status(value) -> str:
        status = ReportGenerator._safe_text(value)
        if status in {"supported", "inferred", "hypothesized"}:
            return status
        return "inferred"

    @staticmethod
    def _html_anchor_id(value) -> str:
        text = re.sub(
            r"[^A-Za-z0-9_-]",
            "_",
            ReportGenerator._safe_text(value).replace(" ", "_"),
        )
        return text or "hypothesis"

    @staticmethod
    def _is_bridge_evidence(item: dict) -> bool:
        return item.get("evidence_role") == "bridge_evidence"

    @staticmethod
    def _bridge_evidence_markdown() -> str:
        return "（补充桥接证据）"

    @staticmethod
    def _bridge_evidence_html() -> str:
        return '<span class="bridge-evidence">补充桥接证据</span>'

    @classmethod
    def _format_gap_retrieval_summary(cls, summary: dict) -> str:
        if not cls._gap_summary_complete(summary):
            return ""
        formatted = (
            f"缺口步骤: {summary['gap_parent_steps']}；"
            f"已补证: {summary['resolved_parent_steps']}；"
            f"已检索: {summary['searched_parent_steps']}；"
            f"未解决原子Claim: {summary['unresolved_atomic_claims']}"
        )
        acceptance_keys = (
            "uncited_atomic_before",
            "supported_or_partial_after",
            "coverage_gain",
            "acceptance_threshold",
            "acceptance_passed",
        )
        if all(key in summary for key in acceptance_keys):
            try:
                gain = float(summary["coverage_gain"])
                threshold = float(summary["acceptance_threshold"])
            except (TypeError, ValueError):
                return formatted
            formatted += (
                f"\nuncited_atomic_before: {summary['uncited_atomic_before']}"
                f"\nsupported_or_partial_after: "
                f"{summary['supported_or_partial_after']}"
                f"\ncoverage_gain: {gain:.1%}"
                f"\nacceptance_threshold: {threshold:.1%}"
                f"\nacceptance_passed: {bool(summary['acceptance_passed'])}"
            )
            if not summary["acceptance_passed"]:
                formatted += "\n实证覆盖不足"
        return formatted

    @classmethod
    def _source_ref_str_markdown(cls, src: dict) -> str:
        ref_parts = []
        first_author = cls._safe_markdown_inline(src.get("first_author", ""))
        if first_author:
            ref_parts.append(f"{first_author} et al.")
        journal = cls._safe_markdown_inline(
            src.get("journal_short", src.get("journal", ""))
        )
        if journal:
            ref_parts.append(f"*{journal}*")
        year = cls._safe_markdown_inline(src.get("year", ""))
        if year:
            ref_parts.append(f"({year})")
        return ", ".join(ref_parts)

    @classmethod
    def _source_ref_str_html(cls, src: dict) -> str:
        ref_parts = []
        first_author = src.get("first_author", "")
        if first_author:
            ref_parts.append(f"{cls._html_escape(first_author)} et al.")
        journal = src.get("journal_short", src.get("journal", ""))
        if journal:
            ref_parts.append(f"<i>{cls._html_escape(journal)}</i>")
        year = src.get("year", "")
        if year:
            ref_parts.append(f"({cls._html_escape(year)})")
        return ", ".join(ref_parts)

    @classmethod
    def _source_id_parts(cls, src: dict) -> tuple[str, str, str]:
        return (
            cls._safe_text(src.get("pmid", "")),
            cls._safe_text(src.get("doi", "")),
            cls._safe_text(src.get("evidence_id", "")),
        )

    @classmethod
    def _source_id_tag_markdown(cls, src: dict) -> str:
        pmid, doi, evidence_id = cls._source_id_parts(src)
        if pmid:
            return f"[PMID:{cls._safe_markdown_inline(pmid)}]"
        if doi:
            return f"[DOI:{cls._safe_markdown_inline(doi)}]"
        if evidence_id:
            return f"[{cls._safe_markdown_inline(evidence_id)}]"
        return ""

    @classmethod
    def _source_id_tag_html(cls, src: dict) -> str:
        pmid, doi, evidence_id = cls._source_id_parts(src)
        if pmid:
            return f"[PMID:{cls._html_escape(pmid)}]"
        if doi:
            return f"[DOI:{cls._html_escape(doi)}]"
        if evidence_id:
            return f"[{cls._html_escape(evidence_id)}]"
        return ""

    @classmethod
    def _render_markdown_source_lines(cls, src: dict) -> list[str]:
        ref_str = cls._source_ref_str_markdown(src)
        id_tag = cls._source_id_tag_markdown(src)
        bridge_label = (
            f" {cls._bridge_evidence_markdown()}"
            if cls._is_bridge_evidence(src)
            else ""
        )
        lines = []
        if ref_str or id_tag:
            if ref_str:
                lines.append(
                    f"      - **来源**: {ref_str} {id_tag}{bridge_label}".rstrip()
                )
            else:
                lines.append(
                    f"      - **来源**: {id_tag}{bridge_label}".rstrip()
                )
        record_url = cls._safe_markdown_inline(src.get("record_url", ""))
        if src.get("key_finding"):
            lines.append(
                "        - **关键发现**: "
                f"{cls._safe_markdown_inline(src.get('key_finding'))[:150]}"
            )
        if record_url and not ref_str and not id_tag:
            lines.append(f"        - **链接**: {record_url}")
        return lines

    @classmethod
    def _render_markdown_verified_span_lines(cls, span: dict) -> list[str]:
        evidence_id = cls._safe_markdown_inline(span.get("evidence_id", ""))
        bridge_label = (
            f" {cls._bridge_evidence_markdown()}"
            if cls._is_bridge_evidence(span)
            else ""
        )
        lines = [f"      - **核验原文** [{evidence_id}]{bridge_label}:"]
        lines.extend(
            cls._markdown_indented_block_lines(
                span.get("quote", ""),
                indent="            ",
            )
        )
        return lines

    @classmethod
    def _render_html_verified_span_block(cls, span: dict) -> str:
        evidence_id = cls._html_escape(span.get("evidence_id", ""))
        quote = cls._html_escape(span.get("quote", ""))
        bridge_label = (
            f" {cls._bridge_evidence_html()}"
            if cls._is_bridge_evidence(span)
            else ""
        )
        return (
            '<div class="evidence-trace">📝 核验原文 '
            f'[{evidence_id}]:<pre style="margin:4px 0 8px 12px;'
            'white-space:pre-wrap;">'
            f'{quote}</pre>{bridge_label}</div>\n'
        )

    @classmethod
    def _render_html_source_block(cls, src: dict) -> str:
        ref_str = cls._source_ref_str_html(src)
        id_tag = cls._source_id_tag_html(src)
        bridge_label = (
            f" {cls._bridge_evidence_html()}"
            if cls._is_bridge_evidence(src)
            else ""
        )
        block = ""
        if ref_str or id_tag:
            if ref_str:
                block += (
                    f'<div class="evidence-trace">📎 '
                    f'{ref_str} {id_tag}{bridge_label}</div>\n'
                )
            else:
                block += (
                    f'<div class="evidence-trace">📎 '
                    f'{id_tag}{bridge_label}</div>\n'
                )
        if src.get("key_finding"):
            block += (
                f'<div class="evidence-trace">📝 '
                f'{cls._html_escape(cls._safe_text(src.get("key_finding"))[:150])}'
                f'</div>\n'
            )
        return block

    @classmethod
    def _claim_status_summary_markdown(cls, hypothesis: dict) -> str:
        claims = (
            hypothesis.get("verification", {}).get("claim_grounding", [])
        )
        counts = {}
        for claim in claims:
            verdict = str(claim.get("verdict", "unverifiable"))
            counts[verdict] = counts.get(verdict, 0) + 1
        return ", ".join(
            f"{cls._safe_markdown_inline(verdict)}={count}"
            for verdict, count in sorted(counts.items())
        )

    @staticmethod
    def _claim_status_summary(hypothesis: dict) -> str:
        claims = (
            hypothesis.get("verification", {}).get("claim_grounding", [])
        )
        counts = {}
        for claim in claims:
            verdict = str(claim.get("verdict", "unverifiable"))
            counts[verdict] = counts.get(verdict, 0) + 1
        return ", ".join(
            f"{verdict}={count}"
            for verdict, count in sorted(counts.items())
        )

    def generate(self, hypotheses: list[dict], experiments: list[dict],
                 query: dict) -> dict:
        """生成所有格式"""
        # Extract report-only metadata before persisting the user query.
        pipeline_stats = query.pop("_pipeline_stats", None)
        verification_report = query.pop("_verification_report", None)
        verification_summary = query.pop("_verification_summary", None)
        verification_report_raw = query.pop(
            "_verification_report_raw",
            None,
        )
        verification_adjustments = query.pop(
            "_verification_adjustments",
            None,
        )
        verification_sanitization = query.pop(
            "_verification_sanitization",
            None,
        )
        report = {
            "query": query,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_hypotheses": len(hypotheses),
            "n_experiments": len(experiments),
            "hypotheses": hypotheses,
            "experiments": experiments,
        }
        if pipeline_stats:
            report["pipeline_stats"] = pipeline_stats
        if verification_report is not None:
            report["verification_report"] = verification_report
        if verification_summary:
            report["verification_summary"] = verification_summary
        if verification_report_raw is not None:
            report["verification_report_raw"] = verification_report_raw
        if verification_adjustments is not None:
            report["verification_adjustments"] = verification_adjustments
        if verification_sanitization is not None:
            report["verification_sanitization"] = verification_sanitization
        json_path = self.output_dir / "report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        md_path = self.output_dir / "report.md"
        md_content = self._generate_markdown(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        html_path = self.output_dir / "report.html"
        html_content = self._generate_html(report)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return {"json": str(json_path), "markdown": str(md_path), "html": str(html_path)}
    def _generate_markdown(self, report: dict) -> str:
        query = report["query"]
        lines = [
            "# DP-Indicator 报告",
            "",
            "## 查询信息",
            f"- 靶点: {self._safe_markdown_inline(query.get('target', ''))}",
            (
                "- 方向: "
                f"{self._safe_markdown_inline(query.get('direction', 'target_to_indication'))}"
            ),
            f"- 生成时间: {self._safe_markdown_inline(report['timestamp'])}",
            "",
        ]
        degradation = query.get('_degradation_notes', [])
        if degradation:
            lines.append("## ⚠️ 数据完整性说明")
            lines.append("")
            for note in degradation:
                lines.append(f"- {self._safe_markdown_inline(note)}")
            lines.append("")
        retrieval_log = query.get('_retrieval_log', [])
        if retrieval_log:
            lines.append("## 检索过程")
            lines.append("")
            for entry in retrieval_log:
                tag = "🎯" if "focus" in entry.get("type", "") else "🔍"
                db = self._safe_markdown_inline(entry.get('db', ''))
                term = self._safe_markdown_inline(entry.get('term', ''))
                lines.append(
                    f"{tag} [{db}] `{term}` → {entry.get('n_results', 0)} 条"
                )
            lines.append("")
        lines.append("## 假设排名(按综合评分)")
        lines.append("")
        for hyp in report.get("hypotheses", []):
            score = hyp.get("overall_score", 0)
            ind = self._safe_markdown_inline(
                hyp.get("indication", hyp.get("indication_name", "Unknown"))
            )
            statement = self._safe_markdown_inline(
                hyp.get("statement", hyp.get("one_sentence_statement", ""))
            )
            rank = self._safe_markdown_inline(hyp.get('rank', '?'))
            lines.append(f"### {rank}. {ind}")
            lines.append(f"- **综合评分**: {score}")
            lines.append(
                f"- **可行性/成药性**: {hyp.get('feasibility_score', 0):.3f}"
            )
            lines.append(f"- **假设**: {statement}")
            lines.append(
                "- **可证伪预测**: "
                f"{self._safe_markdown_inline(hyp.get('falsifiable_prediction', ''))}"
            )
            supp = hyp.get("supporting_evidence", [])
            if supp:
                lines.append(f"- **关联证据** ({len(supp)} 条):")
                for ev in supp[:5]:
                    status_icon = {
                        "RCT_human": "🏥", "clinical_trial": "🏥", "animal": "🐭",
                        "in_vitro": "🔬", "literature": "📄",
                        "database_association": "🗃️",
                        "expert_curation": "🗃️", "review": "📚",
                    }.get(ev.get("evidence_type", ""), "📄")
                    source_db = self._safe_markdown_inline(ev.get('source_db', ''))
                    title = self._safe_markdown_inline(ev.get('title', ''))[:100]
                    lines.append(
                        f"  - {status_icon} [{source_db}] "
                        f"grade={ev.get('grade_score',0)} | {title}"
                    )
                if len(supp) > 5:
                    lines.append(f"  - ... 及 {len(supp)-5} 条其他证据")
            scores = hyp.get("scores", {})
            if scores:
                lines.append("- **维度评分**:")
                for k, label in [
                    ("G1", "联结合理性"), ("G2", "先验支持度"),
                    ("G3", "可证伪性"), ("G4", "可行性/成药性"),
                ]:
                    lines.append(
                        f"  - {k} {label}: "
                        f"{self._safe_markdown_inline(scores.get(k, 0))}"
                    )
            chain = hyp.get("causal_chain", {})
            if isinstance(chain, dict) and "mechanism_axes" in chain:
                lines.append("- **因果链**:")
                for i, axis in enumerate(chain.get("mechanism_axes", [])):
                    axis_name = self._safe_markdown_inline(
                        axis.get("axis_name", f"机制路径 {i+1}")
                    )
                    lines.append(f"  - **{axis_name}**:")
                    for step in axis.get("steps", []):
                        status = self._safe_markdown_inline(
                            step.get("status", "inferred")
                        )
                        status_icon = {
                            "supported": "✅", "inferred": "➡️",
                            "hypothesized": "❓",
                        }.get(step.get("status", "inferred"), "➡️")
                        layer = self._safe_markdown_inline(step.get("layer", "?"))
                        mechanism = self._safe_markdown_inline(
                            step.get("mechanism", "")
                        )
                        lines.append(
                            f"    - {status_icon} {layer}: {mechanism} [{status}]"
                        )
                        verified_spans = [
                            span
                            for span in step.get("verified_spans", [])
                            if span.get("quote_verified") is True
                        ]
                        sources = step.get("sources", [])
                        if sources:
                            for src in sources:
                                lines.extend(self._render_markdown_source_lines(src))
                        for span in verified_spans:
                            lines.extend(
                                self._render_markdown_verified_span_lines(span)
                            )
                        if not verified_spans and step.get("source_text"):
                            source_text = self._safe_markdown_inline(
                                step.get('source_text')
                            )[:120]
                            lines.append(
                                "      - **未核验证据说明**: "
                                f"\"{source_text}...\""
                            )
                cross_talk = chain.get("cross_talk", [])
                if cross_talk:
                    lines.append("  - **交叉作用**:")
                    for ct in cross_talk:
                        desc = self._safe_markdown_inline(ct.get('description', ''))
                        lines.append(f"    - {desc}")
            elif isinstance(chain, list):
                lines.append("- **因果链**:")
                for item in chain:
                    if isinstance(item, dict):
                        layer = self._safe_markdown_inline(
                            item.get("layer", item.get("level", "?"))
                        )
                        mechanism = self._safe_markdown_inline(
                            item.get("mechanism", item.get("description", ""))
                        )
                        status = self._safe_markdown_inline(
                            item.get("status", "inferred")
                        )
                        status_icon = {
                            "supported": "✅", "inferred": "➡️",
                            "hypothesized": "❓",
                        }.get(item.get("status", "inferred"), "➡️")
                        lines.append(
                            f"  - {status_icon} {layer}: {mechanism} [{status}]"
                        )
            elif isinstance(chain, dict):
                lines.append("- **因果链**:")
                for level in ["L1", "L2", "L3", "L4", "L5"]:
                    if level in chain:
                        entry = chain[level]
                        status = self._safe_markdown_inline(
                            entry.get("status", "inferred")
                        )
                        status_icon = {
                            "supported": "✅", "inferred": "➡️",
                            "hypothesized": "❓",
                        }.get(entry.get("status", "inferred"), "➡️")
                        desc = self._safe_markdown_inline(
                            entry.get("description", entry.get("mechanism", ""))
                        )
                        lines.append(
                            f"  - {status_icon} {level}: {desc} [{status}]"
                        )

            eh_mapping = hyp.get("evidence_mapping", {})
            if eh_mapping and not eh_mapping.get("_source"):
                pos = eh_mapping.get("positive_evidence", [])
                neg = eh_mapping.get("contradicting_evidence", [])
                if pos or neg:
                    lines.append("- **证据映射**:")
                    if pos:
                        lines.append(f"  - 直接支持: {len(pos)} 条")
                    if neg:
                        lines.append(f"  - 矛盾证据: {len(neg)} 条")

            claim_summary = self._claim_status_summary_markdown(hyp)
            if claim_summary:
                lines.append(f"- **Claim核验**: {claim_summary}")

            critic = hyp.get("critic_review", {})
            if critic and not critic.get("_source"):
                fatal = critic.get("fatal_weakness", {})
                if fatal and (fatal.get("reasoning") or fatal.get("weakness")):
                    fatal_text = fatal.get("reasoning") or fatal.get("weakness")
                    lines.append(
                        "- **评审弱点**: "
                        f"{self._safe_markdown_inline(fatal_text)[:200]}"
                    )
                fix = critic.get("suggested_fix", "")
                if isinstance(fix, dict):
                    fix_text = fix.get("reasoning") or fix.get("conclusion") or ""
                else:
                    fix_text = str(fix) if fix else ""
                if fix_text:
                    lines.append(
                        "- **改进建议**: "
                        f"{self._safe_markdown_inline(fix_text)[:200]}"
                    )

            if hyp.get("scoring_method") == "failed":
                err = hyp.get('scores', {}).get('_error', 'Unknown error')
                lines.append(
                    f"- ⚠️ **评分失败**: {self._safe_markdown_inline(err)}"
                )

            lines.append("")
        if report.get("experiments"):
            lines.append("## 实验提案")
            lines.append("")
            for exp in report["experiments"]:
                exp_id = self._safe_markdown_inline(exp.get('experiment_id', ''))
                title = self._safe_markdown_inline(exp.get('title', ''))
                lines.append(f"### {exp_id}: {title}")
                lines.append(
                    f"- 模型: {self._safe_markdown_inline(exp.get('model', ''))}"
                )
                lines.append(
                    "- 干预: "
                    f"{self._safe_markdown_inline(exp.get('intervention', ''))}"
                )
                lines.append(
                    f"- 读数: {self._safe_markdown_inline(exp.get('readout', ''))}"
                )
                lines.append(
                    f"- 对照: {self._safe_markdown_inline(exp.get('control', ''))}"
                )
                lines.append(
                    "- 优先级: "
                    f"{self._safe_markdown_inline(exp.get('priority', ''))}"
                )
                lines.append(
                    "- 支持预期: "
                    f"{self._safe_markdown_inline(exp.get('expected_supporting', ''))}"
                )
                lines.append(
                    "- 反驳预期: "
                    f"{self._safe_markdown_inline(exp.get('expected_refuting', ''))}"
                )
                lines.append(
                    f"- 时间线: {self._safe_markdown_inline(exp.get('timeline', ''))}"
                )
                lines.append(
                    f"- 理由: {self._safe_markdown_inline(exp.get('rationale', ''))}"
                )
                lines.append("")

        first_hyp = report.get("hypotheses", [{}])[0]
        reflection = first_hyp.get("meta_reflection", {})
        if reflection and not reflection.get("_source"):
            lines.append("## 🧠 元认知反思")
            lines.append("")
            unc = reflection.get("uncertainty_ranking", {})
            if unc:
                lines.append(
                    "**最不确定假设**: "
                    f"{self._safe_markdown_inline(unc.get('hypothesis', 'N/A'))}"
                )
                lines.append(
                    f"- 原因: {self._safe_markdown_inline(unc.get('reason', ''))}"
                )
                lines.append("")
            gap = reflection.get("evidence_gap", {})
            if gap:
                lines.append(
                    "**最需补充证据**: "
                    f"{self._safe_markdown_inline(gap.get('what', 'N/A'))}"
                )
                lines.append(
                    f"- 理由: {self._safe_markdown_inline(gap.get('why', ''))}"
                )
                lines.append("")
            inter = reflection.get("interconnections", "")
            if inter:
                lines.append(
                    f"**假设间关联**: {self._safe_markdown_inline(inter)}"
                )
                lines.append("")
            biases = reflection.get("potential_biases", [])
            if biases:
                lines.append("**潜在偏见**:")
                for b in biases:
                    lines.append(f"- {self._safe_markdown_inline(b)}")
                lines.append("")
            ci = reflection.get("counter_intuitive", {})
            if ci:
                lines.append(
                    "**反直觉发现**: "
                    f"{self._safe_markdown_inline(ci.get('finding', ''))}"
                )
                lines.append(
                    f"- 评估: {self._safe_markdown_inline(ci.get('assessment', ''))}"
                )
                lines.append("")
            next_step = reflection.get("recommended_next_step", "")
            if next_step:
                lines.append(
                    "**推荐下一步**: "
                    f"{self._safe_markdown_inline(next_step)}"
                )
                lines.append("")
            conf = reflection.get("overall_confidence")
            if conf is not None:
                lines.append(
                    f"**整体置信度**: {self._safe_markdown_inline(conf)}"
                )
                lines.append("")

        pipeline_stats = report.get("pipeline_stats")
        if pipeline_stats:
            lines.append("## 流水线统计")
            lines.append("")
            stat_keys = [
                ("total_raw_results", "原始检索结果"),
                ("total_queries", "检索查询次数"),
                ("after_dedup", "去重后证据数"),
                ("after_prefilter", "预过滤后证据数"),
                ("after_grading", "分级后有效证据"),
            ]
            for key, label in stat_keys:
                value = pipeline_stats.get(key, 'N/A')
                lines.append(
                    f"- {label}: {self._safe_markdown_inline(value)}"
                )
            lines.append("")

        verification_summary = report.get("verification_summary")
        v_report = report.get("verification_report", {})
        gap_summary_data = (v_report.get("gap_retrieval") or {}).get("summary")
        gap_summary = (
            self._format_gap_retrieval_summary(gap_summary_data)
            if isinstance(gap_summary_data, dict)
            else ""
        )
        if verification_summary or gap_summary:
            lines.append("## 证据验证报告")
            lines.append("")
            if verification_summary:
                lines.append("")
                lines.extend(
                    self._markdown_verification_summary_lines(
                        verification_summary
                    )
                )
                lines.append("")
            if gap_summary:
                lines.append(f"- **缺口证据检索**: {gap_summary}")
                lines.append("")
            for hyp in report.get("hypotheses", []):
                verification = hyp.get("verification", {})
                if not verification:
                    continue
                ind = self._safe_markdown_inline(hyp.get("indication", ""))
                all_issues = []
                all_issues.extend(verification.get("v2_citation_issues", []))
                all_issues.extend(verification.get("v3_description_issues", []))
                all_issues.extend(verification.get("v4_relevance_issues", []))
                if all_issues:
                    lines.append(f"### {ind} - 验证问题")
                    for issue in all_issues:
                        sev_icon = {
                            "high": "🔴", "medium": "🟡", "low": "🔵",
                            "warning": "⚠️",
                        }.get(issue.get("severity", ""), "⚠️")
                        eid = self._safe_markdown_inline(issue.get("evidence_id", ""))
                        issue_text = self._safe_markdown_inline(issue.get('issue', ''))
                        lines.append(f"- {sev_icon} [{eid}] {issue_text}")
                    lines.append("")
                adj = verification.get("score_adjustment", 0)
                if adj:
                    lines.append(f"### {ind} - 评分调整: {adj:+.3f}")
                    lines.append("")

        lines.append("---")
        lines.append("*报告由 DP-Indicator fix10 生成*")
        return "\n".join(lines)
    def _generate_html(self, report: dict) -> str:
        """自包含 HTML 报告(含 Mermaid 图 + 跳转锚点)"""
        hypotheses = report.get("hypotheses", [])
        # Generate one Mermaid diagram for each hypothesis.
        mermaid_diagrams = {}
        for hyp in hypotheses:
            ind = hyp.get("indication", hyp.get("indication_name", ""))
            hyp_id = self._html_anchor_id(hyp.get("hypothesis_id", ind))
            chain = hyp.get("causal_chain", {})
            links = []
            if isinstance(chain, dict) and "mechanism_axes" in chain:
                for axis in chain.get("mechanism_axes", []):
                    steps = axis.get("steps", [])
                    for i, step in enumerate(steps):
                        layer = step.get("layer", f"L{i}")
                        links.append({
                            "from_node": layer,
                            "to_node": f"{layer}_out",
                            "relationship": step.get("mechanism", ""),
                            "status": step.get("status", "inferred"),
                        })
            elif isinstance(chain, list):
                links = chain
            elif isinstance(chain, dict) and "links" in chain:
                links = chain["links"]
            elif isinstance(chain, dict):
                for level in ["L1", "L2", "L3", "L4", "L5"]:
                    if level in chain:
                        entry = chain[level]
                        links.append({
                            "from_node": level,
                            "to_node": "",
                            "relationship": entry.get("description", ""),
                            "status": entry.get("status", "inferred"),
                        })
            mermaid = ["graph TD"]
            for i, link in enumerate(links):
                if isinstance(link, dict):
                    from_node = self._safe_mermaid_node_id(
                        link.get("from_node", f"L{i}"),
                        f"L{i}",
                    )
                    to_node = (
                        self._safe_mermaid_node_id(
                            link.get("to_node", f"{from_node}_out"),
                            f"{from_node}_out",
                        )
                        if link.get("to_node")
                        else f"{from_node}_out"
                    )
                    rel = self._safe_mermaid_label(
                        link.get("relationship", "→")
                    )
                    status = self._safe_html_status(link.get("status", "inferred"))
                    style_map = {
                        "supported": "#27ae60",
                        "inferred": "#f39c12",
                        "hypothesized": "#e74c3c",
                    }
                    color = style_map.get(status, "#999")
                    mermaid.append(f'    {from_node} -->|"{rel}"| {to_node}')
                    mermaid.append(
                        f"    classDef s{i} fill:{color},color:#fff,stroke:#333"
                    )
                    mermaid.append(f"    class {to_node} s{i}")
            mermaid_diagrams[hyp_id] = (
                "\n".join(mermaid) if len(mermaid) > 1 else ""
            )

        degradation_html = ""
        degradation = report['query'].get('_degradation_notes', [])
        if degradation:
            degradation_html = (
                '<div style="background:#fff3cd;padding:12px;border-radius:4px;'
                'margin:16px 0;border-left:4px solid #ffc107;">'
                '<h3>⚠️ 数据完整性说明</h3><ul>'
                + ''.join(
                    f'<li>{self._html_escape(n)}</li>' for n in degradation
                )
                + '</ul></div>'
            )

        html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>DP-Indicator Report</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'strict' }});
</script>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;padding:20px}}
h1{{color:#1a5276}}
.hypothesis{{background:#f8f9fa;padding:16px;margin:16px 0;border-radius:8px;border-left:4px solid #3498db}}
.score{{font-weight:bold;color:#2ecc71}}
.chain{{background:#fff;padding:10px;margin:8px 0;border-radius:4px}}
.chain span{{margin-right:15px}}
.status-supported{{color:#27ae60}}
.status-inferred{{color:#f39c12}}
.status-hypothesized{{color:#e74c3c}}
.bridge-evidence{{color:#8e44ad;font-weight:bold;margin-left:6px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:8px}}
th{{background:#1a5276;color:#fff}}
.evidence-trace{{font-size:0.9em;color:#666}}
.mermaid-container{{background:#fff;padding:12px;margin:8px 0;border-radius:4px;border:1px solid #eee;overflow-x:auto}}
navigation{{position:sticky;top:0;background:#fff;padding:8px 0;z-index:100;border-bottom:1px solid #eee}}
navigation a{{margin-right:12px;color:#1a5276;text-decoration:none;font-size:0.9em}}
navigation a:hover{{text-decoration:underline}}
.nav-section{{font-weight:bold;margin:12px 0 4px;color:#1a5276}}
</style>
</head>
<body>
<div class="navigation">
<a href="#summary">📊 摘要</a> | <a href="#table">📋 排名表</a> | <a href="#details">🔗 因果链详情</a> | <a href="#mermaids">📈 Mermaid 图</a>
</div>
<h1 id="summary">DP-Indicator 报告</h1>
<p><strong>版本:</strong> DP-Indicator fix10</p>
<p>靶点: <b>{self._html_escape(report['query'].get('target',''))}</b> | 时间: {self._html_escape(report['timestamp'])}</p>
{degradation_html}
<h2 id="table">假设排名</h2>
<table>
<tr><th>排名</th><th>适应症</th><th>综合评分</th><th>可行性</th><th>假设</th></tr>
"""
        for hyp in hypotheses:
            ind = hyp.get('indication', hyp.get('indication_name', ''))
            stmt = hyp.get('statement', hyp.get('one_sentence_statement', ''))
            hyp_id_raw = self._html_anchor_id(hyp.get('hypothesis_id', ind))
            html += f"""<tr>
<td>{self._html_escape(hyp.get('rank','?'))}</td>
<td><b><a href="#hyp-{hyp_id_raw}">{self._html_escape(ind)}</a></b></td>
<td class="score">{hyp.get('overall_score',0):.3f}</td>
<td>{hyp.get('feasibility_score',0):.3f}</td>
<td>{self._html_escape(stmt)[:80]}</td>
</tr>\n"""
        html += "</table>\n"
        # Mermaid overview.
        html += '<h2 id="mermaids">📈 因果链概览 (Mermaid)</h2>\n'
        for hyp in hypotheses:
            ind = hyp.get("indication", hyp.get("indication_name", ""))
            hyp_id = self._html_anchor_id(hyp.get("hypothesis_id", ind))
            diagram = mermaid_diagrams.get(hyp_id, "")
            if diagram:
                html += f'<div class="mermaid-container">\n<h4><a href="#hyp-{hyp_id}">{self._html_escape(ind)}</a></h4>\n'
                html += f'<pre class="mermaid">\n{diagram}\n</pre>\n</div>\n'
        html += '<h2 id="details">因果链详情与证据追溯</h2>\n'
        for hyp in hypotheses:
            ind = hyp.get("indication", hyp.get("indication_name", ""))
            hyp_id = self._html_anchor_id(hyp.get("hypothesis_id", ind))
            chain = hyp.get("causal_chain", {})
            html += f'<div class="hypothesis" id="hyp-{hyp_id}">\n<h3><a href="#summary">↑</a> {self._html_escape(ind)}</h3>\n'
            html += '<div class="chain">\n'

            if isinstance(chain, dict) and "mechanism_axes" in chain:
                # Render the structured mechanism-axis format.
                for i, axis in enumerate(chain.get("mechanism_axes", [])):
                    axis_name = axis.get("axis_name", f"机制路径 {i+1}")
                    html += f'<div style="margin:8px 0;font-weight:bold;color:#1a5276;">{self._html_escape(axis_name)}</div>\n'
                    for step in axis.get("steps", []):
                        status = self._safe_html_status(step.get("status", "inferred"))
                        status_class = f"status-{status}"
                        status_label = {
                            "supported": "✅ 已支持",
                            "inferred": "➡️ 推断",
                            "hypothesized": "❓ 待验证",
                        }.get(status, self._html_escape(status))
                        layer = self._html_escape(step.get("layer", "?"))
                        mechanism = self._html_escape(step.get("mechanism", ""))
                        html += f'<div><span class="{status_class}">{status_label}</span> {layer}: {mechanism}</div>\n'
                        verified_spans = [
                            span
                            for span in step.get("verified_spans", [])
                            if span.get("quote_verified") is True
                        ]
                        sources = step.get("sources", [])
                        if sources:
                            for src in sources:
                                html += self._render_html_source_block(src)
                        for span in verified_spans:
                            html += self._render_html_verified_span_block(span)
                        if not verified_spans and step.get("source_text"):
                            html += (
                                '<div class="evidence-trace">📝 '
                                '未核验证据说明: "'
                                f'{self._html_escape(self._safe_text(step.get("source_text"))[:120])}'
                                '..."</div>\n'
                            )
                cross_talk = chain.get("cross_talk", [])
                if cross_talk:
                    html += '<div style="margin:4px 0;font-style:italic;color:#666;">交叉作用:</div>\n'
                    for ct in cross_talk:
                        html += (
                            '<div class="evidence-trace">↔️ '
                            f'{self._html_escape(ct.get("description", ""))}</div>\n'
                        )
            else:
                # Render the other supported causal-chain shapes.
                if isinstance(chain, list):
                    links = chain
                elif isinstance(chain, dict) and "links" in chain:
                    links = chain["links"]
                elif isinstance(chain, dict):
                    links = []
                    for level in ["L1", "L2", "L3", "L4", "L5"]:
                        if level in chain:
                            entry = chain[level]
                            links.append({
                                "level": level,
                                "description": entry.get("description", ""),
                                "status": entry.get("status", "inferred"),
                                "evidence_ids": entry.get("evidence_ids", []),
                                "source_text": entry.get("source_text", ""),
                            })
                else:
                    links = []
                for link in links:
                    if isinstance(link, str):
                        html += f'<div>{self._html_escape(link)}</div>\n'
                        continue
                    status = self._safe_html_status(link.get("status", "inferred"))
                    status_class = f"status-{status}"
                    status_label = {
                        "supported": "✅ 已支持",
                        "inferred": "➡️ 推断",
                        "hypothesized": "❓ 待验证",
                    }.get(status, self._html_escape(status))
                    html += f'<div><span class="{status_class}">{status_label}</span> '
                    html += (
                        f'L{self._html_escape(link.get("level","?"))}: '
                        f'{self._html_escape(link.get("from_node",""))} → '
                        f'{self._html_escape(link.get("to_node",""))} '
                        f'({self._html_escape(link.get("relationship",""))})</div>\n'
                    )
                    eids = link.get("evidence_ids", [])
                    if eids:
                        html += (
                            '<div class="evidence-trace">📎 证据: '
                            f'{", ".join(self._html_escape(eid) for eid in eids)}'
                            '</div>\n'
                        )
                    src = link.get("source_text", "")
                    if src:
                        html += (
                            '<div class="evidence-trace">📝 '
                            '未核验证据说明: "'
                            f'{self._html_escape(self._safe_text(src)[:120])}'
                            '..."</div>\n'
                        )
            claim_summary = self._claim_status_summary(hyp)
            if claim_summary:
                html += (
                    '<div class="evidence-trace"><b>Claim核验:</b> '
                    f'{self._html_escape(claim_summary)}</div>\n'
                )
            html += '</div></div>\n'
        verification_summary = report.get("verification_summary")
        v_report = report.get("verification_report", {})
        gap_summary_data = (v_report.get("gap_retrieval") or {}).get("summary")
        gap_summary = (
            self._format_gap_retrieval_summary(gap_summary_data)
            if isinstance(gap_summary_data, dict)
            else ""
        )
        if verification_summary or gap_summary:
            html += '<h2 id="verification">证据验证报告</h2>\n'
            if verification_summary:
                html += (
                    '<pre style="background:#f4f4f4;padding:12px;'
                    'border-radius:4px;">'
                    f'{self._html_escape(verification_summary)}</pre>\n'
                )
            if gap_summary:
                html += (
                    '<div class="evidence-trace"><b>缺口证据检索:</b> '
                    f'{self._html_escape(gap_summary)}</div>\n'
                )
        html += "</body></html>"
        return html
