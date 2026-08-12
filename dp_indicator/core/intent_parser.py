from __future__ import annotations
import re
import json
import asyncio

_STOPWORDS = {
    "THE", "AND", "NOT", "CAN", "FOR", "BUT", "OR", "ARE", "WAS", "WERE",
    "HAS", "HAD", "HAVE", "DOES", "DID", "WILL", "SHALL", "MAY", "MIGHT",
    "MUST", "SHOULD", "COULD", "WOULD", "THIS", "THAT", "THESE", "THOSE",
    "WITH", "FROM", "INTO", "UPON", "OVER", "UNDER", "THAN", "THEN", "THEM",
    "THEY", "THEIR", "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "WHOM", "WHY",
    "HOW", "ALL", "ANY", "BOTH", "EACH", "FEW", "MORE", "MOST", "OTHER",
    "SOME", "SUCH", "NO", "NOR", "ONLY", "OWN", "SAME", "SO", "TOO",
    "VERY", "JUST", "NOW", "ALSO", "BE", "IS", "AM", "BEEN", "BEING",
    "GET", "GETS", "GOT", "GETTING", "USE", "USED", "USES", "USING",
    "DO", "DON", "DOESN", "DIDN", "WASN", "WEREN", "HASN", "HADN",
}


def _empty_query() -> dict:
    return {
        "raw_input": "",
        "target": "",
        "synonyms": [],
        "focus_areas": [],
        "exclude_areas": [],
        "prior_evidence": None,
        "exploration_mode": "target_to_indication",
        "direction": "target_to_indication",
        "needs_clarification": False,
        "clarification_prompt": "",
    }


def _regex_parse(user_input: str) -> dict:
    """Regex-based fallback parser."""
    query = _empty_query()
    query["raw_input"] = user_input
    text = user_input.strip()

    matches = re.findall(r'(?a:\b)([A-Za-z][A-Za-z0-9]*\.?\d*(?:-[A-Za-z0-9]+)?)(?a:\b)', text)
    matches = [m for m in matches if m]
    gene_candidates = []
    for m in matches:
        if any(c.isdigit() for c in m):
            gene_candidates.append(m)
        elif '-' in m:
            gene_candidates.append(m)
        elif m.isupper() and len(m) >= 2 and m not in _STOPWORDS:
            gene_candidates.append(m)
    gene_candidates.sort(
        key=lambda x: (0 if (any(c.isdigit() for c in x) or '-' in x) else 1, x))
    if gene_candidates:
        query["target"] = gene_candidates[0]
        query["synonyms"] = gene_candidates[1:]
    else:
        query["needs_clarification"] = True
        query["clarification_prompt"] = (
            "未检测到明确的靶点/基因/蛋白名称。"
            "请提供具体的靶点标识符，以便系统进行适应症探索。")

    exclude_patterns = [
        r'排除[：:]\s*([^\n，,]+)',
        r'exclude[：:]\s*([^\n，,]+)',
        r'不涉及[：:]\s*([^\n，,]+)',
    ]
    for pat in exclude_patterns:
        for m in re.findall(pat, text, re.IGNORECASE):
            cleaned = m.strip()
            if cleaned and len(cleaned) > 1 and cleaned not in ("法", "掉", "除", "不"):
                query["exclude_areas"].append(cleaned)
    for m in re.findall(r'排除(?!法|掉|除)([^\n，,\s]{2,})', text):
        query["exclude_areas"].append(m.strip())

    focus_patterns = [
        r'关注[：:]\s*([^\n，,]+)',
        r'focus[：:]\s*([^\n，,]+)',
        r'重点[：:]\s*([^\n，,]+)',
        r'主要[：:]\s*([^\n，,]+)',
    ]
    for pat in focus_patterns:
        for m in re.findall(pat, text, re.IGNORECASE):
            cleaned = m.strip()
            if cleaned and len(cleaned) > 1:
                query["focus_areas"].append(cleaned)
    for m in re.findall(r'关注(?!领域|方向|问题)([^\n，,\s]{2,})', text):
        query["focus_areas"].append(m.strip())

    if "已有" in text or "prior" in text.lower() or "实验" in text:
        query["prior_evidence"] = text
    return query


_LLM_PROMPT = """You are a biomedical entity recognition system. Extract target exploration info from the user's natural language input.

Rules:
- "target" is the primary molecule/gene/protein name (e.g. Kv1.3, PD-L1, BTK, TNF-alpha)
- "synonyms" are alternative names or closely related paralogs mentioned
- "focus_areas" are disease areas the user wants to emphasize (e.g. "autoimmune", "oncology")
- "exclude_areas" are disease areas to skip
- "direction" is "target_to_indication" (find new diseases for a known target) or "indication_to_target" (find new targets for a known disease)
- If you cannot identify a clear target, set "needs_clarification" to true

Return ONLY valid JSON:
{{
  "target": "string or empty",
  "synonyms": ["list"],
  "focus_areas": ["list"],
  "exclude_areas": ["list"],
  "direction": "target_to_indication",
  "needs_clarification": false
}}

User input: "{user_input}" """


async def parse_query_with_llm(user_input: str, llm) -> dict:
    """LLM-based intent parser. Falls back to regex on failure."""
    query = _empty_query()
    query["raw_input"] = user_input

    try:
        result, _ = await asyncio.wait_for(
            llm.structured([
                {"role": "system", "content": "You are a biomedical entity recognition system. Return ONLY valid JSON."},
                {"role": "user", "content": _LLM_PROMPT.format(user_input=user_input)},
            ], max_tokens=512, task="retriever"),
            timeout=30,
        )

        if isinstance(result, dict) and not result.get("error"):
            target = str(result.get("target", "")).strip()
            if target:
                query["target"] = target
                query["synonyms"] = result.get("synonyms", [])
                query["focus_areas"] = result.get("focus_areas", [])
                query["exclude_areas"] = result.get("exclude_areas", [])
                query["direction"] = result.get("direction", "target_to_indication")
                query["exploration_mode"] = query["direction"]
                return query
            elif result.get("needs_clarification"):
                query["needs_clarification"] = True
                query["clarification_prompt"] = (
                    "无法从输入中识别明确的靶点/基因/蛋白名称。"
                    "请提供具体的靶点标识符。"
                )
                return query
    except Exception as e:
        print(f"  [intent_parser] LLM parse failed: {e}, falling back to regex", flush=True)

    # Fallback to regex
    return _regex_parse(user_input)


def parse_query(user_input: str) -> dict:
    """Synchronous regex-based parser. Used when no LLM is available."""
    return _regex_parse(user_input)
