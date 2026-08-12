from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


async def cmd_init(args):
    from dp_indicator.core.orchestrator import Orchestrator
    orch = Orchestrator(api_key=args.api_key)
    query = await orch.parse_intent(args.input)
    if query.get("needs_clarification"):
        print(f"⚠️ {query['clarification_prompt']}")
        return {"status": "needs_clarification", "prompt": query["clarification_prompt"]}
    status = await orch.init(query)
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return status


async def cmd_explore(args):
    from dp_indicator.core.orchestrator import Orchestrator
    orch = Orchestrator(api_key=args.api_key)
    query = await orch.parse_intent(args.input)
    if query.get("needs_clarification"):
        print(f"⚠️ {query['clarification_prompt']}")
        return {"status": "needs_clarification", "prompt": query["clarification_prompt"]}
    if getattr(args, 'focus', None):
        query["focus_areas"] = [args.focus]
        print(f"🎯 方向引导: {args.focus}")
    print(f"🔍 探索靶点: {query['target']}")
    evidence = await orch.explore(query)
    print(f"✅ 获取 {len(evidence)} 条证据")
    pool_path = Path("data/evidence_pool.json")
    pool_path.parent.mkdir(exist_ok=True)
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False, default=str)
    print(f"💾 证据池已保存: {pool_path}")
    return evidence


async def cmd_hypothesize(args):
    from dp_indicator.core.orchestrator import Orchestrator
    orch = Orchestrator(api_key=args.api_key)
    query = await orch.parse_intent(args.input)
    if query.get("needs_clarification"):
        print(f"⚠️ {query['clarification_prompt']}")
        return
    # Load evidence from stage checkpoint or file
    ckpt = orch._load_stage_checkpoint("explore")
    knowledge_base = []
    if ckpt and ckpt.get("evidence_pool"):
        evidence_pool = ckpt["evidence_pool"]
        knowledge_base = ckpt.get("knowledge_base", [])
        print(f"📂 从 explore checkpoint 加载 {len(evidence_pool)} 条证据、{len(knowledge_base)} 条知识库事实")
    else:
        pool_path = Path("data/evidence_pool.json")
        if not pool_path.exists():
            print("❌ 未找到证据池。请先运行 explore。")
            return
        with open(pool_path, "r", encoding="utf-8") as f:
            evidence_pool = json.load(f)
        print(f"📂 从 data/evidence_pool.json 加载 {len(evidence_pool)} 条证据")

    print(f"🧠 生成假设: {query['target']}")
    hypotheses = await orch.hypothesize(query, evidence_pool, knowledge_base=knowledge_base)
    if not hypotheses:
        print("⚠️ 假设生成失败")
        return
    for h in hypotheses:
        print(f"  {h.get('rank','?')}. {h.get('indication','')} | "
              f"评分={h.get('overall_score',0):.3f} | "
              f"可行性={h.get('feasibility_score',0):.3f}")
    return hypotheses


async def cmd_design(args):
    from dp_indicator.core.orchestrator import Orchestrator
    orch = Orchestrator(api_key=args.api_key)
    # Load from hypothesize checkpoint
    ckpt = orch._load_stage_checkpoint("hypothesize")
    if not ckpt:
        print("❌ 未找到 hypothesize checkpoint。请先运行 hypothesize。")
        return
    hypotheses = ckpt.get("hypotheses", [])
    evidence_pool = ckpt.get("evidence_pool", [])
    query = ckpt.get("query", {})
    print(f"🔬 为 {len(hypotheses)} 个假设设计验证实验...")
    experiments = await orch.design(hypotheses, evidence_pool, query=query)
    print(f"✅ 生成 {len(experiments)} 个实验方案:")
    for exp in experiments:
        print(f"  {exp.get('experiment_id','?')}: {exp.get('title','')} [{exp.get('priority','')}]")
    return experiments


async def cmd_report(args):
    from dp_indicator.core.orchestrator import Orchestrator
    orch = Orchestrator(api_key=args.api_key)
    # Load from design checkpoint (preferred) or hypothesize checkpoint
    ckpt = orch._load_stage_checkpoint("design")
    if ckpt:
        hypotheses = ckpt.get("hypotheses", [])
        experiments = ckpt.get("experiments", [])
        query = ckpt.get("query", {})
    else:
        ckpt = orch._load_stage_checkpoint("hypothesize")
        if not ckpt:
            print("❌ 未找到 checkpoint。请先运行 hypothesize 或 design。")
            return
        hypotheses = ckpt.get("hypotheses", [])
        experiments = []
        query = ckpt.get("query", {})
    print(f"📄 生成报告: {len(hypotheses)} 个假设, {len(experiments)} 个实验")
    paths = orch.generate_report(hypotheses, experiments, query)
    for fmt, path in paths.items():
        print(f"  {fmt}: {path}")
    return paths


async def cmd_run_all(args):
    """一键模式 — explore → hypothesize → design → report, 无暂停。"""
    from dp_indicator.core.orchestrator import Orchestrator

    print("=" * 60)
    print("🚀 DP-Indicator fix10 — 一键模式")
    print(f"📌 方向: {args.input}")
    print("=" * 60)

    orch = Orchestrator(api_key=args.api_key)
    query = await orch.parse_intent(args.input)
    if query.get("needs_clarification"):
        print(f"⚠️ {query['clarification_prompt']}")
        return
    if getattr(args, 'focus', None):
        query["focus_areas"] = [args.focus]
        print(f"🎯 方向引导: {args.focus}")

    try:
        # Phase 1: Explore
        print("\n📍 Phase 1/4: Explore...")
        evidence = await orch.explore(query)
        if not evidence:
            print("⚠️ 探索阶段未获取到证据，终止")
            return
        print(f"  → {len(evidence)} 条有效证据")

        # Phase 2: Hypothesize
        print("\n📍 Phase 2/4: Hypothesize...")
        hypotheses = await orch.hypothesize(query, evidence)
        if not hypotheses:
            print("⚠️ 假设生成失败，终止")
            return
        for h in hypotheses:
            print(f"  {h.get('rank','?')}. {h.get('indication','')} | "
                  f"评分={h.get('overall_score',0):.3f} | "
                  f"可行性={h.get('feasibility_score',0):.3f}")

        # Phase 3: Design
        print("\n📍 Phase 3/4: Design experiments...")
        experiments = await orch.design(hypotheses, evidence, query=query)
        print(f"  → {len(experiments)} 个实验方案")

        # Phase 4: Report
        print("\n📍 Phase 4/4: Report...")
        paths = orch.generate_report(hypotheses, experiments, query)
        for fmt, path in paths.items():
            print(f"  {fmt}: {path}")

        print("\n✅ 全流程完成")

    except Exception as e:
        print(f"\n❌ 运行错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    # Status output uses emoji/unicode markers throughout the pipeline. On Windows
    # the console's default codepage (e.g. GBK/cp936) can't encode them and would
    # crash the run with UnicodeEncodeError on the first print(). Force UTF-8.
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="DP-Indicator fix10 — 靶点新适应症探索"
    )
    parser.add_argument("--api-key", default=None, help="Bohrium API Key")
    subparsers = parser.add_subparsers(dest="command")

    p_init = subparsers.add_parser("init", help="初始化检查")
    p_init.add_argument("input", help="探索方向(自由文本)")
    p_init.set_defaults(func=cmd_init)

    p_explore = subparsers.add_parser("explore", help="探索 + 检索")
    p_explore.add_argument("input", help="探索方向(自由文本)")
    p_explore.add_argument("--focus", default=None, help="方向引导(如'autoimmune disease')")
    p_explore.set_defaults(func=cmd_explore)

    p_hyp = subparsers.add_parser("hypothesize", help="从证据生成假设")
    p_hyp.add_argument("input", help="探索方向(自由文本)")
    p_hyp.set_defaults(func=cmd_hypothesize)

    p_design = subparsers.add_parser("design", help="为假设设计验证实验")
    p_design.set_defaults(func=cmd_design)

    p_report = subparsers.add_parser("report", help="生成报告")
    p_report.set_defaults(func=cmd_report)

    p_all = subparsers.add_parser("run-all", help="一键模式(explore→hypothesize→design→report)")
    p_all.add_argument("input", help="探索方向(自由文本)")
    p_all.add_argument("--focus", default=None, help="方向引导")
    p_all.set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    if not args.api_key:
        args.api_key = os.environ.get("BH_API_KEY", "")
    try:
        asyncio.run(args.func(args))
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 致命错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        from dp_indicator.clients.databases import shutdown
        shutdown()


if __name__ == "__main__":
    main()
