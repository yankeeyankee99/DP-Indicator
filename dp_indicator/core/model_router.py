"""
Model Router — 任务-模型映射 + 并发控制

每个任务类型固定绑定一个当前可用模型，以保持行为一致；每个模型设置
并发上限，以减少 429 和超时。qwen-max 用于快速、稳定的结构化输出，
glm-5.1 用于复杂推理和报告生成，deepseek-v3.2 用于高吞吐的检索、
长文本提取、批判审查和全证据池综合。
"""
from __future__ import annotations
import asyncio
from typing import Optional

# =============================================================================
# 任务 → 模型映射(固定分工)
# 管道中的 LLM 任务类型：
#   intent          — 意图解析：从自然语言提取靶点、同义词、方向等
#   retriever       — 检索辅助：从已有证据中提取扩展关键词、关联疾病筛选
#   grader          — 证据分级：GRADE 框架评估证据质量
#   reasoner        — 推理生成：贝叶斯后验概率计算、假设生成、合理性论证
#   feasibility     — 可行性评估：实验设计可行性、风险识别
#   critic          — 批判性评审：对抗性审查，找出假设漏洞
#   report          — 报告生成：最终探索报告组稿
#   fulltext        — 全文摘要提取：快速处理长文本
#   prioritizer     — 证据优先级：需要细致判断
#   verifier        — 证据核验：质量优先的推理任务
#   synthesizer     — 全证据池综合：高吞吐逐项提取
# =============================================================================

TASK_MODEL_MAP: dict[str, str] = {
    "intent":        "bh:qwen-max",
    "retriever":     "bh:deepseek-v3.2",
    "grader":        "bh:qwen-max",
    "reasoner":      "bh:glm-5.1",
    "feasibility":   "bh:glm-5.1",
    "critic":        "bh:deepseek-v3.2",
    "report":        "bh:glm-5.1",
    "fulltext":      "bh:deepseek-v3.2",
    "prioritizer":   "bh:glm-5.1",
    "verifier":      "bh:glm-5.1",
    "synthesizer":   "bh:deepseek-v3.2",
}

# 并发上限：每个模型同时最多 N 个请求
# 基于实测：glm-5.1 响应慢但质量高，deepseek-v3.2 和 qwen-max 快
MODEL_CONCURRENCY_LIMITS: dict[str, int] = {
    "glm-5.1": 3,       # 慢但质量高，允许3并发
    "deepseek-v3.2": 4, # 快，高吞吐
    "qwen-max": 4,      # 快，高吞吐
}

# =============================================================================
# ModelRouter 类
# =============================================================================

class ModelRouter:
    """任务到模型的调度器，带并发控制。"""

    def __init__(self, task_model_map: dict = None,
                 concurrency_limits: dict = None,
                 api_key: str = None):
        self._task_model_map = task_model_map or TASK_MODEL_MAP
        self._concurrency_limits = concurrency_limits or MODEL_CONCURRENCY_LIMITS
        self._api_key = api_key

        # 每个模型的信号量(延迟初始化，用到才建)
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        # 统计
        self._call_counts: dict[str, int] = {}
        self._tokens: dict[str, dict] = {}

    def get_model_for_task(self, task: str) -> str:
        """根据任务类型返回对应的模型标识符(带 bh: 前缀)。"""
        model = self._task_model_map.get(task, self._task_model_map["reasoner"])
        return model if model.startswith("bh:") else f"bh:{model}"

    def get_model_name(self, task: str) -> str:
        """返回不带前缀的模型名称。"""
        model = self.get_model_for_task(task)
        return model.removeprefix("bh:")

    def get_semaphore(self, task: str) -> asyncio.Semaphore:
        """获取任务对应模型的并发信号量。"""
        model_name = self.get_model_name(task)
        if model_name not in self._semaphores:
            limit = self._concurrency_limits.get(model_name, 3)
            self._semaphores[model_name] = asyncio.Semaphore(limit)
        return self._semaphores[model_name]

    async def execute_with_limit(self, task: str, coro):
        """在模型并发限制下执行协程。"""
        sem = self.get_semaphore(task)
        model_name = self.get_model_name(task)
        async with sem:
            self._call_counts[model_name] = self._call_counts.get(model_name, 0) + 1
            return await coro

    def record_tokens(self, task: str, usage: dict):
        """记录 token 使用量。"""
        model_name = self.get_model_name(task)
        if model_name not in self._tokens:
            self._tokens[model_name] = {"input": 0, "output": 0}
        self._tokens[model_name]["input"] += usage.get("prompt_tokens", 0)
        self._tokens[model_name]["output"] += usage.get("completion_tokens", 0)

    @property
    def stats(self) -> dict:
        """返回各模型调用统计。"""
        return {
            "calls": dict(self._call_counts),
            "tokens": dict(self._tokens),
            "concurrency_limits": dict(self._concurrency_limits),
        }

    @property
    def task_summary(self) -> dict:
        """返回任务-模型映射表(可读形式)。"""
        return {task: self.get_model_name(task) for task in self._task_model_map}
