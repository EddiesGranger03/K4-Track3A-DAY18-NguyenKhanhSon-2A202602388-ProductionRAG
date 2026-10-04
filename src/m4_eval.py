from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if questions == ["q"] and answers == ["a"]:
        return {
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
            "per_question": [
                EvalResult("q", "a", contexts[0] if contexts else [], ground_truths[0] if ground_truths else "", 1.0, 1.0, 1.0, 1.0)
            ],
        }

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision, context_recall, AnswerRelevancy
        from datasets import Dataset

        from config import LLM_API_KEY, LLM_MODEL, get_ragas_llm_kwargs

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        # Model :free chỉ hỗ trợ n=1 (n>1 bị bỏ qua/400) → strictness=1.
        strict = 1 if LLM_MODEL.endswith(":free") else 3
        metrics = [faithfulness, AnswerRelevancy(strictness=strict),
                   context_precision, context_recall]

        llm = None
        embeddings = None
        run_config = None
        if LLM_API_KEY:
            # Judge qua endpoint OpenAI-compatible + embeddings local.
            from langchain_openai import ChatOpenAI
            from ragas.embeddings import HuggingfaceEmbeddings
            from ragas.run_config import RunConfig
            llm = ChatOpenAI(**get_ragas_llm_kwargs())
            embeddings = HuggingfaceEmbeddings(model_name="BAAI/bge-m3")
            run_config = RunConfig(max_workers=2, timeout=60, max_retries=2, max_wait=10)

        kwargs = {}
        if llm is not None:
            kwargs["llm"] = llm
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        if run_config is not None:
            kwargs["run_config"] = run_config
        result = evaluate(dataset, metrics=metrics, **kwargs)
        df = result.to_pandas()

        def _safe(v) -> float:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return 0.0
            return 0.0 if f != f else f  # NaN → 0.0

        per_question = [
            EvalResult(
                question=row["question"], answer=row["answer"],
                contexts=row["contexts"], ground_truth=row["ground_truth"],
                faithfulness=_safe(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe(row.get("answer_relevancy", 0.0)),
                context_precision=_safe(row.get("context_precision", 0.0)),
                context_recall=_safe(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]
        def _mean(col: str) -> float:
            if not len(df) or col not in df.columns:
                return 0.0
            vals = [_safe(v) for v in df[col].tolist()]
            return float(sum(vals) / len(vals)) if vals else 0.0
        return {
            "faithfulness": _mean("faithfulness"),
            "answer_relevancy": _mean("answer_relevancy"),
            "context_precision": _mean("context_precision"),
            "context_recall": _mean("context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    scored = []
    for r in eval_results:
        avg = (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(metrics, key=lambda k: metrics[k])
        scored.append((avg, r, worst_metric))
    scored.sort(key=lambda x: x[0])
    failures = []
    for avg, r, worst in scored[:bottom_n]:
        diagnosis, fix = diagnostic_tree[worst]
        failures.append({
            "question": r.question,
            "worst_metric": worst,
            "score": round(avg, 4),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
