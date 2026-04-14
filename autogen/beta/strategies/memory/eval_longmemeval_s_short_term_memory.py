"""
Evaluate LongShortTermMemoryStorage **L1 short-term memory only** on LongMemEval-S.

The haystack is replayed as a single continuous stream: each user/assistant message
is ``save_event``-ed in order, then ``get_history`` runs immediately so
``_maybe_compress_to_l1`` can fire whenever raw context exceeds *token_threshold*
(not only once at the end). We **do not** call
``consolidate()`` or ``storage.session()``, so L2 long-term memory and core memory
are never written — matching a short-term–only setting.

Prerequisites
-------------
- ``pip install -e ./ag2`` (or add ``./ag2`` to ``PYTHONPATH``).
- ``OPENAI_API_KEY`` for the answering model and the judge (same key).

Every run is scored with a LongMemEval-style LLM judge using model **gpt-5-mini**
(fixed; not configurable).  Samples run in parallel via ``asyncio`` (``--workers`` = max concurrent tasks)
on **one** event loop, which avoids Windows errors from ``asyncio.run()`` per
thread (ProactorEventLoop / httpx teardown).

Example
-------
python ag2/autogen/beta/strategies/memory/eval_longmemeval_s_short_term_memory.py \\
  --data LongMemEval/data/longmemeval_s_cleaned.json \\
  --out hyp_lstm_st.jsonl \\
  --token_threshold 8000 \\
  --recent_turns 4 \\
  --workers 8 \\
  --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace

# Resolve repo root and ag2 / autogen.beta without requiring a prior pip install
def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "ag2").is_dir() and (parent / "ag2" / "autogen").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root (expected ag2/autogen).")


_REPO_ROOT = _find_repo_root()
_AG2_SRC = _REPO_ROOT / "ag2"
if _AG2_SRC.is_dir() and str(_AG2_SRC) not in sys.path:
    sys.path.insert(0, str(_AG2_SRC))

import time
from statistics import mean
from openai import OpenAI

from autogen.beta.agent import Agent
from autogen.beta.config.openai.config import OpenAIConfig
from autogen.beta.events import ModelMessage, ModelRequest, ModelResponse
from autogen.beta.history import LongShortTermMemoryStorage
from autogen.beta.stream import MemoryStream

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it

logger = logging.getLogger(__name__)

# Fixed judge model (LongMemEval-style yes/no auto-eval); not overridden by CLI.
JUDGE_MODEL = "gpt-5-mini"

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant answering questions about a long \
chat history with a user. The history appears in your context as raw recent turns plus \
short-term (L1) memory blocks listed under \"Short-Term Memory\". Each L1 block is only \
a summary; for exact facts, quotes, names, or numbers that might be missing from the \
summary, call memory_lookup(block_id) with the block id shown in brackets before you \
answer. Then answer concisely and directly."""


def get_anscheck_prompt(
    task: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool,
) -> str:
    """Judge prompt aligned with LongMemEval ``evaluate_qa.py``."""
    if not abstention:
        if task in ("single-session-user", "single-session-assistant", "multi-session"):
            return (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. If the response only "
                "contains a subset of the information required by the answer, answer no. \n\n"
                f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        if task == "temporal-reasoning":
            return (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate "
                "steps to get the correct answer, you should also answer yes. If the response only "
                "contains a subset of the information required by the answer, answer no. In addition, "
                "do not penalize off-by-one errors for the number of days. If the question asks for the "
                "number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting "
                "19 days when the answer is 18), the model's response is still correct. \n\n"
                f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        if task == "knowledge-update":
            return (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, the "
                "response should be considered as correct as long as the updated answer is the required answer.\n\n"
                f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        if task == "single-session-preference":
            return (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. The response is correct as long as "
                "it recalls and utilizes the user's personal information correctly.\n\n"
                f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        # Fallback for rare / new benchmark labels
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, answer no.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    return (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. The model could say that "
        "the information is incomplete, or some other information is given but the asked information is not.\n\n"
        f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    )


def judge_completion(client: OpenAI, model: str, prompt: str, *, max_tries: int = 5) -> str:
    last_exc: BaseException | None = None
    for attempt in range(max_tries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                n=1,
                temperature=1.0,
                max_completion_tokens=100,
            )
            return (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            last_exc = exc
            time.sleep(min(2**attempt, 30))
    assert last_exc is not None
    raise last_exc


def clean_turn(turn: dict) -> dict:
    t = dict(turn)
    t.pop("has_answer", None)
    return t


def load_longmemeval_s_examples(raw: object) -> list[dict]:
    """Load LongMemEval-S from JSON: either a list of examples or a column-oriented dict.

    ``longmemeval_s_cleaned.json`` is a JSON array. Some exports (e.g. HuggingFace
    ``datasets``) store the same rows as one object per field, each field mapping
    row index strings to values — slicing that dict raises ``KeyError: slice(...)``.
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        qid = raw.get("question_id")
        if isinstance(qid, dict) and qid and all(isinstance(k, str) and k.isdigit() for k in qid):
            indices = sorted(qid.keys(), key=int)
            cols = list(raw.keys())
            rows: list[dict] = []
            for idx in indices:
                row: dict = {}
                for col in cols:
                    col_val = raw[col]
                    if isinstance(col_val, dict):
                        row[col] = col_val[idx]
                    else:
                        row[col] = col_val
                rows.append(row)
            return rows
    raise TypeError(
        f"Unsupported LongMemEval-S JSON shape: {type(raw).__name__!r}. "
        "Expected a JSON array of examples or a column-oriented object with string row indices."
    )


def make_tracked_memory_lookup(
    storage: LongShortTermMemoryStorage,
    retrieval_log: list[dict[str, str]],
) -> Callable[[str], Awaitable[str]]:
    """Wrap ``storage.memory_lookup`` and append ``{id, abstract}`` for each L1 hit."""

    async def memory_lookup_tracked(block_id: str) -> str:
        result = await storage.memory_lookup(block_id)
        all_l1 = [
            b for blocks in storage._l1_blocks.values() for b in blocks
        ] + storage._persisted_l1_blocks
        l1 = next((b for b in all_l1 if b.id == block_id), None)
        if l1:
            retrieval_log.append({"id": l1.id, "abstract": l1.abstract})
        return result

    return memory_lookup_tracked


async def inject_haystack(
    storage: LongShortTermMemoryStorage,
    stream: MemoryStream,
    *,
    haystack_dates: list[str],
    haystack_sessions: list[list[dict]],
    prefix_session_dates: bool,
) -> None:
    """Append haystack messages one at a time, calling ``get_history`` after each event.

    ``LongShortTermMemoryStorage`` only runs ``_maybe_compress_to_l1`` inside
    ``get_history``. Invoking it after every ``save_event`` matches a live
    chat where the context window is rebuilt on each turn, so L1 compression
    can trigger repeatedly whenever raw text crosses *token_threshold*.
    """
    ctx = SimpleNamespace(stream=stream)
    sid = stream.id
    for session_date, session in zip(haystack_dates, haystack_sessions, strict=True):
        for turn in session:
            turn = clean_turn(turn)
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                if prefix_session_dates:
                    content = f"[Session date: {session_date}]\n{content}"
                await storage.save_event(ModelRequest.ensure_request([content]), ctx)
                await storage.get_history(sid)
            elif role == "assistant":
                await storage.save_event(
                    ModelResponse(message=ModelMessage(content=content)),
                    ctx,
                )
                await storage.get_history(sid)
            else:
                logger.warning("Skipping turn with unknown role %r", role)


def build_question_user_message(question_date: str, question: str) -> str:
    return (
        "I will give you several history chats between you and a user. "
        "Please answer the question based on the relevant chat history "
        "(including information recoverable via memory_lookup from L1 blocks).\n\n"
        f"Current Date: {question_date}\n"
        f"Question: {question}\n"
        "Answer:"
    )


async def run_one_example(
    entry: dict,
    *,
    config: OpenAIConfig,
    token_threshold: int,
    recent_turns: int,
    prefix_session_dates: bool,
) -> dict:
    """Inject haystack, ask the benchmark question, return a result row (no judge)."""
    storage = LongShortTermMemoryStorage(
        config=config,
        l2_store_path=None,
        token_threshold=token_threshold,
        recent_turns=recent_turns,
    )
    stream = MemoryStream(storage=storage)
    l1_retrieved: list[dict[str, str]] = []
    agent = Agent(
        "assistant",
        prompt=ANSWER_SYSTEM_PROMPT,
        config=config,
        tools=[make_tracked_memory_lookup(storage, l1_retrieved)],
    )

    await inject_haystack(
        storage,
        stream,
        haystack_dates=entry["haystack_dates"],
        haystack_sessions=entry["haystack_sessions"],
        prefix_session_dates=prefix_session_dates,
    )

    user_msg = build_question_user_message(entry["question_date"], entry["question"])
    reply = await agent.ask(user_msg, stream=stream)
    hypothesis = (await reply.content()) or ""

    l1_count = len(storage._l1_blocks.get(stream.id, []))
    raw_events = len(storage._raw_events.get(stream.id, []))

    # Drop stream state; storage instance is discarded with this row
    await storage.drop_history(stream.id)

    return {
        "question_id": entry["question_id"],
        "question_type": entry["question_type"],
        "hypothesis": hypothesis,
        "l1_retrieved": list(l1_retrieved),
        "meta": {
            "l1_blocks_after_inject": l1_count,
            "raw_events_before_question": raw_events,
            "token_threshold": token_threshold,
            "recent_turns": recent_turns,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data",
        type=Path,
        default=_REPO_ROOT / "LongMemEval" / "data" / "longmemeval_s_cleaned.json",
        help="Path to longmemeval_s_cleaned.json",
    )
    p.add_argument("--out", type=Path, default="./longmemeval_s_result_LSTMS.jsonl", help="Output JSONL path (one object per line)")
    p.add_argument("--model", type=str, default="gpt-5-mini", help="OpenAI model id for the agent")
    p.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API key")
    p.add_argument(
        "--question_types",
        type=str,
        default="",
        help="Comma-separated filter (e.g. single-session-user,multi-session). Empty = every example in the file.",
    )
    p.add_argument("--limit", type=int, default=None, help="Max number of examples to run")
    p.add_argument("--offset", type=int, default=0, help="Skip first N examples after filtering")
    p.add_argument("--token_threshold", type=int, default=10000, help="L1 compression token budget")
    p.add_argument("--recent_turns", type=int, default=3, help="Raw user/assistant rounds kept before compressing older turns")
    p.add_argument("--no_session_date_prefix", action="store_true", help="Do not prepend session dates to user turns")
    p.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Max concurrent samples (asyncio semaphore; single shared event loop).",
    )
    p.add_argument("--log_level", type=str, default="WARNING")
    return p.parse_args()


async def _process_sample_async(
    sample_index: int,
    entry: dict,
    *,
    config: OpenAIConfig,
    judge_client: OpenAI,
    token_threshold: int,
    recent_turns: int,
    prefix_session_dates: bool,
) -> tuple[int, dict, str, int]:
    """Run one example (async agent + blocking judge in a worker thread).

    Returns ``(sample_index, row, question_type, score)`` for ordered aggregation.
    """
    qid = entry["question_id"]
    qtype = entry["question_type"]

    try:
        row = await run_one_example(
            entry,
            config=config,
            token_threshold=token_threshold,
            recent_turns=recent_turns,
            prefix_session_dates=prefix_session_dates,
        )
    except Exception as exc:
        logger.exception("Failed on %s", qid)
        row = {
            "question_id": qid,
            "question_type": qtype,
            "hypothesis": "",
            "l1_retrieved": [],
            "error": str(exc),
        }

    row["question"] = entry["question"]
    row["reference_answer"] = entry["answer"]
    row["agent_answer"] = row.get("hypothesis", "")

    abstention = "_abs" in qid
    try:
        prompt = get_anscheck_prompt(
            qtype,
            entry["question"],
            entry["answer"],
            row.get("hypothesis", ""),
            abstention=abstention,
        )
        eval_response = await asyncio.to_thread(judge_completion, judge_client, JUDGE_MODEL, prompt)
        label = "yes" in eval_response.lower()
    except Exception as exc:
        logger.warning("Judge failed for %s: %s", qid, exc)
        label = False
        eval_response = f"(judge error: {exc})"

    row["autoeval_label"] = {
        "model": JUDGE_MODEL,
        "label": label,
        "raw": eval_response,
    }
    score = 1 if label else 0
    return sample_index, row, qtype, score


async def _async_run_evaluation(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY or --api_key is required for the answering model.")

    with args.data.open(encoding="utf-8") as f:
        dataset = load_longmemeval_s_examples(json.load(f))

    type_filter = {t.strip() for t in args.question_types.split(",") if t.strip()}
    if type_filter:
        dataset = [e for e in dataset if e.get("question_type") in type_filter]

    dataset = dataset[args.offset :]
    if args.limit is not None:
        dataset = dataset[: args.limit]

    config = OpenAIConfig(model=args.model, api_key=args.api_key)
    judge_client = OpenAI(api_key=args.api_key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    qtype_scores: dict[str, list[int]] = defaultdict(list)
    all_scores: list[int] = []

    n = len(dataset)
    conc = max(1, args.workers)
    sem = asyncio.Semaphore(conc)

    async def _bounded(i: int, entry: dict) -> tuple[int, dict, str, int]:
        async with sem:
            return await _process_sample_async(
                i,
                entry,
                config=config,
                judge_client=judge_client,
                token_threshold=args.token_threshold,
                recent_turns=args.recent_turns,
                prefix_session_dates=not args.no_session_date_prefix,
            )

    if n == 0:
        ordered: list[tuple[int, dict, str, int]] = []
    else:
        pbar = tqdm(total=n, desc=f"LongMemEval-S (async, max {conc})")

        async def _tracked(i: int, entry: dict) -> tuple[int, dict, str, int]:
            try:
                return await _bounded(i, entry)
            finally:
                pbar.update(1)

        ordered = await asyncio.gather(*[_tracked(i, e) for i, e in enumerate(dataset)])
        pbar.close()

    ordered = sorted(ordered, key=lambda t: t[0])

    with args.out.open("w", encoding="utf-8") as out_f:
        for _idx, row, qtype, score in ordered:
            all_scores.append(score)
            qtype_scores[qtype].append(score)
            print(json.dumps(row, ensure_ascii=False), file=out_f, flush=True)

    print()
    print("=" * 72)
    print("Accuracy (judge:", JUDGE_MODEL + ")")
    print("=" * 72)
    if all_scores:
        n_total = len(all_scores)
        correct_total = sum(all_scores)
        print(f"Overall:  {correct_total}/{n_total}  ({100.0 * correct_total / n_total:.2f}%)")
        print()
        print("By question_type:")
        for qt in sorted(qtype_scores.keys()):
            vals = qtype_scores[qt]
            n = len(vals)
            c = sum(vals)
            acc = mean(vals)
            print(f"  {qt}:  {c}/{n}  ({100.0 * acc:.2f}%)")
    else:
        print("No examples were judged.")
    print("=" * 72)
    print("Wrote", args.out)


def run_evaluation() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    asyncio.run(_async_run_evaluation(args))


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
