from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel

class ModelUsageSummary(BaseModel):
    model: str
    completion_tokens: int
    cost: float
    prompt_tokens: int
    total_tokens: int

class ActualUsageSummary(BaseModel):
    usages: Optional[list[ModelUsageSummary]] = None
    total_cost: Optional[float] = None

class TotalUsageSummary(BaseModel):
    usages: Optional[list[ModelUsageSummary]] = None
    total_cost: Optional[float] = None


Mode = Literal["both", "total", "actual"]

class UsageSummary(BaseModel):
    actual: ActualUsageSummary
    total: TotalUsageSummary
    mode: Mode

    def _print_usage(self, usage_summary: Union[ActualUsageSummary, TotalUsageSummary], usage_type: str = "total", f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print
        word_from_type = "including" if usage_type == "total" else "excluding"
        if usage_summary.usages is None or len(usage_summary.usages) == 0:
            f("No actual cost incurred (all completions are using cache).", flush=True)
            return

        f(f"Usage summary {word_from_type} cached usage: ", flush=True)
        f(f"Total cost: {round(usage_summary.total_cost, 5)}", flush=True)

        for usage in usage_summary.usages:
            f(
                f"* Model '{usage.model}': cost: {round(usage.cost, 5)}, prompt_tokens: {usage.prompt_tokens}, completion_tokens: {usage.completion_tokens}, total_tokens: {usage.total_tokens}",
                flush=True,
            )

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        if self.total.usages is None:
            f('No usage summary. Please call "create" first.', flush=True)
            return

        f("-" * 100, flush=True)
        if self.mode == "both":
            self._print_usage(self.actual, "actual", f)
            f()
            if self.total.model_dump_json() != self.actual.model_dump_json():
                self._print_usage(self.total, "total", f)
            else:
                f(
                    "All completions are non-cached: the total cost with cached completions is the same as actual cost.",
                    flush=True,
                )
        elif self.mode == "total":
            self._print_usage(self.total, "total", f)
        elif self.mode == "actual":
            self._print_usage(self.actual, "actual", f)
        else:
            raise ValueError(f'Invalid mode: {self.mode}, choose from "actual", "total", ["actual", "total"]')
        f("-" * 100, flush=True)

    

def _change_usage_summary_format(
        actual_usage_summary: Optional[dict[str, Any]] = None, 
        total_usage_summary: Optional[dict[str, Any]] = None
    ) -> dict[str, dict[str, Any]]:
    summary = {}
    
    for usage_type, usage_summary in {"actual": actual_usage_summary, "total": total_usage_summary}.items():
        if usage_summary is None:
            summary[usage_type] = {"usages": None, "total_cost": None}
            continue
        
        usage_summary_altered_format = {"usages": []}
        for k, v in usage_summary.items():
            if isinstance(k, str) and isinstance(v, dict):
                current_usage = {key: value for key, value in v.items()}
                current_usage["model"] = k
                usage_summary_altered_format["usages"].append(current_usage)
            else:
                usage_summary_altered_format[k] = v
        summary[usage_type] = usage_summary_altered_format

    return summary
    

def create_usage_summary_model(
        actual_usage_summary: Optional[dict[str, Any]] = None, 
        total_usage_summary: Optional[dict[str, Any]] = None,
        mode: Mode = "both"
    ) -> UsageSummary:
    # print(f"{actual_usage_summary=}")
    # print(f"{total_usage_summary=}")

    summary_dict = _change_usage_summary_format(actual_usage_summary, total_usage_summary)

    usage_summary = UsageSummary(**summary_dict, mode=mode)

    return usage_summary
