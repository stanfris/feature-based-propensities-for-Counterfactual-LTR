import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentReport:
    dataset: str
    model: str
    n_sessions_used: int
    cutoff: int
    position_bias_source: str
    position_bias_csv_path: str | None
    position_bias_csv_method: str | None
    alpha_clip: float | None
    propensity_method: str | None
    propensity_checkpoint_dir: str | None
    results: dict[str, Any]


def build_report(
    *,
    dataset: str,
    model: str,
    n_sessions_used: int,
    cutoff: int,
    position_bias_source: str,
    position_bias_csv_path: str | None,
    position_bias_csv_method: str | None,
    alpha_clip: float | None,
    propensity_method: str | None,
    propensity_checkpoint_dir: str | None,
    metrics: dict[str, Any],
    click_prediction: dict[str, Any] | None = None,
) -> ExperimentReport:
    results = {"metrics": metrics}
    if click_prediction is not None:
        results["click_prediction"] = click_prediction
    return ExperimentReport(
        dataset=dataset,
        model=model,
        n_sessions_used=n_sessions_used,
        cutoff=cutoff,
        position_bias_source=position_bias_source,
        position_bias_csv_path=position_bias_csv_path,
        position_bias_csv_method=position_bias_csv_method,
        alpha_clip=alpha_clip,
        propensity_method=propensity_method,
        propensity_checkpoint_dir=propensity_checkpoint_dir,
        results=results,
    )


def report_to_dict(report: ExperimentReport) -> dict[str, Any]:
    return {
        "dataset": report.dataset,
        "model": report.model,
        "n_sessions_used": int(report.n_sessions_used),
        "cutoff": int(report.cutoff),
        "position_bias_source": report.position_bias_source,
        "position_bias_csv_path": report.position_bias_csv_path,
        "position_bias_csv_method": report.position_bias_csv_method,
        "alpha_clip": None if report.alpha_clip is None else float(report.alpha_clip),
        "propensity_method": report.propensity_method,
        "propensity_checkpoint_dir": report.propensity_checkpoint_dir,
        "results": report.results,
    }


def write_results(report: ExperimentReport, output_path: Path) -> None:
    output = report_to_dict(report)
    print(f"Writing results to {output_path}")
    with output_path.open("w") as f:
        json.dump(output, f)


__all__ = [
    "ExperimentReport",
    "build_report",
    "report_to_dict",
    "write_results",
]
