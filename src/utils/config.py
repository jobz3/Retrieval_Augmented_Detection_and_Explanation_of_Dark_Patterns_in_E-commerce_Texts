"""
Shared configuration loader for the research scaffold.

This module makes config.yaml the active source of truth for canonical paths
and Phase 1 defaults while staying dependency-light.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if value == "":
        return ""

    if value[0] in {'"', "'"} and value[-1] == value[0]:
        return ast.literal_eval(value)

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    def next_container(start_idx: int, current_indent: int) -> Any:
        for next_idx in range(start_idx + 1, len(lines)):
            candidate = _strip_inline_comment(lines[next_idx]).rstrip()
            if not candidate.strip():
                continue
            indent = len(candidate) - len(candidate.lstrip(" "))
            if indent <= current_indent:
                break
            if candidate.strip().startswith("- "):
                return []
            return {}
        return {}

    for line_idx, raw_line in enumerate(lines):
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unexpected list item at line {line_idx + 1}: {raw_line}")
            parent.append(_parse_scalar(stripped[2:]))
            continue

        key, sep, remainder = stripped.partition(":")
        if sep == "":
            raise ValueError(f"Invalid config line {line_idx + 1}: {raw_line}")

        if not isinstance(parent, dict):
            raise ValueError(f"Unexpected mapping at line {line_idx + 1}: {raw_line}")

        value = remainder.strip()
        if value == "":
            container = next_container(line_idx, indent)
            parent[key] = container
            stack.append((indent, container))
        else:
            parent[key] = _parse_scalar(value)

    return root


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Top-level config structure must be a mapping.")
        return data
    except ImportError:
        return _minimal_yaml_load(text)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class PathsConfig:
    upstream_dataset_repo: Path
    raw_data: Path
    processed_data: Path
    german_data: Path
    outputs: Path
    indices: Path
    models: Path
    results: Path
    baseline_results: Path


@dataclass(frozen=True)
class DataConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    raw_audit_filename: str
    split_manifest_filename: str
    categories: list[str]


@dataclass(frozen=True)
class BaselineModelConfig:
    model_name: str
    max_length: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    warmup_ratio: float


@dataclass(frozen=True)
class BaselinesConfig:
    bert: BaselineModelConfig
    roberta: BaselineModelConfig

    def by_name(self, key: str) -> BaselineModelConfig:
        if key == "bert":
            return self.bert
        if key == "roberta":
            return self.roberta
        raise ValueError(f"Unknown baseline model '{key}'. Choose: bert, roberta")


@dataclass(frozen=True)
class RetrievalConfig:
    k: int
    encoder: str
    phase1_encoders: list[str]
    optional_encoders: list[str]
    sbert_model: str
    bert_model: str
    roberta_model: str
    strategy: str
    diversity_lambda: float


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str
    models: list[str]
    default_model: str
    temperature: float
    max_tokens: int
    format: str


@dataclass(frozen=True)
class EvaluationConfig:
    rubric_dimensions: list[str]
    human_eval_sample_size: int


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    baselines: BaselinesConfig
    retrieval: RetrievalConfig
    ollama: OllamaConfig
    evaluation: EvaluationConfig


def _build_baseline_model(data: dict[str, Any]) -> BaselineModelConfig:
    return BaselineModelConfig(
        model_name=str(data["model_name"]),
        max_length=int(data["max_length"]),
        batch_size=int(data["batch_size"]),
        learning_rate=float(data["learning_rate"]),
        num_epochs=int(data["num_epochs"]),
        warmup_ratio=float(data["warmup_ratio"]),
    )


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total}"
        )


def _validate_phase1_retrieval_encoders(retrieval: RetrievalConfig) -> None:
    expected = {"bert", "roberta"}
    actual = set(retrieval.phase1_encoders)
    if actual != expected:
        raise ValueError(
            f"Phase 1 retrieval encoders must be exactly {sorted(expected)}, got {sorted(actual)}"
        )


@lru_cache(maxsize=1)
def load_config(config_path: Path = CONFIG_PATH) -> AppConfig:
    data = _read_yaml(config_path)

    project_data = data["project"]
    path_data = data["paths"]
    data_section = data["data"]
    baselines_section = data["baselines"]
    retrieval_data = data["retrieval"]
    ollama_data = data["ollama"]
    evaluation_data = data["evaluation"]

    project = ProjectConfig(
        name=str(project_data["name"]),
        seed=int(project_data["seed"]),
    )

    paths = PathsConfig(
        upstream_dataset_repo=_resolve_path(str(path_data["upstream_dataset_repo"])),
        raw_data=_resolve_path(str(path_data["raw_data"])),
        processed_data=_resolve_path(str(path_data["processed_data"])),
        german_data=_resolve_path(str(path_data["german_data"])),
        outputs=_resolve_path(str(path_data["outputs"])),
        indices=_resolve_path(str(path_data["indices"])),
        models=_resolve_path(str(path_data["models"])),
        results=_resolve_path(str(path_data["results"])),
        baseline_results=_resolve_path(str(path_data["baseline_results"])),
    )

    data_cfg = DataConfig(
        train_ratio=float(data_section["train_ratio"]),
        val_ratio=float(data_section["val_ratio"]),
        test_ratio=float(data_section["test_ratio"]),
        raw_audit_filename=str(data_section["raw_audit_filename"]),
        split_manifest_filename=str(data_section["split_manifest_filename"]),
        categories=[str(item) for item in data_section["categories"]],
    )
    _validate_ratios(data_cfg.train_ratio, data_cfg.val_ratio, data_cfg.test_ratio)

    baselines = BaselinesConfig(
        bert=_build_baseline_model(baselines_section["bert"]),
        roberta=_build_baseline_model(baselines_section["roberta"]),
    )

    retrieval = RetrievalConfig(
        k=int(retrieval_data["k"]),
        encoder=str(retrieval_data["encoder"]),
        phase1_encoders=[str(item) for item in retrieval_data["phase1_encoders"]],
        optional_encoders=[str(item) for item in retrieval_data.get("optional_encoders", [])],
        sbert_model=str(retrieval_data["sbert_model"]),
        bert_model=str(retrieval_data["bert_model"]),
        roberta_model=str(retrieval_data["roberta_model"]),
        strategy=str(retrieval_data["strategy"]),
        diversity_lambda=float(retrieval_data["diversity_lambda"]),
    )
    _validate_phase1_retrieval_encoders(retrieval)

    ollama = OllamaConfig(
        base_url=str(ollama_data["base_url"]),
        models=[str(item) for item in ollama_data["models"]],
        default_model=str(ollama_data["default_model"]),
        temperature=float(ollama_data["temperature"]),
        max_tokens=int(ollama_data["max_tokens"]),
        format=str(ollama_data["format"]),
    )

    evaluation = EvaluationConfig(
        rubric_dimensions=[str(item) for item in evaluation_data["rubric_dimensions"]],
        human_eval_sample_size=int(evaluation_data["human_eval_sample_size"]),
    )

    return AppConfig(
        project=project,
        paths=paths,
        data=data_cfg,
        baselines=baselines,
        retrieval=retrieval,
        ollama=ollama,
        evaluation=evaluation,
    )


def raw_audit_path(config: AppConfig | None = None) -> Path:
    cfg = config or load_config()
    return cfg.paths.processed_data / cfg.data.raw_audit_filename


def split_manifest_path(config: AppConfig | None = None) -> Path:
    cfg = config or load_config()
    return cfg.paths.processed_data / cfg.data.split_manifest_filename
