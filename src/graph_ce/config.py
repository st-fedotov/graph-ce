"""Configuration loading and validation.

The YAML config is the single source of truth for every hyperparameter.
Nothing else in the codebase should hardcode values that appear here.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class ProblemConfig(BaseModel):
    n: int = Field(ge=2, description="Graph size (number of vertices).")

    @property
    def num_edges(self) -> int:
        return self.n * (self.n - 1) // 2

    @property
    def conjecture_threshold(self) -> float:
        # Conjecture 2.1: lambda_1 + mu >= sqrt(n-1) + 1.
        # Score = threshold - lambda_1 - mu; score > 0 means counterexample.
        return math.sqrt(self.n - 1) + 1.0


class ModelConfig(BaseModel):
    hidden_sizes: list[int] = Field(min_length=1)
    learning_rate: float = Field(gt=0.0)
    optimizer: str = Field(pattern=r"^(sgd|adam)$")
    init: str = Field(pattern=r"^(keras|pytorch_default)$")


class CemConfig(BaseModel):
    n_sessions: int = Field(ge=1)
    elite_percentile: float = Field(ge=0.0, le=100.0)
    super_elite_percentile: float = Field(ge=0.0, le=100.0)
    max_iters: int = Field(ge=1)
    train_epochs_per_iter: int = Field(ge=1)
    train_batch_size: int = Field(ge=1)

    @model_validator(mode="after")
    def _check_percentiles(self) -> "CemConfig":
        if self.super_elite_percentile < self.elite_percentile:
            raise ValueError(
                "super_elite_percentile must be >= elite_percentile "
                f"(got {self.super_elite_percentile} < {self.elite_percentile})"
            )
        return self


class ParallelismConfig(BaseModel):
    n_islands: int = Field(ge=1)
    cores_per_island: int = Field(ge=1)
    start_method: str = Field(pattern=r"^(spawn|fork|forkserver)$")


class MigrationConfig(BaseModel):
    enabled: bool
    interval_iters: int = Field(ge=1)
    top_k: int = Field(ge=1)


class StoppingConfig(BaseModel):
    wall_clock_seconds: float = Field(gt=0.0)
    score_threshold: float


class LoggingConfig(BaseModel):
    log_interval_iters: int = Field(ge=1)
    metrics_interval_iters: int = Field(ge=1)
    output_dir: str
    stdout_mirror: bool


class SeedConfig(BaseModel):
    master_seed: Optional[int] = None


class Config(BaseModel):
    problem: ProblemConfig
    model: ModelConfig
    cem: CemConfig
    parallelism: ParallelismConfig
    migration: MigrationConfig
    stopping: StoppingConfig
    logging: LoggingConfig
    seed: SeedConfig

    @property
    def input_dim(self) -> int:
        # State bits (num_edges) + position one-hot (num_edges).
        return 2 * self.problem.num_edges

    def dump_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False)


def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


def apply_overrides(cfg_dict: dict, overrides: list[str]) -> dict:
    """Apply 'a.b.c=value' style overrides to a config dict.

    Values are parsed as YAML scalars, so '42', 'true', '[1,2]' all work.
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov!r}")
        key, _, value_str = ov.partition("=")
        keys = key.split(".")
        try:
            value = yaml.safe_load(value_str)
        except yaml.YAMLError as e:
            raise ValueError(f"Could not parse override value for {key}: {value_str!r}") from e
        cursor = cfg_dict
        for k in keys[:-1]:
            if k not in cursor or not isinstance(cursor[k], dict):
                raise ValueError(f"Override path does not exist: {key}")
            cursor = cursor[k]
        if keys[-1] not in cursor:
            raise ValueError(f"Override path does not exist: {key}")
        cursor[keys[-1]] = value
    return cfg_dict
