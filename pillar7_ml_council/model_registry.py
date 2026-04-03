from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelRecord:
    model_name: str
    agent_name: str
    version: str
    created_at_utc: str
    artifact_path: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    calibration_summary: Dict[str, Any] = field(default_factory=dict)
    train_summary: Dict[str, Any] = field(default_factory=dict)
    split_summary: Dict[str, Any] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    def __init__(self, registry_path: str = "pillar7_ml_council/artifacts/model_registry.json") -> None:
        self.registry_path = Path(registry_path)
        _ensure_dir(self.registry_path.parent)

        if not self.registry_path.exists():
            self._write_registry({"models": []})

    def _read_registry(self) -> Dict[str, Any]:
        with self.registry_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_registry(self, payload: Dict[str, Any]) -> None:
        with self.registry_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def register_model(
        self,
        *,
        model_name: str,
        agent_name: str,
        version: str,
        artifact_path: str,
        metrics: Optional[Dict[str, Any]] = None,
        calibration_summary: Optional[Dict[str, Any]] = None,
        train_summary: Optional[Dict[str, Any]] = None,
        split_summary: Optional[Dict[str, Any]] = None,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        registry = self._read_registry()

        record = ModelRecord(
            model_name=model_name,
            agent_name=agent_name,
            version=version,
            created_at_utc=_utc_now_iso(),
            artifact_path=artifact_path,
            metrics=metrics or {},
            calibration_summary=calibration_summary or {},
            train_summary=train_summary or {},
            split_summary=split_summary or {},
            feature_columns=feature_columns or [],
            target_column=target_column,
            notes=notes,
        )

        registry["models"].append(record.to_dict())
        self._write_registry(registry)

        return record.to_dict()

    def list_models(self) -> List[Dict[str, Any]]:
        registry = self._read_registry()
        return registry.get("models", [])

    def get_model(
        self,
        *,
        model_name: str,
        agent_name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        models = self.list_models()

        matches: List[Dict[str, Any]] = []
        for model in models:
            if model.get("model_name") != model_name:
                continue
            if agent_name is not None and model.get("agent_name") != agent_name:
                continue
            if version is not None and model.get("version") != version:
                continue
            matches.append(model)

        if not matches:
            return None

        matches = sorted(matches, key=lambda x: x.get("created_at_utc", ""))
        return matches[-1]

    def latest_for_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        models = [m for m in self.list_models() if m.get("agent_name") == agent_name]
        if not models:
            return None
        models = sorted(models, key=lambda x: x.get("created_at_utc", ""))
        return models[-1]


def summarize_model_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": record.get("model_name"),
        "agent_name": record.get("agent_name"),
        "version": record.get("version"),
        "created_at_utc": record.get("created_at_utc"),
        "artifact_path": record.get("artifact_path"),
        "target_column": record.get("target_column"),
        "feature_count": len(record.get("feature_columns", [])),
        "metrics": record.get("metrics", {}),
    }