from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def _write_status(path: Path, **values: object) -> None:
    current: dict[str, object] = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(values)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _run(stage: str, command: list[str], status_path: Path) -> None:
    _write_status(status_path, state="running", stage=stage, command=command)
    print(f"[{stage}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def run(config: str, target_rows: int) -> None:
    artifact_dir = Path("artifacts/9-step-on-people")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    status_path = artifact_dir / "overnight_status.json"
    started = datetime.now(timezone.utc).isoformat()
    _write_status(
        status_path,
        model_family="9-step-on-people",
        state="running",
        stage="starting",
        pid=os.getpid(),
        started_at=started,
    )
    try:
        _run(
            "prepare_clean_data",
            [sys.executable, "-m", "onebit_llm.prepare_clean", "--target-rows", str(target_rows)],
            status_path,
        )
        _run(
            "train_both_models",
            [sys.executable, "-m", "onebit_llm.train", "--model", "all", "--config", config],
            status_path,
        )
        _run(
            "evaluate_fixed_prompts",
            [sys.executable, "-m", "onebit_llm.evaluate", "--config", config],
            status_path,
        )
        _write_status(
            status_path,
            state="completed",
            stage="done",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
    except BaseException as error:
        _write_status(
            status_path,
            state="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            failed_at=datetime.now(timezone.utc).isoformat(),
        )
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare, train, and evaluate 9-step-on-people")
    parser.add_argument("--config", default="configs/9-step-on-people.yaml")
    parser.add_argument("--target-rows", type=int, default=80_000)
    args = parser.parse_args(argv)
    run(args.config, args.target_rows)


if __name__ == "__main__":
    main()
