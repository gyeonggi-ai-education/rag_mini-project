"""Print an unverified initial set, or compare explicitly configured adapters.

Default: uv run python scripts/evaluate.py
Execution: --run --settings settings.json --adapter module:factory --output report.json
factory(RunSettings) -> (retriever, answerer_or_None). The retriever must declare
run_settings matching its actual configuration. No default provider is assumed.
Reports include fixed evaluation questions; avoid using private user questions.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.evaluation import EvaluationSet, RunSettings, compare, initial_dataset


def main(argv=None, *, factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Versioned EvaluationSet JSON; default: reviewed 8+2 set")
    parser.add_argument("--settings", type=Path, help="Explicit RunSettings JSON, required for --run")
    parser.add_argument("--adapter", help="Explicit Python module:factory for verified adapters")
    parser.add_argument("--run", action="store_true", help="Execute read-only search/answer adapters")
    parser.add_argument("--output", type=Path, help="Save report JSON; otherwise print to stdout")
    args = parser.parse_args(argv)
    try:
        data = EvaluationSet.model_validate_json(args.dataset.read_text(encoding="utf-8")) if args.dataset else initial_dataset()
        if args.run:
            if args.settings is None:
                raise ValueError("Settings required")
            settings = RunSettings.model_validate_json(args.settings.read_text(encoding="utf-8"))
            if factory is None:
                if not args.adapter or args.adapter.count(":") != 1:
                    raise ValueError("Adapter required")
                module, name = args.adapter.split(":")
                factory = getattr(importlib.import_module(module), name)
            retriever, answerer = factory(settings)
            report = compare(data, settings, retriever=retriever, answerer=answerer)
        else:
            if args.settings is not None or args.adapter is not None:
                raise ValueError("Use --run to execute configured adapters")
            report = {"dataset": data.model_dump(), "dataset_sha256": data.fingerprint(),
                      "external_verified": False, "metrics": None,
                      "status": "unverified",
                      "reason": "No external evaluation executed. Supply verified indexing/embedding settings and explicit Rerank/expansion adapters with --run."}
        encoded = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(encoded + "\n", encoding="utf-8")
        else:
            print(encoded)
        if args.run and any(record["error"] for result in report["modes"].values()
                            for record in result["records"]):
            return 1
    except Exception:
        print("Evaluation failed: check dataset, settings and configured adapters.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
