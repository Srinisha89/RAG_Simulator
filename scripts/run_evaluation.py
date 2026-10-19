import json
from pathlib import Path

from rag_simulator.evaluation import run_default_evaluation


ROOT = Path(__file__).resolve().parents[1]

rows, summary, csv_path, report_path = run_default_evaluation(project_root=ROOT)
print(json.dumps(summary, indent=2))
print(f"CSV: {csv_path}")
print(f"Summary: {report_path}")