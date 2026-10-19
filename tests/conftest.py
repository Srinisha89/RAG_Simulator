import json
from pathlib import Path

import pytest


@pytest.fixture
def policy_documents():
    path = Path(__file__).resolve().parents[1] / "data" / "policy_documents.json"
    return json.loads(path.read_text(encoding="utf-8"))