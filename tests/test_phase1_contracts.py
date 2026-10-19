import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PhaseOneFixtureContracts(unittest.TestCase):
    def load_json(self, relative_path: str):
        return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))

    def test_employee_fixtures_are_synthetic_and_complete(self):
        employees = self.load_json("data/employees.json")
        self.assertEqual(len(employees), 15)
        required_fields = {
            "employee_id",
            "name",
            "department",
            "visa_type",
            "visa_expiry_date",
            "extension_status",
            "spouse_visa_type",
            "us_start_date",
            "group_ids",
        }
        for employee in employees:
            self.assertTrue(required_fields.issubset(employee))
            self.assertIn(employee["visa_type"], {"H-1B", "L-1"})
            self.assertIn(
                employee["extension_status"], {"none", "pending", "approved"}
            )
            self.assertTrue(employee["employee_id"].startswith("SYN-"))
            self.assertIn("fictional", employee["name"].lower())

    def test_policy_documents_cover_required_synthetic_cases(self):
        documents = self.load_json("data/policy_documents.json")
        self.assertEqual(len(documents), 20)
        for document in documents:
            self.assertTrue(document["title"].startswith("FICTIONAL:"))
            self.assertIn(document["type"], {"company_policy", "uscis_style_guidance"})
            self.assertTrue(document["sections"])
            self.assertTrue(document["version"])
            self.assertTrue(document["last_updated"])
            self.assertTrue(document["access_group_ids"])

        self.assertTrue(any(document.get("version_history") for document in documents))
        self.assertTrue(any(document.get("restricted") for document in documents))
        self.assertTrue(any(document.get("stale_trap") for document in documents))
        self.assertTrue(any(document.get("exception_split_across_sections") for document in documents))

    def test_app_exposes_demo_safety_copy_and_roles(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("FICTIONAL DATA", app_source)
        self.assertIn("NOT LEGAL ADVICE", app_source)
        self.assertEqual(app_source.count('"Question"'), 1)
        for role in ("employee", "immigration_team", "admin"):
            self.assertIn(role, app_source)
        self.assertIn("simulation, not authentication", app_source.lower())

    def test_anthropic_live_mode_is_locked_off(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('"Anthropic live mode (disabled)"', app_source)
        self.assertIn("live_model_enabled = False", app_source)


if __name__ == "__main__":
    unittest.main()