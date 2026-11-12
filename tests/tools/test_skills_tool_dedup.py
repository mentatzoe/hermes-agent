import json
from tools.skills_tool_dedup import _record_skill_view, _check_skill_view_dedup, reset_skill_view_dedup


def test_dedup_retains_resolved_source_identity(tmp_path):
    source = tmp_path / "fixture.md"
    source.write_text("Complete procedure.")
    payload = {"success": True, "name": "fixture", "_source_path": str(source)}
    reset_skill_view_dedup("fixture-task")
    _record_skill_view("fixture-task", "fixture", None, payload)
    stub = json.loads(_check_skill_view_dedup("fixture-task", "fixture", None))
    assert stub["_source_path"] == payload["_source_path"]
    assert stub["success"] and stub["dedup"]
    reset_skill_view_dedup("fixture-task")
