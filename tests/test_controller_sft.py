from cl_gism.controller_sft import parse_json_object, training_messages, trajectory_id


def test_training_messages_are_compact_json() -> None:
    row = {
        "source": {"qid": 7},
        "input": {"question": "q"},
        "target": {"loop_decision": {"action": "CONTINUE_CURRENT_LOOP"}},
    }
    messages = training_messages(row)
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[1]["content"] == '{"question":"q"}'
    assert trajectory_id(row) == "7"


def test_parse_json_object_tolerates_fence() -> None:
    assert parse_json_object('```json\n{"ok": true}\n```') == {"ok": True}
    assert parse_json_object("not json") is None
