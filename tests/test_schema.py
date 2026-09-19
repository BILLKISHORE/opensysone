import json

import pytest

from opensysone.schema import (
    NONE_OPTION,
    CompiledRequest,
    SchemaError,
    compile_request,
    render_state,
    text_of,
)
from opensysone.wire import SystemOneRequest


def _compile(questions, state="Everything is down.", **kw) -> CompiledRequest:
    return compile_request(SystemOneRequest.model_validate({"state": state, "questions": questions, **kw}))


def test_text_of_renders_json_content_canonically():
    assert text_of("plain") == "plain"
    assert text_of(None) == ""
    assert text_of({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'


def test_render_state_keeps_strings_and_pretty_prints_json():
    assert render_state("hi") == "hi"
    out = render_state({"z": [1, 2], "a": "x"})
    assert json.loads(out) == {"z": [1, 2], "a": "x"}
    assert out.startswith("{\n")


def test_noul_compiles_to_boolean_with_criteria_in_description():
    c = _compile({"urgent": {"type": "noul", "instructions": "Reply within the hour?",
                             "criteria": {"true": "needs a reply now", "false": "can wait"}}})
    spec = c.schema_dict["urgent"]
    assert spec["type"] == "boolean"
    assert "Reply within the hour?" in spec["description"]
    assert "needs a reply now" in spec["description"]
    plan = c.plans["urgent"]
    assert plan.kind == "noul"
    assert plan.options == ("true", "false")


def test_choice_compiles_to_enum_with_glosses():
    c = _compile({"team": {"type": "choice", "instructions": "Which team?",
                           "criteria": {"outage": "service down", "billing": "charges"}}})
    spec = c.schema_dict["team"]
    assert spec["type"] == "enum"
    assert spec["choices"] == ["outage", "billing"]
    assert spec["choice_descriptions"] == {"outage": "service down", "billing": "charges"}
    assert c.plans["team"].options == ("outage", "billing")
    assert c.plans["team"].abstain_option is None


def test_score_compiles_to_indexed_enum_with_legend():
    c = _compile({"tone": {"type": "score", "instructions": "How upset?",
                           "criteria": ["calm", "annoyed", "furious"]}})
    spec = c.schema_dict["tone"]
    assert spec["choices"] == ["0", "1", "2"]
    assert spec["choice_descriptions"]["2"] == "furious"
    assert c.plans["tone"].kind == "score"
    assert c.plans["tone"].legend == ("calm", "annoyed", "furious")


def test_missing_instructions_fall_back_to_qid_words():
    c = _compile({"needs_human": {"type": "noul"}})
    assert c.schema_dict["needs_human"]["description"] == "needs human"


def test_context_is_rendered_state():
    c = _compile({"q": {"type": "noul"}}, state={"a": 1})
    assert json.loads(c.context) == {"a": 1}


def test_empty_option_name_is_a_schema_error():
    with pytest.raises(SchemaError) as e:
        _compile({"team": {"type": "choice", "criteria": {"": "x", "b": "y"}}})
    assert e.value.loc == ["body", "questions", "team", "criteria"]


def test_multi_compiles_with_set_constraints():
    c = _compile({"tags": {"type": "multi", "instructions": "Which apply?",
                           "criteria": {"x": "ex", "y": "why", "z": "zed"},
                           "constraints": [{"type": "mutually_exclusive", "options": ["x", "y"]}]}})
    spec = c.schema_dict["tags"]
    assert spec["type"] == "multi"
    assert spec["choices"] == ["x", "y", "z"]
    assert spec["set_constraints"] == [{"type": "mutually_exclusive", "options": ["x", "y"]}]
    assert c.plans["tags"].kind == "multi"


def test_pairwise_compiles_to_two_option_enum():
    c = _compile({"better": {"type": "pairwise", "a": "Answer one", "b": "Answer two"}})
    spec = c.schema_dict["better"]
    assert spec["choices"] == ["a", "b"]
    assert spec["choice_descriptions"] == {"a": "Answer one", "b": "Answer two"}
    assert spec["description"].startswith("Compare candidate a and candidate b")
    assert c.plans["better"].kind == "pairwise"


def test_abstain_adds_none_option_last():
    c = _compile({"team": {"type": "choice", "criteria": {"a": "x", "b": "y"}, "abstain": True}})
    assert c.schema_dict["team"]["choices"] == ["a", "b", NONE_OPTION]
    assert c.plans["team"].abstain_option == NONE_OPTION


def test_abstain_rejects_reserved_name():
    with pytest.raises(SchemaError):
        _compile({"team": {"type": "choice", "criteria": {NONE_OPTION: "x", "b": "y"}, "abstain": True}})


def test_case_level_constraints_pass_through_when_valid():
    c = _compile(
        {"refund": {"type": "noul"}, "team": {"type": "choice", "criteria": {"billing": "b", "ops": "o"}}},
        constraints=[{"type": "implies", "parent": "refund", "child": "team", "mapping": {"true": "billing"}}],
    )
    assert c.constraints[0]["type"] == "implies"


def test_bad_constraint_type_is_schema_error():
    with pytest.raises(SchemaError) as e:
        _compile({"q": {"type": "noul"}}, constraints=[{"type": "sometimes"}])
    assert e.value.loc[:2] == ["body", "constraints"]


def test_contradictory_set_constraints_surface_as_schema_error():
    with pytest.raises(SchemaError) as e:
        _compile({"flags": {"type": "multi", "criteria": {"x": "", "y": ""},
                            "constraints": [{"type": "mutually_exclusive", "options": ["x", "y"]},
                                            {"type": "implies", "if_option": "x", "then_option": "y"}]}})
    assert "flags" in e.value.msg or e.value.loc[2] == "flags"
