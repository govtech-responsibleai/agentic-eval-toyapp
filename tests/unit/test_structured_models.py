"""Tests for parsing omitted nullable keys as explicit nulls."""

from typing import Any

import pytest
from pydantic import ConfigDict, Field, ValidationError

from agentic_eval_toyapp.structured_models import NullableOmittedAsNone


class _Sample(NullableOmittedAsNone):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    nickname: str | None
    age: int | None
    legacy: str | None
    tags: tuple[str, ...]
    renamed: str | None = Field(alias="other")


_FULL: dict[str, Any] = {
    "name": "Ada",
    "nickname": None,
    "age": None,
    "legacy": None,
    "tags": [],
    "other": None,
}


def _without(*keys: str) -> dict[str, Any]:
    return {key: value for key, value in _FULL.items() if key not in keys}


def test_omitted_nullable_keys_equal_explicit_nulls() -> None:
    omitted = _Sample.model_validate(_without("nickname", "age", "legacy", "other"))

    assert omitted == _Sample.model_validate(_FULL)
    assert omitted.nickname is None
    assert omitted.age is None
    assert omitted.legacy is None
    assert omitted.renamed is None


def test_present_values_are_never_overwritten() -> None:
    parsed = _Sample.model_validate(
        {**_FULL, "nickname": "Ad", "age": 0, "other": "alias value"}
    )

    assert (parsed.nickname, parsed.age, parsed.renamed) == ("Ad", 0, "alias value")


@pytest.mark.parametrize("key", ["name", "tags"])
def test_non_nullable_and_list_fields_stay_required(key: str) -> None:
    with pytest.raises(ValidationError, match="Field required"):
        _Sample.model_validate(_without(key))


def test_extra_keys_are_still_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _Sample.model_validate({**_FULL, "unexpected": 1})


@pytest.mark.parametrize("payload", ["[]", '"text"', "null", "7"])
def test_non_object_input_fails_normally(payload: str) -> None:
    with pytest.raises(ValidationError):
        _Sample.model_validate_json(payload)


def test_json_schema_has_no_defaults_and_keeps_required_list() -> None:
    schema = _Sample.model_json_schema(by_alias=True)

    assert "default" not in str(schema)
    assert schema["required"] == ["name", "nickname", "age", "legacy", "tags", "other"]
