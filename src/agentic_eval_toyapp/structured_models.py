"""Shared base for structured model-output contracts."""

from types import UnionType
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, model_validator


def _allows_none(annotation: object) -> bool:
    """Report whether a field annotation is ``None`` or a union containing it."""
    if annotation is None or annotation is type(None):
        return True
    if get_origin(annotation) in (Union, UnionType):
        return any(_allows_none(argument) for argument in get_args(annotation))
    return False


class NullableOmittedAsNone(BaseModel):
    """Treat an omitted nullable key as an explicit ``null`` when parsing.

    Some gateway routes do not enforce the strict JSON schema sent with the
    request, so the model may leave out keys it considers empty. The schema is
    left untouched, and no ``= None`` default is declared, because a ``default``
    keyword in the schema is rejected by some strict-mode gateways. Fields whose
    annotation does not allow ``None`` (including list fields) stay required.
    """

    @model_validator(mode="before")
    @classmethod
    def _fill_omitted_nullable_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        filled: dict[Any, Any] = dict(data)  # pyright: ignore[reportUnknownArgumentType]
        for name, field in cls.model_fields.items():
            key = field.alias or name
            if _allows_none(field.annotation) and key not in filled:
                filled[key] = None
        return filled
