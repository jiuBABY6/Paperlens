"""Allow-listed specialist tools and strict JSON schemas for model tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    agents: frozenset[str]

    def openai_schema(self, *, strict: bool = True) -> dict[str, Any]:
        parameters = deepcopy(self.parameters)
        if strict:
            # OpenAI-compatible strict schemas require every declared property
            # to be listed as required. Defaults are still enforced locally.
            parameters["required"] = list(parameters.get("properties", {}))
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
                "strict": strict,
            },
        }


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object", "properties": properties, "required": required,
        "additionalProperties": False,
    }


TOOL_SPECS = {
    "search_text": ToolSpec("search_text", "Search current-paper text evidence.", _object({
        "query": {"type": "string", "minLength": 2},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
    }, ["query"]), frozenset({"text_agent"})),
    "read_sentence": ToolSpec("read_sentence", "Read one retrieved sentence by evidence ID.", _object({
        "sentence_id": {"type": "string", "minLength": 1},
    }, ["sentence_id"]), frozenset({"text_agent"})),
    "search_figures": ToolSpec("search_figures", "Search figures in the current paper.", _object({
        "query": {"type": "string", "minLength": 2},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
    }, ["query"]), frozenset({"figure_agent"})),
    "read_figure": ToolSpec("read_figure", "Read figure metadata and offline description.", _object({
        "figure_id": {"type": "string", "minLength": 1},
    }, ["figure_id"]), frozenset({"figure_agent"})),
    "analyze_figure_for_query": ToolSpec("analyze_figure_for_query", "Inspect one figure image for the user question.", _object({
        "figure_id": {"type": "string", "minLength": 1},
        "question": {"type": "string", "minLength": 2},
    }, ["figure_id", "question"]), frozenset({"figure_agent"})),
    "search_tables": ToolSpec("search_tables", "Search tables in the current paper.", _object({
        "query": {"type": "string", "minLength": 2},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
    }, ["query"]), frozenset({"table_agent"})),
    "read_table": ToolSpec("read_table", "Read a table's structured rows and columns.", _object({
        "table_id": {"type": "string", "minLength": 1},
    }, ["table_id"]), frozenset({"table_agent"})),
    "analyze_table_image_with_vlm": ToolSpec("analyze_table_image_with_vlm", "Inspect a table image only when structured parsing is unusable.", _object({
        "table_id": {"type": "string", "minLength": 1},
        "question": {"type": "string", "minLength": 2},
    }, ["table_id", "question"]), frozenset({"table_agent"})),
}


def schemas_for(agent_name: str, *, strict: bool = True) -> list[dict[str, Any]]:
    return [
        spec.openai_schema(strict=strict)
        for spec in TOOL_SPECS.values() if agent_name in spec.agents
    ]


def validate_arguments(agent_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Small dependency-free JSON Schema subset sufficient for our closed registry."""
    spec = TOOL_SPECS.get(tool_name)
    if not spec or agent_name not in spec.agents:
        raise PermissionError(f"tool_not_allowed:{tool_name}")
    if not isinstance(arguments, dict):
        raise ValueError("arguments_must_be_object")
    schema = spec.parameters
    allowed = set(schema["properties"])
    if set(arguments) - allowed:
        raise ValueError("additional_properties_not_allowed")
    for name in schema["required"]:
        if name not in arguments:
            raise ValueError(f"missing_argument:{name}")
    value = dict(arguments)
    for name, rule in schema["properties"].items():
        if name not in value:
            continue
        if rule["type"] == "string":
            if not isinstance(value[name], str) or len(value[name].strip()) < rule.get("minLength", 0):
                raise ValueError(f"invalid_string:{name}")
            value[name] = value[name].strip()
        if rule["type"] == "integer":
            if not isinstance(value[name], int) or isinstance(value[name], bool):
                raise ValueError(f"invalid_integer:{name}")
            if value[name] < rule.get("minimum", value[name]):
                raise ValueError(f"integer_below_minimum:{name}")
            if value[name] > rule.get("maximum", value[name]):
                raise ValueError(f"integer_above_maximum:{name}")
    return value
