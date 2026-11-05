"""Ask Groq for JSON, validate it, and retry once if it is invalid."""

import json
import re

from pydantic import ValidationError

from classbuild.llm import GenerationError

MAX_ATTEMPTS = 2  # first try + one retry when the output is invalid

BASE_SYSTEM_PROMPT = (
    "You are an expert curriculum designer. Respond with a single JSON object only. "
    "Do not use markdown code fences and do not add any text before or after the JSON."
)

_UNSUPPORTED_KEYWORDS = (
    "title", "default", "minLength", "maxLength", "minItems", "maxItems",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "description",
)


def to_strict_schema(schema: dict) -> dict:
    """Turn a Pydantic JSON schema into a strict, self-contained schema.

    References are inlined, every property is required and extra properties are
    forbidden. Size limits are dropped here because Pydantic validates them afterwards.
    """
    definitions = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(definitions[node["$ref"].split("/")[-1]])
        result = {key: value for key, value in node.items() if key not in _UNSUPPORTED_KEYWORDS and key != "$defs"}
        if "properties" in result:
            result["properties"] = {name: walk(value) for name, value in result["properties"].items()}
            result["required"] = list(result["properties"])
            result["additionalProperties"] = False
        if "items" in result:
            result["items"] = walk(result["items"])
        for combiner in ("anyOf", "oneOf", "allOf"):
            if combiner in result:
                result[combiner] = walk(result[combiner])
        return result

    return walk(schema)


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in the response.")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("The response is not a JSON object.")
    return data


def request_structured_json(
    client,
    *,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    schema_name: str,
    validate,
    label: str,
    retry_system_prompt: str | None = None,
    history: list[dict] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
):
    """Return validate(data). `history` (earlier chat turns) is sent on every attempt."""
    last_error = ""
    for attempt in range(max_attempts):
        prompt = user_prompt
        system = system_prompt
        if attempt > 0:
            system = retry_system_prompt or system_prompt
            prompt += (
                f"\n\nYour previous answer was rejected: {last_error}\n"
                "Return a corrected JSON object only."
            )
        raw = client.generate_json(
            system, prompt, schema=schema, schema_name=schema_name, history=history
        )
        try:
            return validate(extract_json(raw))
        except (ValueError, ValidationError) as exc:
            last_error = str(exc)[:300]
    raise GenerationError(
        f"The model did not return a valid {label} after {max_attempts} attempts. Please try again."
    )
