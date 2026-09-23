from __future__ import annotations

import json

import numpy as np


def corrupt_triples(
    triples: np.ndarray,
    count: int,
    num_entities: int,
    num_relations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create negatives by corrupting one component of sampled positive triples."""
    if count < 2:
        raise ValueError("count must be at least 2")
    if len(triples) == 0:
        raise ValueError("cannot corrupt an empty set of triples")

    sampled = rng.choice(len(triples), size=count, replace=count > len(triples))
    corrupted = triples[sampled].copy()
    known_positives = {tuple(row) for row in triples.tolist()}
    for triple in corrupted:
        source = triple.copy()
        for _ in range(100):
            triple[:] = source
            column = int(rng.integers(0, 3))
            upper_bound = num_relations if column == 1 else num_entities
            original = triple[column]
            replacement = int(rng.integers(upper_bound))
            if upper_bound > 1:
                while replacement == original:
                    replacement = int(rng.integers(upper_bound))
            triple[column] = replacement
            if tuple(triple) not in known_positives:
                break
        else:
            raise RuntimeError("Could not generate a negative triple outside the positive set")
    return corrupted


def parse_generated_anomalies(content: str, expected_count: int) -> list[tuple[str, str, str]]:
    try:
        payload = json.loads(content)
        items = payload["anomalies"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("OpenRouter returned an invalid anomaly JSON document") from error
    if not isinstance(items, list) or len(items) != expected_count:
        raise ValueError(
            f"OpenRouter returned {len(items) if isinstance(items, list) else 0} anomalies; "
            f"expected {expected_count}"
        )

    anomalies: list[tuple[str, str, str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"Generated anomaly {index} is not an object")
        values = tuple(
            str(item.get(field, "")).strip() for field in ("subject", "relation", "object")
        )
        if any(not value for value in values):
            raise ValueError(f"Generated anomaly {index} contains an empty field")
        anomalies.append(values)
    if len(set(anomalies)) != len(anomalies):
        raise ValueError("OpenRouter returned duplicate anomalies")
    return anomalies


def validate_generated_anomalies(
    anomalies: list[tuple[str, str, str]],
    context_triples: list[tuple[str, str, str]],
    *,
    require_context_vocabulary: bool = False,
) -> None:
    context_set = set(context_triples)
    context_entities = {
        label for subject, _, object_ in context_triples for label in (subject, object_)
    }
    context_relations = {relation for _, relation, _ in context_triples}
    for anomaly in anomalies:
        subject, relation, object_ = anomaly
        if require_context_vocabulary and (
            subject not in context_entities
            or object_ not in context_entities
            or relation not in context_relations
        ):
            raise ValueError("A generated anomaly uses a label outside the snapshot context")
        if not require_context_vocabulary and (
            subject not in context_entities and object_ not in context_entities
        ):
            raise ValueError("A generated anomaly does not reuse an entity from the context")
        if anomaly in context_set:
            raise ValueError("OpenRouter reproduced a positive context triple")


def generate_anomalies_genai(
    count: int,
    context_triples: list[tuple[str, str, str]],
    *,
    api_key: str,
    model: str,
    seed: int,
    require_context_vocabulary: bool = False,
) -> list[tuple[str, str, str]]:
    """Generate implausible geopolitical triples through OpenRouter."""
    from openai import OpenAI

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")
    if not context_triples:
        raise ValueError("At least one context triple is required")

    selected_context = context_triples[:100]
    if require_context_vocabulary and len(context_triples) > 100:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(context_triples), size=100, replace=False))
        selected_context = [context_triples[int(index)] for index in indices]
    context_entities = sorted(
        {label for subject, _, object_ in selected_context for label in (subject, object_)}
    )
    context_relations = sorted({relation for _, relation, _ in selected_context})
    entity_to_token = {label: f"E{index}" for index, label in enumerate(context_entities)}
    relation_to_token = {label: f"R{index}" for index, label in enumerate(context_relations)}
    token_to_entity = {token: label for label, token in entity_to_token.items()}
    token_to_relation = {token: label for label, token in relation_to_token.items()}
    if require_context_vocabulary:
        context = [
            {
                "subject": entity_to_token[subject],
                "subject_label": subject,
                "relation": relation_to_token[relation],
                "relation_label": relation,
                "object": entity_to_token[object_],
                "object_label": object_,
            }
            for subject, relation, object_ in selected_context
        ]
    else:
        context = [
            {"subject": subject, "relation": relation, "object": object_}
            for subject, relation, object_ in selected_context
        ]
    subject_schema: dict[str, object] = {"type": "string"}
    relation_schema: dict[str, object] = {"type": "string"}
    object_schema: dict[str, object] = {"type": "string"}
    if require_context_vocabulary:
        subject_schema["enum"] = list(token_to_entity)
        relation_schema["enum"] = list(token_to_relation)
        object_schema["enum"] = list(token_to_entity)
    schema = {
        "name": "knowledge_graph_anomalies",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "anomalies": {
                    "type": "array",
                    "minItems": count,
                    "maxItems": count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject": subject_schema,
                            "relation": relation_schema,
                            "object": object_schema,
                        },
                        "required": ["subject", "relation", "object"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["anomalies"],
            "additionalProperties": False,
        },
    }
    vocabulary_instruction = (
        "Return only the safe E<number> entity tokens and R<number> relation tokens supplied in "
        "the context; the associated labels explain their meaning. "
        if require_context_vocabulary
        else "Each triple must reuse at least one subject or object appearing in the context. "
    )
    prompt = (
        f"Generate exactly {count} distinct, deliberately implausible geopolitical knowledge-graph "
        f"triples. {vocabulary_instruction}Events should be historically, geographically, or "
        "logically absurd. Do not reproduce a context triple. Context: "
        + json.dumps(context, ensure_ascii=False)
    )
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only data matching the requested JSON schema."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        temperature=0.8,
        seed=seed,
    )
    content = completion.choices[0].message.content
    if not content:
        raise ValueError("OpenRouter returned an empty response")
    anomalies = parse_generated_anomalies(content, count)
    if require_context_vocabulary:
        anomalies = [
            (
                token_to_entity[subject],
                token_to_relation[relation],
                token_to_entity[object_],
            )
            for subject, relation, object_ in anomalies
        ]
    validate_generated_anomalies(
        anomalies,
        selected_context if require_context_vocabulary else context_triples,
        require_context_vocabulary=require_context_vocabulary,
    )
    return anomalies
