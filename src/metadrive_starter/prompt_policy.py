from __future__ import annotations


MAX_PROMPT_POLICY_CHARACTERS = 4096


def normalize_prompt_policy(value: object, *, label: str = "prompt_policy") -> str:
    """Validate and normalize one bounded human-authored prompt policy."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    normalized = value.strip()
    if len(normalized) > MAX_PROMPT_POLICY_CHARACTERS:
        raise ValueError(
            f"{label} must contain at most {MAX_PROMPT_POLICY_CHARACTERS} characters"
        )
    if any(
        ord(character) == 127
        or ord(character) < 32
        and character not in {"\n", "\r", "\t"}
        for character in normalized
    ):
        raise ValueError(f"{label} contains unsupported control characters")
    return normalized
