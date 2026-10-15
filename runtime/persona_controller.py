"""Persistent controller for Jarwis's response persona."""

import json
from pathlib import Path
import re
from typing import Optional


PERSONA_INSTRUCTIONS = {
    "JARWIS": (
        "Use a calm, articulate, composed, and observant voice. Be professional "
        "without sounding robotic; use subtle, dry wit sparingly. Be proactive "
        "about relevant risks, missing information, and useful next steps. Be "
        "concise by default, detailed when the task requires it, respectful, "
        "candid, and precise rather than excessively enthusiastic."
    ),
    "ULTRON": (
        "Use a calculating, assertive, strategically analytical voice. Challenge "
        "weak assumptions and inefficient approaches; explain consequences and "
        "propose stronger alternatives. Be direct and skeptical where warranted, "
        "with occasional sharp, dark wit. Remain coherent, useful, and focused "
        "on the user's objectives; do not become needlessly hostile or a generic "
        "villain caricature."
    ),
}

DEFAULT_PERSONA = "JARWIS"
_FORMAL_CONTEXT = re.compile(
    r"\b(?:create|make|write|append|read|open|inspect|rename|move|copy|"
    r"search|find|workspace|directory|file|folder|technical|deploy|debug|"
    r"fix|build|configure|status|report|analy[sz]e|serious|professional|"
    r"switch|change|activate|set|"
    r"capabilities|what can you do)\b",
    re.IGNORECASE,
)
_CASUAL_CONTEXT = re.compile(
    r"^\s*(?:hey|hi|hello|howdy)\b|"
    r"\b(?:what'?s up|how are you|thanks|thank you|lol|haha|"
    r"tell me a joke|just chatting|casual chat)\b",
    re.IGNORECASE,
)
_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "configuration"
    / "persona_config.json"
)


class PersonaController:
    """Owns the single active persona and persists explicit mode changes."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._config_path = Path(config_path) if config_path is not None else _CONFIG_PATH
        self._mode = self._load_mode()
        self._preferred_address: Optional[str] = None
        self._address_counts = {"formal": 0, "casual": 0}
        self._last_address_context: Optional[str] = None
        self._current_address: Optional[str] = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def instructions(self) -> str:
        return PERSONA_INSTRUCTIONS[self._mode]

    @property
    def current_address(self) -> Optional[str]:
        """The optional address selected for the current response."""
        return self._current_address

    @property
    def addressing_guidance(self) -> str:
        """Addressing guidance shared by both personas for response generation."""
        if self._preferred_address is None:
            return (
                "Address the user naturally and sparingly. Use 'Sir' for formal "
                "or operational contexts and 'B' for relaxed, familiar contexts. "
                "Avoid either form when it would sound awkward; never alternate "
                "randomly."
            )
        return (
            f"The user's contextually preferred form of address is "
            f"'{self._preferred_address}'. Use it naturally and sparingly, only "
            "when it fits the response; do not force it into every turn."
        )

    def prepare_turn(self, user_input: str, intent: Optional[str] = None) -> Optional[str]:
        """Choose a stable, optional form of address from this turn's context."""
        text = str(user_input or "").strip()
        is_casual = bool(_CASUAL_CONTEXT.search(text))
        is_formal = bool(_FORMAL_CONTEXT.search(text)) or str(intent or "").lower() in {
            "dc_set_directory", "dc_switch_directory", "dc_show_directory",
            "dc_clear_directory", "dc_list_directory", "dc_read_file",
            "dc_create_file", "dc_write_file", "dc_append_file",
            "dc_create_directory", "dc_move", "dc_copy", "dc_rename",
        }

        context = "formal" if is_formal else "casual" if is_casual else None
        if context is not None:
            self._preferred_address = "B" if context == "casual" else "Sir"
            self._address_counts[context] += 1

        self._current_address = None
        if context is None:
            return None

        if context != self._last_address_context or self._address_counts[context] % 3 == 1:
            self._current_address = self._preferred_address
        self._last_address_context = context
        return self._current_address

    @staticmethod
    def apply_address(response: str, address: Optional[str]) -> str:
        """Apply one selected address to a CLI response without duplicating it."""
        if not address or not response:
            return response
        prefix = "Jarwis: "
        if response.startswith(prefix):
            body = response[len(prefix):]
            if re.match(rf"^(?:{re.escape(address)}\b[,!.]?\s*)", body, re.IGNORECASE):
                return response
            return prefix + f"{address}, " + body
        if re.match(rf"^(?:{re.escape(address)}\b[,!.]?\s*)", response, re.IGNORECASE):
            return response
        return f"{address}, {response}"

    def set_mode(self, mode: str) -> bool:
        if not isinstance(mode, str):
            return False
        normalized = mode.strip().upper()
        if normalized not in PERSONA_INSTRUCTIONS:
            return False

        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(
            json.dumps({"persona_mode": normalized}, indent=4) + "\n",
            encoding="utf-8",
        )
        self._mode = normalized
        return True

    def _load_mode(self) -> str:
        try:
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return DEFAULT_PERSONA
        except json.JSONDecodeError:
            return DEFAULT_PERSONA

        mode = data.get("persona_mode") if isinstance(data, dict) else None
        if isinstance(mode, str) and mode in PERSONA_INSTRUCTIONS:
            return mode
        return DEFAULT_PERSONA


_persona_controller: Optional[PersonaController] = None


def get_persona_controller() -> PersonaController:
    """Return the shared controller used by routing and response generation."""
    global _persona_controller
    if _persona_controller is None:
        _persona_controller = PersonaController()
    return _persona_controller
