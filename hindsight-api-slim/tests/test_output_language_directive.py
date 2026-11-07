"""The configurable output-language directive (RUN-2355): English in Latin letters, the speakers' own words kept.

The stock directive translates everything, "entity names" included. For code-mixed chat (Benglish) that turns
Gurudev, bara, bokachoda and tui into English or drops them, and recall can no longer match the words people use.
``HINDSIGHT_API_LLM_OUTPUT_LANGUAGE_DIRECTIVE=keep-words`` (or any text) replaces the wording for retain,
consolidation and reflect alike, and keeps the mutual exclusion with the source-language rule.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import hindsight_api.config as config_module
from hindsight_api.config import HindsightConfig
from hindsight_api.engine.consolidation.prompts import build_consolidation_system_prompt
from hindsight_api.engine.prompt_utils import (
    KEEP_WORDS_OUTPUT_LANGUAGE_DIRECTIVE,
    STOCK_OUTPUT_LANGUAGE_DIRECTIVE,
    output_language_directive,
)
from hindsight_api.engine.reflect.prompts import build_agent_user_prompt, build_final_prompt
from hindsight_api.engine.reflect.tools_schema import _done_tool_for_output_language
from hindsight_api.engine.retain.fact_extraction import (
    _DEFAULT_LANGUAGE_RULE as RETAIN_SOURCE_LANGUAGE_RULE,
)
from hindsight_api.engine.retain.fact_extraction import (
    _EXAMPLES_SOURCE_LANGUAGE_NOTE,
    _build_extraction_prompt_and_schema,
)


def _with_directive(monkeypatch, directive):
    monkeypatch.setattr(config_module, "get_config", lambda: SimpleNamespace(llm_output_language_directive=directive))


def _retain_prompt(language):
    config = MagicMock()
    config.entity_labels = None
    config.entities_allow_free_form = True
    config.retain_extraction_mode = "concise"
    config.retain_extract_causal_links = False
    config.retain_mission = None
    config.retain_custom_instructions = None
    config.llm_output_language = language
    return _build_extraction_prompt_and_schema(config).system_prompt


def test_unset_keeps_the_stock_directive_byte_for_byte(monkeypatch):
    _with_directive(monkeypatch, None)

    assert output_language_directive("French") == "\n\n" + STOCK_OUTPUT_LANGUAGE_DIRECTIVE.replace(
        "{language}", "French"
    )
    assert "Translate any source content into French" in output_language_directive("French")
    assert output_language_directive(None) == ""


def test_keep_words_writes_latin_letters_and_keeps_the_speakers_words(monkeypatch):
    _with_directive(monkeypatch, "keep-words")
    text = output_language_directive("English")

    assert text == "\n\n" + KEEP_WORDS_OUTPUT_LANGUAGE_DIRECTIVE.replace("{language}", "English")
    for required in (
        "English sentences in Latin letters only",
        "Never use Bengali, Devanagari or any other script",
        "word for word",
        "own Latin spelling",
        "name, nickname, form of address, slang word, swear word and quoted phrase",
        "Gurudev",
        "bara",
        "wanra",
        "bokachoda",
        "chutmarani",
        "tumi/tui/apni",
        "short English gloss in brackets only where the meaning would otherwise be lost",
        "Never translate, correct or clean them up",
    ):
        assert required in text, required
    # The stock instruction that destroys the words is gone.
    assert "Translate any source content" not in text
    assert "entity names" not in text.split("(for example")[0].replace("every name", "")


def test_any_other_value_is_the_directive_itself_with_the_language_filled_in(monkeypatch):
    _with_directive(monkeypatch, "Answer in {language}; keep names as typed.")

    assert output_language_directive("English") == "\n\nAnswer in English; keep names as typed."


def test_one_directive_text_closes_retain_consolidation_and_reflect(monkeypatch):
    _with_directive(monkeypatch, "keep-words")
    directive = output_language_directive("English")

    assert _retain_prompt("English").endswith(directive)
    assert build_consolidation_system_prompt(llm_output_language="English").endswith(directive)
    assert build_agent_user_prompt("q?", "English").endswith(directive)
    assert build_final_prompt("q?", [], {}, llm_output_language="English").endswith(directive)


def test_the_source_language_rule_is_still_dropped_when_a_language_is_set(monkeypatch):
    _with_directive(monkeypatch, "keep-words")

    with_language = _retain_prompt("English")
    without_language = _retain_prompt(None)

    assert RETAIN_SOURCE_LANGUAGE_RULE in without_language
    assert RETAIN_SOURCE_LANGUAGE_RULE not in with_language
    # The examples' header repeats that rule; it goes too, or the model would see it contradict the directive.
    assert _EXAMPLES_SOURCE_LANGUAGE_NOTE in without_language
    assert _EXAMPLES_SOURCE_LANGUAGE_NOTE not in with_language
    assert "ALL output values MUST be in the input language" not in with_language


def test_the_reflect_answer_tool_repeats_the_configured_directive_and_the_stock_rule_otherwise(monkeypatch):
    base = {
        "function": {
            "description": "Stop.",
            "parameters": {"properties": {"answer": {"description": "The answer."}}},
        }
    }
    _with_directive(monkeypatch, "keep-words")
    tool = _done_tool_for_output_language(base, "English")["function"]["description"]
    assert "Latin letters only" in tool and "bokachoda" in tool

    _with_directive(monkeypatch, None)
    tool = _done_tool_for_output_language(base, "English")["function"]["description"]
    assert "Write the answer exclusively in English" in tool


def test_the_environment_sets_it_and_a_server_started_without_it_has_none(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_API_LLM_OUTPUT_LANGUAGE_DIRECTIVE", raising=False)
    assert HindsightConfig.from_env().llm_output_language_directive is None
    monkeypatch.setenv("HINDSIGHT_API_LLM_OUTPUT_LANGUAGE_DIRECTIVE", "keep-words")
    assert HindsightConfig.from_env().llm_output_language_directive == "keep-words"


@pytest.mark.parametrize("language", ["English", "German"])
def test_the_language_is_the_configured_one(monkeypatch, language):
    _with_directive(monkeypatch, "keep-words")

    assert f"{language} sentences in Latin letters" in output_language_directive(language)
