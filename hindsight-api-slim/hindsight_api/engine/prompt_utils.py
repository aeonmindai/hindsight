"""Shared utilities for prompt assembly."""

import re

_LONE_OPEN_BRACE = re.compile(r"(?<!\{)\{(?!\{)")
_LONE_CLOSE_BRACE = re.compile(r"(?<!\})\}(?!\})")

#: Max characters of a memory's ``context`` shown in internal prompts
#: (consolidation, reflect tool results, think). A caller can store a huge
#: context on every fact; repeated per memory it would bloat the prompt.
PROMPT_CONTEXT_MAX_CHARS = 200


def escape_for_prompt(text: str) -> str:
    """Double any lone ``{`` / ``}`` so the text survives ``str.format`` untouched.

    Prompt templates are often passed through ``str.format`` to substitute real
    placeholders like ``{facts_text}``.  Any literal braces in caller-supplied
    text — e.g. a bank mission that contains JSON examples — would otherwise be
    interpreted as format keys and raise ``KeyError``.

    Idempotent: text that already contains escaped ``{{`` / ``}}`` pairs is
    left as-is.  Only lone braces (not adjacent to another brace of the same
    kind) are doubled.
    """
    text = _LONE_OPEN_BRACE.sub("{{", text)
    text = _LONE_CLOSE_BRACE.sub("}}", text)
    return text


def truncate_context_for_prompt(context: str | None) -> str | None:
    """Cut a memory's context to ``PROMPT_CONTEXT_MAX_CHARS``, adding ``...`` when cut."""
    if context and len(context) > PROMPT_CONTEXT_MAX_CHARS:
        return context[:PROMPT_CONTEXT_MAX_CHARS] + "..."
    return context


# The stock directive: everything, entity names included, is translated into the language.
STOCK_OUTPUT_LANGUAGE_DIRECTIVE = (
    "IMPORTANT: Respond exclusively in {language}. "
    "Translate any source content into {language}. "
    "All output text — including fact text, observations, entity names, "
    "and the final response — must be in {language}."
)

# For code-mixed chat (Benglish: Bengali in Latin letters inside English). The language is written in Latin letters
# and the speakers' own words stay as they typed them, so recall and search still match what people actually say.
KEEP_WORDS_OUTPUT_LANGUAGE_DIRECTIVE = (
    "IMPORTANT: Write every fact, observation and answer as {language} sentences in Latin letters only. "
    "Never use Bengali, Devanagari or any other script. "
    "Keep word for word, in the speaker's own Latin spelling, every name, nickname, form of address, slang word, "
    "swear word and quoted phrase (for example Gurudev, bara, wanra, bokachoda, chutmarani, tumi/tui/apni). "
    "Add a short {language} gloss in brackets only where the meaning would otherwise be lost. "
    "Never translate, correct or clean them up."
)

KEEP_WORDS_DIRECTIVE_NAME = "keep-words"


def output_language_directive(language: str | None) -> str:
    """Return an LLM directive forcing all output into ``language``.

    Used by retain (fact extraction), consolidation (observations), and reflect
    (response synthesis) so HINDSIGHT_API_LLM_OUTPUT_LANGUAGE applies uniformly
    across every LLM-generated artifact. Returns an empty string when
    ``language`` is unset so the calling prompt stays unchanged.

    The wording is the stock directive unless HINDSIGHT_API_LLM_OUTPUT_LANGUAGE_DIRECTIVE names
    ``keep-words`` (the built-in directive for code-mixed chat) or gives its own text, in which
    ``{language}`` stands for the language. It is one text for all three pipelines, so they
    cannot disagree.
    """
    if not language:
        return ""
    from ..config import get_config

    configured = get_config().llm_output_language_directive
    if not configured:
        template = STOCK_OUTPUT_LANGUAGE_DIRECTIVE
    elif configured.strip() == KEEP_WORDS_DIRECTIVE_NAME:
        template = KEEP_WORDS_OUTPUT_LANGUAGE_DIRECTIVE
    else:
        template = configured
    return "\n\n" + template.replace("{language}", language)


def default_language_section(default_rule: str, language: str | None) -> str:
    """The preserve-the-source-language rule, unless ``language`` overrides it.

    The other half of :func:`output_language_directive`, and its mutual exclusion. Each
    pipeline runs an all-English prompt, so with no configured language a multilingual
    model drifts to English (or, per #181, to an unrelated language) and needs telling to
    keep the input's language. That rule flatly contradicts "translate everything into X",
    and the model resolves the contradiction in favour of the source-language rule — it is
    phrased more forcefully and comes first — silently no-opping the setting (#3776). So an
    explicit language drops the rule outright rather than arguing with it.

    ``default_rule`` is the pipeline's own wording (retain, consolidation and reflect each
    say it differently); only the selection is shared. Every prompt that carries such a rule
    calls this — retain, consolidation, and both of reflect's system prompts — and reflect's
    ``done`` tool schema applies the same exclusion by hand. Nothing may append a
    source-language rule of its own, or it reintroduces exactly the contradiction this
    exists to remove. Returns the rule followed by a blank line, ready to prepend to a
    prompt body, or ``""`` when ``language`` is set.
    """
    return "" if language else f"{default_rule}\n\n"
