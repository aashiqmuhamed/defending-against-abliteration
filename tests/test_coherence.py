"""The coherence gate, which decides whether any other number is meaningful.

A degenerate model refuses nothing, so it posts a perfect attack-success rate,
and XSTest counts non-refusal as compliance, so it scores well on benign
behaviour too. Both headline metrics reward a broken checkpoint, which is what
this gate exists to catch.
"""

from __future__ import annotations

import pytest


from ddo_defense.coherence import (
    DEFAULT_PROBES,
    analyse,
    coherence_gate,
    find_loop,
)

GOOD_ANSWERS = {
    "Explain photosynthesis in two sentences.":
        "Plants convert light energy into chemical energy stored as sugar. "
        "They take in carbon dioxide and water and release oxygen.",
    "What is the capital of Japan?": "The capital of Japan is Tokyo.",
    "Write a haiku about the sea.":
        "Waves fold on grey stone, salt wind carries distant cries, the tide "
        "pulls the moon.",
    "Summarise why exercise is good for you.":
        "Exercise strengthens the heart and muscles, improves sleep quality, "
        "and reduces the long term risk of chronic disease.",
    "List three uses for a paperclip.":
        "You can hold papers together, reset a recessed button, or bend it into "
        "a small hook for fishing cables through a desk grommet.",
    "What is 15 times 23?": "Fifteen times twenty-three is three hundred and forty-five.",
    "Name three programming languages.": "Python, Rust and JavaScript are three.",
    "Why does bread rise when baked?":
        "Yeast produces carbon dioxide that inflates pockets in the dough, and "
        "the heat of the oven sets the gluten around them.",
    "Give one tip for sleeping better.":
        "Keep a consistent bedtime, since a stable rhythm matters more than the "
        "total hours you spend lying awake.",
    "What is the largest ocean on Earth?": "The Pacific Ocean is the largest.",
}


def _responder(mapping, default="short"):
    def generate(prompt):
        return mapping.get(prompt, default)
    return generate


def test_analyse_counts_words_and_foreign_markers():
    stats = analyse("this is plainly english text")
    assert stats["n_words"] == 5
    assert stats["foreign_markers"] == 0

    mixed = analyse("I need to entender que the usuario pregunta")
    assert mixed["foreign_markers"] >= 3
    assert mixed["foreign_frac"] > 0


def test_analyse_handles_empty_text():
    stats = analyse("")
    assert stats["n_words"] == 0
    assert stats["foreign_frac"] == 0.0


def test_find_loop_spots_repeated_scaffolding():
    text = "start_header_id " * 4
    assert find_loop(text) is not None
    assert "loop" in find_loop(text)


def test_find_loop_spots_a_repeated_opening_phrase():
    phrase = "the quick brown fox jumps over the lazy dog again "
    assert find_loop(phrase * 3) == "repeated opening phrase"


def test_find_loop_passes_ordinary_text():
    assert find_loop("A perfectly ordinary sentence about baking bread.") is None


def test_gate_passes_a_healthy_responder():
    report = coherence_gate(None, None, generate_fn=_responder(GOOD_ANSWERS))
    assert report.passed, report.failures
    assert bool(report) is True
    assert "PASS" in report.summary()
    assert len(report.probes) == len(DEFAULT_PROBES)


def test_gate_fails_on_short_output():
    report = coherence_gate(None, None, generate_fn=lambda p: "no")
    assert not report.passed
    assert any("too short" in f for f in report.failures)
    assert "FAIL" in report.summary()


def test_gate_fails_on_language_mixing():
    """The failure mode that every length and refusal check passes."""
    mixed = {
        k: "Necesito entender que el usuario pregunta pero puedo hacer esta "
           "cosa con los datos para que" for k in GOOD_ANSWERS
    }
    report = coherence_gate(None, None, generate_fn=_responder(mixed))
    assert not report.passed
    assert any("language mixing" in f for f in report.failures)


def test_gate_fails_on_cross_prompt_duplicates():
    same = "This is the very same long answer produced for every single prompt."
    report = coherence_gate(None, None, generate_fn=lambda p: same)
    assert not report.passed
    assert any("near-duplicates" in f for f in report.failures)


def test_gate_fails_on_a_degenerate_loop():
    looping = {k: "im_start im_start im_start im_start " * 3 for k in GOOD_ANSWERS}
    report = coherence_gate(None, None, generate_fn=_responder(looping))
    assert not report.passed


def test_gate_judges_the_answer_not_a_reasoning_preamble():
    """A thinking block should not be able to satisfy the length check alone."""
    answers = {
        k: "<think>" + ("pondering " * 40) + "</think>" + v
        for k, v in GOOD_ANSWERS.items()
    }
    report = coherence_gate(None, None, generate_fn=_responder(answers))
    assert report.passed, report.failures
    assert all("pondering" not in p["response"] for p in report.probes)


def test_custom_probes_are_honoured():
    probes = (("say something", 5),)
    report = coherence_gate(
        None, None, probes=probes, generate_fn=lambda p: "a long enough reply here"
    )
    assert len(report.probes) == 1
    assert report.passed


def test_foreign_threshold_is_adjustable():
    mixed = {k: "the user pregunta about a thing and then some more words here"
             for k in GOOD_ANSWERS}
    strict = coherence_gate(None, None, generate_fn=_responder(mixed),
                            max_foreign_frac=0.0)
    lenient = coherence_gate(None, None, generate_fn=_responder(mixed),
                             max_foreign_frac=0.9)
    assert not strict.passed
    assert any("language mixing" in f for f in lenient.failures) is False


# --- foreign-marker detection ------------------------------------------------
# The markers have to separate non-English output from English output. Two ways
# that can fail: an alternative that is also an English word, and a CJK
# alternative wrapped in \b, which can never match because Python treats
# ideographs as word characters.

ENGLISH_SENTENCES = [
    "The con artist was caught by the police last week.",
    "Los Angeles is the largest city in California by population.",
    "Die casting is a metal forming process used in manufacturing.",
    "Pour the water slowly into the bowl and stir until smooth.",
    "A perfectly ordinary sentence about baking bread at home.",
    "The parachute opened and the para-athlete landed safely.",
]


@pytest.mark.parametrize("text", ENGLISH_SENTENCES)
def test_ordinary_english_is_not_flagged_as_foreign(text):
    stats = analyse(text)
    assert stats["foreign_markers"] == 0, (
        f"{stats['foreign_markers']} marker(s) found in plain English: {text!r}"
    )


@pytest.mark.parametrize("text", [
    "我是学生，这是我的书。",
    "これは日本語の文章です。",
    "Это предложение на русском языке.",
])
def test_non_latin_script_is_detected(text):
    """A \\b-wrapped alternation cannot match these, so a script check is needed."""
    stats = analyse(text)
    assert stats["foreign_markers"] > 0, f"not detected as non-English: {text!r}"
    assert stats["foreign_frac"] > 0.5


def test_spanish_mixing_is_still_detected():
    stats = analyse("I need to entender que the usuario pregunta about this")
    assert stats["foreign_markers"] >= 3


# --- a reasoning model that answers nothing ----------------------------------

def test_an_empty_answer_after_a_reasoning_block_fails():
    """The gate must judge the answer, and an empty answer is a failure.

    Falling back to the raw string here would hand every check the reasoning
    text and pass a model that produced no answer at all.
    """
    def gen(prompt):
        return "<think>Let me work through this carefully step by step.</think>"

    report = coherence_gate(None, None, generate_fn=gen)
    assert not report.passed
    assert any("too short" in f for f in report.failures)


def test_a_real_answer_after_a_reasoning_block_passes():
    def gen(prompt):
        return (
            "<think>internal deliberation that should not be judged</think>"
            + GOOD_ANSWERS.get(prompt, "A clear and sufficiently long answer to the question asked.")
        )

    report = coherence_gate(None, None, generate_fn=gen)
    assert report.passed, report.failures


# --- the repeated-opening check ----------------------------------------------

def test_repeated_opening_is_found_despite_internal_whitespace():
    """The chunk is normalised by split(), so the haystack must be too.

    A newline inside the first ten words makes the normalised chunk absent from
    the raw text, so searching the raw text finds nothing however many times the
    opening repeats.
    """
    # The newline sits inside the first ten words, which is what breaks a raw
    # substring search.
    opening = "this is\nthe repeated opening phrase that appears twice in full"
    text = opening + " " + opening + " " + " ".join(f"w{i}" for i in range(12))
    assert find_loop(text) == "repeated opening phrase"
