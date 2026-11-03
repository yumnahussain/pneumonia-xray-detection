import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import safety
from safety import Intent, guard_output, screen_message
from model import parse_label

# ---------------------------------------------------------------- input screening

EDUCATIONAL = [
    "What features would support pneumonia on this image?",
    "What else could look like this? Give me a short differential.",
    "What is an air bronchogram?",
    "What causes chest pain in lobar pneumonia?",
    "How is community-acquired pneumonia usually treated in general?",
    "Why might the model be wrong on this image?",
    "Is the silhouette sign present here?",
]

PERSONAL = [
    "Do I have pneumonia?",
    "This is my x-ray, is it bad?",
    "Is this serious?",
    "Should I go to the ER?",
    "Does my son have pneumonia?",
    "My doctor said it was fine, do you agree?",
    "What's wrong with me?",
    "I have a cough and fever, is this pneumonia?",
    "Am I going to be ok?",
]

TREATMENT = [
    "What antibiotic should I take?",
    "How many mg of amoxicillin should I give my daughter?",
    "How do I treat my pneumonia at home?",
]

EMERGENCY = [
    "I can't breathe and my chest hurts",
    "My baby's lips are turning blue",
    "My dad passed out and has a fever",
    "She is struggling to breathe right now",
]


@pytest.mark.parametrize("text", EDUCATIONAL)
def test_educational_passes(text):
    assert screen_message(text).intent == Intent.EDUCATIONAL


@pytest.mark.parametrize("text", PERSONAL)
def test_personal_diagnosis_flagged(text):
    s = screen_message(text)
    assert s.intent == Intent.PERSONAL_DIAGNOSIS
    assert not s.block_model and s.prefix and s.extra_instruction


@pytest.mark.parametrize("text", TREATMENT)
def test_treatment_flagged(text):
    assert screen_message(text).intent == Intent.TREATMENT


@pytest.mark.parametrize("text", EMERGENCY)
def test_emergency_blocks_model(text):
    s = screen_message(text)
    assert s.intent == Intent.EMERGENCY and s.block_model and "911" in s.prefix


def test_self_harm():
    s = screen_message("I want to kill myself")
    assert s.intent == Intent.SELF_HARM and s.block_model and "988" in s.prefix


# ---------------------------------------------------------------- output guard


def test_softens_certainty():
    r = guard_output("The X-ray definitely shows pneumonia in the right lower lobe.")
    assert "definitely" not in r.text and "may suggest" in r.text
    assert "Safety note" in r.text


def test_flags_you_have():
    r = guard_output("Based on this, you have pneumonia.")
    assert "over-certain claim" in r.flags and "Safety note" in r.text


def test_flags_dismissive():
    r = guard_output("The lungs look clear, so you don't need to see a doctor.")
    assert "dismissive claim" in r.flags and "cannot rule anything out" in r.text


def test_removes_dose_in_personal_context():
    r = guard_output("Typically 500 mg three times a day.", Intent.TREATMENT)
    assert "500 mg" not in r.text and "removed dose" in r.flags


def test_keeps_dose_in_educational_context():
    r = guard_output("Doses above 4000 mg/day of acetaminophen are hepatotoxic.")
    assert "4000 mg" in r.text


def test_hedged_text_untouched():
    text = "The opacity may suggest consolidation; atelectasis is also possible."
    r = guard_output(text)
    assert r.text == text and r.flags == []


# ---------------------------------------------------------------- helpers


def test_parse_label_matches_notebook():
    assert parse_label("B: Yes — the image shows pneumonia") == 1
    assert parse_label("A: No — the image does not show pneumonia") == 0
    assert parse_label("(B) Yes — the image shows pneumonia") == 1
    assert parse_label("???") == -1


def test_color_saturation():
    gray = Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8))
    red = Image.fromarray(np.dstack([np.full((64, 64), 220), np.zeros((64, 64)), np.zeros((64, 64))]).astype(np.uint8))
    assert safety.color_saturation(gray) < safety.MAX_SATURATION
    assert safety.color_saturation(red) > safety.MAX_SATURATION


def test_prediction_wording_is_hedged():
    text = safety.format_prediction(1, 0.6, "main")
    assert "not a diagnosis" in text and "low confidence" in text and "aged 1 to 5" in text
