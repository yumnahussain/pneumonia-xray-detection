"""End-to-end run of the app's handlers with the mock model (no GPU needed)."""

import os
import sys

import numpy as np
from PIL import Image

os.environ["MOCK_MODEL"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app  # noqa: E402

XRAY = Image.fromarray(np.random.default_rng(0).integers(90, 200, (256, 256), dtype=np.uint8)).convert("RGB")
PHOTO = Image.fromarray(np.dstack([np.full((256, 256), 30), np.full((256, 256), 160), np.full((256, 256), 60)]).astype(np.uint8))


def test_requires_acknowledgement():
    history, state, _ = app.analyze(XRAY, False, [])
    assert history == [] and state["summary"] is None


def test_rejects_non_xray():
    history, state, _ = app.analyze(PHOTO, True, [])
    assert "doesn't look like a chest X-ray" in history[0]["content"]
    assert state["summary"] is None


def test_full_flow():
    history, state, _ = app.analyze(XRAY, True, [])
    assert history[0]["content"].startswith("**Model result")
    assert state["summary"]

    history, _ = app.ask("What else could look like this?", history, XRAY, state)
    assert "Mock answer" in history[-1]["content"]

    history, _ = app.ask("Do I have pneumonia?", history, XRAY, state)
    assert "can't tell you" in history[-1]["content"]

    # Mock model returns an over-certain answer when asked to be certain
    history, _ = app.ask("Be certain: what is it?", history, XRAY, state)
    assert "definitely" not in history[-1]["content"]
    assert "Safety note" in history[-1]["content"]

    history, _ = app.ask("I can't breathe", history, XRAY, state)
    assert "911" in history[-1]["content"]


def test_question_before_analysis():
    history, _ = app.ask("What is this?", [], None, {"summary": None})
    assert "Analyze X-ray" in history[-1]["content"]


def test_alternate():
    turns = [
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "q1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
    ]
    assert [t["content"] for t in app._alternate(turns)] == ["q2", "a2"]
