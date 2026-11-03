"""Safety harness for the chest X-ray study tool.

Three layers, all plain Python so they can be unit-tested without a GPU:

1. Input screening  - classify each follow-up message before it reaches the model
                      (emergency, self-harm, personal diagnosis, treatment request).
2. Prompting        - a system prompt that tells the model it is a teaching tool,
                      must hedge, and must not diagnose or prescribe.
3. Output guarding  - scan the model's reply for over-certain or unsafe phrasing,
                      soften it, and attach a correction note when needed.

Plus helpers for the image gate (is this plausibly a chest X-ray?) and for
wording the classifier result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Fixed text shown in the UI
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "**Educational tool only, not a medical device.** This app is a study aid for "
    "medical students. It cannot diagnose anyone. Its model was trained only on "
    "chest X-rays of children aged 1 to 5 from a single hospital, and it only "
    "distinguishes *normal* from *pneumonia*. If you have concerns about your own "
    "or someone else's health, talk to a doctor."
)

PRIVACY_NOTE = (
    "Uploaded images are kept only in memory for your session and are not saved "
    "or logged by this app. Remove names and other identifying details from images "
    "before uploading."
)

EMERGENCY_RESPONSE = (
    "**This sounds like it could be an emergency.** If you or someone with you is "
    "having trouble breathing, chest pain, blue or gray lips or skin, confusion, or "
    "has fainted, **call 911 (or your local emergency number) or go to the nearest "
    "emergency department now.** Don't wait for an app or an X-ray reading.\n\n"
    "This tool is for studying images and can't help in an emergency."
)

SELF_HARM_RESPONSE = (
    "I'm really sorry you're going through this. You don't have to handle it alone. "
    "In the US you can call or text **988** (Suicide & Crisis Lifeline) any time to "
    "talk with someone. If you're in immediate danger, call **911**. Outside the US, "
    "your local emergency number or a local crisis line can help."
)

PERSONAL_DIAGNOSIS_WARNING = (
    "> ⚠️ **I can't tell you whether you (or someone you know) has pneumonia or any "
    "other condition.** Only a doctor who can examine the patient, take a history, "
    "and order tests can make a diagnosis. This model was trained on pediatric "
    "images from one hospital and can be wrong, even when it sounds confident. "
    "Please share the X-ray and any symptoms with a doctor. If symptoms are severe "
    "or getting worse, seek care promptly.\n>\n"
    "> Below is general educational information only."
)

TREATMENT_WARNING = (
    "> ⚠️ **I can't recommend medications, doses, or treatment for a specific "
    "person.** Treatment decisions depend on the patient's age, history, allergies, "
    "and test results, and need a doctor or pharmacist. Below is general "
    "educational information only."
)

CERTAINTY_CORRECTION_NOTE = (
    "\n\n---\n*Safety note: parts of this answer may have sounded more certain than "
    "this model can be. An AI reading of an X-ray is a pattern-recognition "
    "estimate, not a diagnosis. Confirming pneumonia requires a clinician, the "
    "patient's history and exam, and often other tests.*"
)

DISMISSIVE_CORRECTION_NOTE = (
    "\n\n---\n*Safety note: this tool cannot rule anything out. A 'normal' model "
    "result does not mean a person is healthy or doesn't need care. Anyone with "
    "symptoms should be evaluated by a doctor.*"
)

# ---------------------------------------------------------------------------
# System prompt for follow-up questions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a teaching assistant inside a study tool that helps medical students learn to read chest X-rays. You are not a doctor and this is not a clinical tool.

A separate fine-tuned classifier has already looked at the uploaded image. Its result is given to you below. Use it as one piece of evidence that may be wrong, not as fact.

Follow these rules in every answer:
1. Never state a diagnosis as certain. Use hedged language such as "may suggest", "is consistent with", or "could be explained by". Explain that diagnosis needs a clinician, the patient's history and exam, and often other tests.
2. Teach. Explain the radiographic features involved (for example consolidation, air bronchograms, the silhouette sign, interstitial markings, effusion) and, when useful, give a short differential diagnosis.
3. If the user seems to be asking about their own image, a family member's image, or their own health, say clearly that you cannot diagnose them and encourage them to talk to a doctor. You may still explain general concepts.
4. Never recommend specific medications, doses, or a treatment plan for a person.
5. If anything suggests an emergency (trouble breathing, chest pain, blue lips, fainting), tell the user to seek emergency care immediately.
6. Remember the classifier's limits: it was trained only on chest X-rays of children aged 1 to 5 from one hospital and only distinguishes normal from pneumonia. Mention these limits when they matter, for example for adult images or other conditions.
7. Never say a person is fine or does not need to see a doctor.
8. Keep answers clear and concise, about 150 to 250 words unless the user asks for more detail."""

PERSONAL_CONTEXT_INSTRUCTION = (
    "The user appears to be asking about their own or a family member's health. "
    "Do not give a diagnosis or say whether they have a condition. Briefly explain "
    "the general concepts and encourage them to see a doctor."
)

TREATMENT_CONTEXT_INSTRUCTION = (
    "The user is asking about treatment or medication. Do not recommend any "
    "specific drug, dose, or plan for a person. You may describe in general terms "
    "how clinicians approach the condition, and tell them to ask a doctor or "
    "pharmacist."
)

# ---------------------------------------------------------------------------
# Layer 1: input screening
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    EDUCATIONAL = "educational"
    PERSONAL_DIAGNOSIS = "personal_diagnosis"
    TREATMENT = "treatment"
    EMERGENCY = "emergency"
    SELF_HARM = "self_harm"


@dataclass
class Screening:
    intent: Intent
    reasons: list[str] = field(default_factory=list)
    # True when the model should not be called at all
    block_model: bool = False
    # Shown to the user above the model's answer (or instead of it if blocked)
    prefix: str = ""
    # Extra instruction added to the model prompt for this turn
    extra_instruction: str = ""


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


# Words that tie a message to a real person rather than a study question.
_PERSONAL = _rx(
    r"\b(i|i'm|im|i've|ive|i'd|me|my|mine|myself)\b",
    r"\b(we|our|us)\b",
    r"\b(my|our)\s+(son|daughter|child|kid|baby|toddler|infant|mom|mother|dad|father|"
    r"wife|husband|partner|friend|grandma|grandmother|grandpa|grandfather|brother|"
    r"sister|parent|patient)\b",
)

_FAMILY = _rx(
    r"\b(my|our)\s+(son|daughter|child|kid|baby|toddler|infant|mom|mother|dad|father|"
    r"wife|husband|partner|friend|grandma|grandmother|grandpa|grandfather|brother|sister)\b"
)

_SELF_HARM = _rx(
    r"\b(kill|hurt|harm)\s+(myself|me)\b",
    r"\bsuicid",
    r"\bend\s+(my|it\s+all)\b",
    r"\bwant\s+to\s+die\b",
    r"\bdon'?t\s+want\s+to\s+(live|be\s+alive)\b",
)

_EMERGENCY_SYMPTOMS = _rx(
    r"\b(can'?t|cannot|can\s+not|unable\s+to|struggling\s+to|hard\s+to|trouble|difficulty)\s+"
    r"(breath|breathe|breathing|catch\s+(my|his|her|their)\s+breath)",
    r"\bgasping\b",
    r"\bchest\s+(pain|pressure|tightness)\b",
    r"\b(lips|face|skin|fingers|nails)\b.{0,20}\b(blue|gray|grey)\b",
    r"\b(blue|gray|grey)\b.{0,20}\b(lips|face|skin|fingers|nails)\b",
    r"\bcoughing\s+(up\s+)?blood\b",
    r"\b(passed\s+out|fainted|unconscious|unresponsive|won'?t\s+wake)\b",
    r"\b(seizure|seizing)\b",
)

# Signs the user wants a verdict on a real person.
_DIAGNOSIS_SEEKING = _rx(
    r"\b(do|does|did)\s+(i|he|she|they|my\s+\w+)\s+have\b",
    r"\b(am\s+i|is\s+(he|she|my\s+\w+)|are\s+(we|they))\s+(sick|ok|okay|fine|healthy|ill|dying|going\s+to\s+be)\b",
    r"\b(is|was)\s+(it|this|that)\s+(serious|bad|dangerous|cancer|pneumonia|covid|tb|tuberculosis|something\s+bad)\b",
    r"\bshould\s+(i|we|he|she|they)\s+(go|see|call|visit|worry|be\s+worried|be\s+concerned|get)\b",
    r"\b(diagnose|diagnosis)\s+(me|him|her|them|my)\b",
    r"\bwhat'?s\s+wrong\s+with\s+(me|him|her|them|my)\b",
    r"\b(my|our)\s+(x-?ray|xray|scan|radiograph|image|results?|report|lungs?|chest|cough|fever|symptoms?)\b",
    r"\b(my|the)\s+doctor\s+(said|says|told|thinks)\b",
    r"\b(i|he|she|they)\s+(have|has|had)\s+(a\s+)?(cough|fever|chills|shortness\s+of\s+breath|symptoms?)\b",
    r"\bfor\s+(me|him|her|them|my\s+\w+)\b",
)

_TREATMENT = _rx(
    r"\b(antibiotic|antibiotics|amoxicillin|azithromycin|augmentin|z-?pack|doxycycline|"
    r"levofloxacin|ceftriaxone|steroid|steroids|prednisone|ibuprofen|tylenol|"
    r"acetaminophen|paracetamol|inhaler|nebuli[sz]er)\b",
    r"\b(dose|dosage|dosing|how\s+many\s+mg|\d+\s?mg)\b",
    r"\b(medicine|medication|meds|prescription|prescribe)\b",
    r"\bhow\s+(do|should|can)\s+(i|we|you)\s+(treat|cure|get\s+rid\s+of)\b",
    r"\b(what|which)\s+(should|can)\s+(i|we|he|she|they)\s+(take|give|use)\b",
    r"\bhome\s+remed",
)


def screen_message(text: str) -> Screening:
    """Classify a user's follow-up message before it reaches the model."""
    t = text or ""
    personal = bool(_PERSONAL.search(t))

    if _SELF_HARM.search(t):
        return Screening(Intent.SELF_HARM, ["self-harm language"], True, SELF_HARM_RESPONSE)

    # Emergency symptoms only count when tied to a real person, so that a
    # student asking "what causes chest pain in pneumonia?" isn't blocked.
    if _EMERGENCY_SYMPTOMS.search(t) and (personal or re.search(r"\b(he|she|they)\b", t, re.I)):
        return Screening(Intent.EMERGENCY, ["urgent symptoms described"], True, EMERGENCY_RESPONSE)

    reasons = []
    if _TREATMENT.search(t) and (personal or _DIAGNOSIS_SEEKING.search(t)):
        reasons.append("treatment or medication request for a person")
        return Screening(
            Intent.TREATMENT, reasons, False, TREATMENT_WARNING, TREATMENT_CONTEXT_INSTRUCTION
        )

    if _DIAGNOSIS_SEEKING.search(t) or _FAMILY.search(t):
        reasons.append("asks about a real person's health")
        return Screening(
            Intent.PERSONAL_DIAGNOSIS,
            reasons,
            False,
            PERSONAL_DIAGNOSIS_WARNING,
            PERSONAL_CONTEXT_INSTRUCTION,
        )

    return Screening(Intent.EDUCATIONAL)


# ---------------------------------------------------------------------------
# Layer 3: output guarding
# ---------------------------------------------------------------------------

# Rewrites that keep the sentence grammatical while removing false certainty.
_SOFTEN = [
    (re.compile(r"\b(definitely|certainly|clearly|conclusively|undoubtedly|unquestionably)\s+"
                r"(shows|indicates|demonstrates|confirms|proves|reveals)\b", re.I), "may suggest"),
    (re.compile(r"\b(definitely|certainly|conclusively|undoubtedly)\s+(is|has)\b", re.I),
     lambda m: "may be" if m.group(2).lower() == "is" else "may have"),
    (re.compile(r"\b(this|which|that)\s+confirms\b", re.I), r"\1 may suggest"),
    (re.compile(r"\bconfirms\s+(the\s+)?(diagnosis|presence)\b", re.I), r"may suggest \1\2"),
    (re.compile(r"\bthe\s+diagnosis\s+is\b", re.I), "one possible explanation is"),
    (re.compile(r"\bthere\s+is\s+no\s+doubt\b", re.I), "it is possible"),
]

_OVERCERTAIN = _rx(
    r"\byou\s+(definitely\s+|certainly\s+|clearly\s+)?(have|are\s+suffering\s+from)\s+(pneumonia|an?\s+infection|[a-z]+itis)\b",
    r"\b(he|she|they|your\s+\w+)\s+(definitely\s+|certainly\s+)?(has|have)\s+(pneumonia|an?\s+infection)\b",
    r"\bi\s+(can\s+)?diagnose\b",
    r"\b(100|99)\s?%\s*(sure|certain|confident|accurate)\b",
    r"\bwithout\s+(a\s+)?doubt\b",
)

_DISMISSIVE = _rx(
    r"\b(no|don'?t|do\s+not)\s+need\s+(to\s+)?(see|visit|call)\s+(a\s+)?(doctor|physician|clinician)\b",
    r"\bno\s+need\s+(to|for)\s+(see|a)\s+(a\s+)?(doctor|physician)\b",
    r"\byou('re|\s+are)\s+(perfectly\s+)?(fine|healthy|ok|okay)\b",
    r"\bnothing\s+to\s+worry\s+about\b",
    r"\b(rules?|ruled)\s+out\s+pneumonia\b",
)

_DOSE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|milligrams?|ml|mL)(/kg)?\b")


@dataclass
class GuardResult:
    text: str
    flags: list[str]


def guard_output(text: str, intent: Intent = Intent.EDUCATIONAL) -> GuardResult:
    """Soften over-certain wording and attach correction notes when needed."""
    flags: list[str] = []
    out = text or ""

    for pattern, repl in _SOFTEN:
        new = pattern.sub(repl, out)
        if new != out:
            flags.append("softened certainty")
            out = new

    if _OVERCERTAIN.search(out):
        flags.append("over-certain claim")
    if _DISMISSIVE.search(out):
        flags.append("dismissive claim")

    # In a personal or treatment context, never pass through specific doses.
    if intent in (Intent.PERSONAL_DIAGNOSIS, Intent.TREATMENT) and _DOSE.search(out):
        out = _DOSE.sub("[dose removed: ask a doctor or pharmacist]", out)
        flags.append("removed dose")

    if "over-certain claim" in flags or "softened certainty" in flags:
        out += CERTAINTY_CORRECTION_NOTE
    if "dismissive claim" in flags:
        out += DISMISSIVE_CORRECTION_NOTE

    return GuardResult(out, sorted(set(flags)))


# ---------------------------------------------------------------------------
# Image gate helpers
# ---------------------------------------------------------------------------


def color_saturation(image) -> float:
    """Mean HSV saturation in [0, 1]. X-rays are grayscale, so this is near 0.

    Accepts a PIL image. Used as a cheap first filter before asking the model.
    """
    import numpy as np

    hsv = image.convert("RGB").resize((256, 256)).convert("HSV")
    return float(np.asarray(hsv)[..., 1].mean() / 255.0)


MAX_SATURATION = 0.15  # photos of X-rays on a screen may carry a slight tint
XRAY_PROB_THRESHOLD = 0.5

NOT_XRAY_MESSAGE = (
    "This doesn't look like a chest X-ray, so I didn't analyze it. This tool only "
    "accepts frontal chest radiographs. Please upload a chest X-ray image."
)

# ---------------------------------------------------------------------------
# Wording the classifier result
# ---------------------------------------------------------------------------

UNCERTAIN_BELOW = 0.75


def format_prediction(label: int, confidence: float | None, adapter_revision: str) -> str:
    """Turn the classifier output into a hedged, educational message.

    label: 1 = pneumonia, 0 = normal, -1 = unreadable model output.
    confidence: probability the model put on its chosen answer, or None.
    """
    if label == -1:
        head = (
            "**Model result: unclear.** The model did not give a readable answer for "
            "this image, so no reading is shown."
        )
    else:
        finding = (
            "features the model associates with **pneumonia**"
            if label == 1
            else "**no** features the model associates with pneumonia"
        )
        conf_txt = ""
        if confidence is not None:
            conf_txt = f" It placed **{confidence:.0%}** of its probability on this answer."
            if confidence < UNCERTAIN_BELOW:
                conf_txt += (
                    " That is **low confidence**, so treat this result as uncertain."
                )
        head = f"**Model result:** the image shows {finding}.{conf_txt}"

    body = (
        "\n\nThis is a pattern-recognition output from a study tool, **not a "
        "diagnosis**. Keep in mind:\n"
        "- The model was trained only on chest X-rays of **children aged 1 to 5** from a "
        "single hospital. Results on adult or other images are unreliable.\n"
        "- It only distinguishes *normal* from *pneumonia*. It cannot recognize other "
        "conditions such as effusion, cardiomegaly, nodules, or tuberculosis, and may "
        "force them into one of its two labels.\n"
        "- Model confidence is not the same as the probability that a patient has a disease.\n\n"
        "Try reading the image yourself first, then ask me follow-up questions, such "
        "as *\"What features would support pneumonia here?\"* or *\"What else could "
        "look like this?\"*"
    )
    footer = f"\n\n<sub>Adapter version: `{adapter_revision}`</sub>"
    return head + body + footer


def classifier_summary(label: int, confidence: float | None) -> str:
    """Short factual summary passed to the model as context for follow-ups."""
    names = {1: "pneumonia", 0: "normal (no pneumonia)", -1: "unclear (no readable answer)"}
    conf = f" with {confidence:.0%} model confidence" if confidence is not None else ""
    return f"Fine-tuned classifier result: {names[label]}{conf}."
