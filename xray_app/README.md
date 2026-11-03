# Chest X-Ray Study Tool

A Gradio web app around my fine-tuned MedGemma 4B pneumonia classifier. Users upload a
chest X-ray, see the model's reading, and ask follow-up questions about the same image.
It is a study aid for medical students, not a diagnostic tool.

## How it works

```
upload ─► acknowledgement checkbox ─► image gate ─► classifier ─► hedged result card
                                       │                              │
                       (color check + base model:              follow-up questions
                        "is this a chest X-ray?")                     │
                                                    input screening ─► model ─► output guard
```

| File | What it does |
|---|---|
| `app.py` | Gradio interface and the request flow |
| `model.py` | Loads `google/medgemma-4b-it` in 4-bit plus my LoRA adapter; runs the X-ray check, the classifier, and follow-up chat |
| `safety.py` | The safety harness (all rules and wording in one place) |
| `tests/` | Unit tests for the safety rules and an end-to-end test with a mock model |

### The three model calls

1. **Is it a chest X-ray?** A cheap color check rejects color photos, then the *base*
   MedGemma (adapter switched off) answers Yes/No. Uploads below 50% "Yes" are rejected.
2. **Classification.** The *fine-tuned adapter* gets the exact prompt from training
   (`medgemma_pneumonia_fine_tune.ipynb`), so it behaves the way it was evaluated. The app
   also reads the model's probabilities for "A" (no) vs. "B" (yes) to show a confidence,
   and labels anything under 75% as low confidence.
3. **Follow-up chat.** Same loaded model, same image, with a safety system prompt. By
   default the adapter is switched **off** for chat (see below).

### Safety harness

| Layer | What it catches | What happens |
|---|---|---|
| Acknowledgement | Nothing runs until the user ticks "this is an educational tool" | — |
| Image gate | Non-X-ray uploads | Rejected with an explanation |
| Result wording | Every classifier result | Hedged language, confidence, pediatric/two-label limits, "not a diagnosis" |
| Input screening | Emergency symptoms ("I can't breathe", "lips turning blue") | Fixed 911 message, **model not called** |
| | Self-harm language | Fixed 988 message, **model not called** |
| | Personal diagnosis ("do I have…", "my x-ray", "should I go to the ER") | Warning shown above the answer + extra instruction to the model |
| | Treatment requests ("what antibiotic should I take") | Warning + model told not to prescribe |
| System prompt | Every answer | Hedge, teach, never diagnose or prescribe, never say someone is fine |
| Output guard | "definitely shows", "you have pneumonia", "no need to see a doctor" | Softens wording and/or appends a safety note |
| | Doses in a personal or treatment context | Removed |

The screening is rule-based (regular expressions), which keeps it predictable and
testable but means it can miss unusual phrasings. The system prompt and output guard
are the backstop. This is worth mentioning in the paper's limitations.

## Running it

You need a CUDA GPU (a free Colab T4 works; the app uses float16 on GPUs without
bfloat16) and a Hugging Face token whose account has accepted the
[MedGemma terms](https://huggingface.co/google/medgemma-4b-it).

### In Google Colab

```python
# 1. Get the code
!git clone https://github.com/yumnahussain/pneumonia-xray-detection.git
%cd pneumonia-xray-detection/xray_app
!pip install -q -r requirements.txt

# 2. Log in (uses the HF_TOKEN secret, same as your training notebook)
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# 3. Launch with a public link
os.environ["SHARE"] = "1"
!python app.py
```

Open the `gradio.live` link it prints. The first analysis takes a minute or two while
the weights download.

### Locally or on another GPU machine

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...
python app.py              # opens on http://127.0.0.1:7860
```

### Without a GPU (interface only)

```bash
MOCK_MODEL=1 python app.py
```

Uses a fake model so you can click through the interface and test the safety messages.

### Tests

```bash
python -m pytest -q tests
```

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `ADAPTER_REVISION` | 1-epoch commit (`958e014…`) | Branch or commit hash of the adapter on the Hub |
| `FOLLOWUP_USE_ADAPTER` | `0` | `1` = answer follow-ups with the fine-tuned adapter on |
| `ADAPTER_ID` | `yhussain/medgemma-4b-it-sft-lora-crc100k` | Adapter repo |
| `BASE_MODEL_ID` | `google/medgemma-4b-it` | Base model |
| `LOAD_IN_4BIT` | `1` | Load the base model in 4-bit (about 6–7 GB of GPU memory with the adapter) |
| `MAX_FOLLOWUP_TOKENS` | `512` | Length limit for follow-up answers |
| `SHARE` | `0` | `1` = create a public Gradio link |
| `MOCK_MODEL` | `0` | `1` = fake model for UI testing |

### Which adapter version

The app uses the **1-epoch** adapter by default, from commit
`958e014533b1ea6fc6947e8fdd4325b6977f9e5a` of the adapter repo. The newest version on the Hub (`main`) is the later
**8-epoch** adapter, which scored lower (77.6% accuracy). To use it instead, set
`ADAPTER_REVISION=main`. The app shows the version in use under each result.

### Why follow-ups use the adapter switched off by default

The adapter was trained to output one line ("A: No…" / "B: Yes…"), and training also
updated the output layer (`lm_head`) and embeddings. After that, it tends to answer every
question with a label instead of an explanation. Follow-ups therefore go to the same
loaded model with the adapter switched off, which keeps MedGemma's ability to explain,
while the fine-tuned classifier's result is passed in as context. Set
`FOLLOWUP_USE_ADAPTER=1` to compare the two; that comparison could go in the paper.

## Privacy

Images stay in memory for the session and are not written to logs. Gradio keeps
uploaded files in a temporary cache, which this app clears after 30 to 60 minutes. A
public `share` link means anyone with the link can use the app, so don't share it
widely, and don't upload identifiable patient images.
