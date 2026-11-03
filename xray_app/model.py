"""Loads MedGemma 4B with the fine-tuned LoRA adapter and runs the three model calls
the app needs:

1. is_chest_xray()  - base MedGemma (adapter off) checks the upload is a chest X-ray
2. classify()       - fine-tuned adapter answers the exact training question
3. chat()           - follow-up questions about the same image

Settings come from environment variables (see README):
  BASE_MODEL_ID          default google/medgemma-4b-it
  ADAPTER_ID             default yhussain/medgemma-4b-it-sft-lora-crc100k
  ADAPTER_REVISION       default main (branch, tag or commit hash on the Hub)
  FOLLOWUP_USE_ADAPTER   "1" to answer follow-ups with the adapter on (default "0")
  LOAD_IN_4BIT           "1" (default) to load the base model in 4-bit
  MOCK_MODEL             "1" to use a fake model (no GPU, for testing the UI)
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass

# Must match the prompt and answer strings used in training
# (medgemma_pneumonia_fine_tune.ipynb), or accuracy will drop.
PNEUMONIA_CLASSES = [
    "A: No — the image does not show pneumonia",
    "B: Yes — the image shows pneumonia",
]
CLASSIFY_PROMPT = (
    "Does this chest X-ray show signs of pneumonia? Answer Yes or No.\n"
    + "\n".join(PNEUMONIA_CLASSES)
)

XRAY_CHECK_PROMPT = (
    "Is this image a chest radiograph (a chest X-ray)? "
    "Answer with only one word: Yes or No."
)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    base_model_id: str = os.environ.get("BASE_MODEL_ID", "google/medgemma-4b-it")
    adapter_id: str = os.environ.get("ADAPTER_ID", "yhussain/medgemma-4b-it-sft-lora-crc100k")
    adapter_revision: str = os.environ.get("ADAPTER_REVISION", "main")
    followup_use_adapter: bool = _env_flag("FOLLOWUP_USE_ADAPTER", "0")
    load_in_4bit: bool = _env_flag("LOAD_IN_4BIT", "1")
    max_followup_tokens: int = int(os.environ.get("MAX_FOLLOWUP_TOKENS", "512"))


def parse_label(text: str) -> int:
    """Same rules as the notebook's postprocess(): 1 = pneumonia, 0 = normal, -1 = unknown."""
    t = (text or "").strip()
    for idx, label in enumerate(PNEUMONIA_CLASSES):
        alt = f"({label.replace(': ', ') ')}"
        if label in t or alt in t:
            return idx
    if t.startswith("B") or "Yes" in t or "yes" in t:
        return 1
    if t.startswith("A") or "No" in t or "no" in t:
        return 0
    return -1


class XrayModel:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No GPU found. MedGemma 4B needs a CUDA GPU (for example a Colab T4 or L4). "
                "Set MOCK_MODEL=1 to try the interface without a model."
            )

        # T4 GPUs don't support bfloat16; fall back to float16 there.
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        kwargs = dict(torch_dtype=self.dtype, device_map="auto", attn_implementation="eager")
        if self.config.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_storage=self.dtype,
            )

        self.processor = AutoProcessor.from_pretrained(self.config.base_model_id)
        self.processor.tokenizer.padding_side = "left"
        base = AutoModelForImageTextToText.from_pretrained(self.config.base_model_id, **kwargs)
        self.model = PeftModel.from_pretrained(
            base, self.config.adapter_id, revision=self.config.adapter_revision
        )
        self.model.eval()

        tok = self.processor.tokenizer
        self._tok_a = tok.encode("A", add_special_tokens=False)[0]
        self._tok_b = tok.encode("B", add_special_tokens=False)[0]
        self._tok_yes = tok.encode("Yes", add_special_tokens=False)[0]
        self._tok_no = tok.encode("No", add_special_tokens=False)[0]

    # ---------------------------------------------------------------- helpers
    def _adapter(self, on: bool):
        return nullcontext() if on else self.model.disable_adapter()

    def _inputs(self, messages, image):
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=[text], images=[[image.convert("RGB")]], return_tensors="pt"
        ).to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        return inputs

    def _generate(self, messages, image, max_new_tokens, adapter_on, scores=False):
        import torch

        inputs = self._inputs(messages, image)
        with self._adapter(adapter_on), torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=scores,
                return_dict_in_generate=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        new_tokens = out.sequences[0, inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        first_logits = out.scores[0][0].float() if scores else None
        return text, first_logits

    @staticmethod
    def _two_way_prob(logits, tok_pos: int, tok_neg: int) -> float:
        """Probability of tok_pos versus tok_neg at the first generated position."""
        import torch

        pair = torch.stack([logits[tok_neg], logits[tok_pos]])
        return float(torch.softmax(pair, dim=0)[1])

    @staticmethod
    def _user(image_first: bool, text: str):
        content = [{"type": "image"}] if image_first else []
        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    # ---------------------------------------------------------------- public API
    def is_chest_xray(self, image) -> float:
        """Base MedGemma's probability that the image is a chest X-ray."""
        _, logits = self._generate(
            [self._user(True, XRAY_CHECK_PROMPT)], image, 3, adapter_on=False, scores=True
        )
        return self._two_way_prob(logits, self._tok_yes, self._tok_no)

    def classify(self, image) -> tuple[int, float | None, str]:
        """Run the fine-tuned adapter on the exact training prompt.

        Returns (label, confidence in the chosen label, raw model text).
        """
        text, logits = self._generate(
            [self._user(True, CLASSIFY_PROMPT)], image, 20, adapter_on=True, scores=True
        )
        label = parse_label(text)
        if label == -1:
            return -1, None, text
        p_pneumonia = self._two_way_prob(logits, self._tok_b, self._tok_a)
        confidence = p_pneumonia if label == 1 else 1 - p_pneumonia
        return label, confidence, text

    def chat(self, image, system_prompt: str, history: list[dict], question: str) -> str:
        """Answer a follow-up question about the image.

        history: earlier turns as [{"role": "user"|"assistant", "content": str}, ...]
        The image and system prompt are attached to the first user turn, which is
        how Gemma 3's chat template expects them.
        """
        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        turns = list(history) + [{"role": "user", "content": question}]
        for i, turn in enumerate(turns):
            if turn["role"] == "user":
                messages.append(self._user(i == 0, turn["content"]))
            else:
                messages.append(
                    {"role": "assistant", "content": [{"type": "text", "text": turn["content"]}]}
                )
        if turns[0]["role"] != "user":  # always make sure the image is attached
            messages.insert(1, self._user(True, "Here is the chest X-ray."))
        text, _ = self._generate(
            messages,
            image,
            self.config.max_followup_tokens,
            adapter_on=self.config.followup_use_adapter,
        )
        return text


class MockModel:
    """Stand-in with the same interface, for testing the UI without a GPU."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.config.adapter_revision = f"{self.config.adapter_revision} (mock)"

    def is_chest_xray(self, image) -> float:
        from safety import color_saturation

        return 0.95 if color_saturation(image) < 0.05 else 0.05

    def classify(self, image):
        import numpy as np

        brightness = float(np.asarray(image.convert("L")).mean()) / 255
        label = 1 if brightness > 0.5 else 0
        return label, 0.62 + 0.3 * abs(brightness - 0.5), PNEUMONIA_CLASSES[label]

    def chat(self, image, system_prompt, history, question):
        if "certain" in question.lower():
            return "This X-ray definitely shows pneumonia, and you have pneumonia."
        return (
            "(Mock answer.) Lobar consolidation appears as a dense opacity, often with air "
            "bronchograms. A differential could include atelectasis, effusion, or a mass."
        )


def load_model():
    if _env_flag("MOCK_MODEL"):
        return MockModel()
    return XrayModel()
