# capstone
fine-tuning a multimodal model to detect pneumonia in chest x-rays using QLoRA 4-bit quantization

## Contents

| Path | What it is |
|---|---|
| `medgemma_pneumonia_fine_tune.ipynb` | QLoRA fine-tuning and evaluation of MedGemma 4B on [hf-vision/chest-xray-pneumonia](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia) |
| `xray_app/` | Gradio study app: upload a chest X-ray, see the fine-tuned model's reading, and ask follow-up questions, with a safety harness against diagnostic use. See [`xray_app/README.md`](xray_app/README.md) to run it. |

Fine-tuned adapter: [yhussain/medgemma-4b-it-sft-lora-crc100k](https://huggingface.co/yhussain/medgemma-4b-it-sft-lora-crc100k)

This project is an educational tool for medical students. It is not a medical device and cannot diagnose anyone.
