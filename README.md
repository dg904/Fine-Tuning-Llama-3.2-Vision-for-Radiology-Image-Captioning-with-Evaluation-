# Fine-Tuning Llama 3.2 Vision for Radiology Image Captioning (with Evaluation)

This notebook fine-tunes `unsloth/Llama-3.2-11B-Vision-Instruct` on the `unsloth/Radiology_mini` dataset using LoRA adapters, then evaluates the fine-tuned model against the base model on standard captioning metrics.

## What it does

1. **Loads** the Llama 3.2 11B Vision model in 4-bit quantization via [Unsloth](https://github.com/unslothai/unsloth) for fast, memory-efficient fine-tuning.
2. **Configures LoRA** adapters across vision, language, attention, and MLP layers.
3. **Loads and formats** the `Radiology_mini` dataset (image + caption pairs) into chat-style training examples.
4. **Runs inference before training** to capture a baseline caption on a sample radiograph.
5. **Fine-tunes** the model with `SFTTrainer` (TRL) using a vision-aware data collator, and reports GPU memory usage and training time.
6. **Runs inference after training** and streams the fine-tuned caption for qualitative comparison.
7. **Evaluates** the base model vs. the fine-tuned model on a held-out sample using BLEU, ROUGE-1/2/L, METEOR, and BERTScore F1, with a side-by-side comparison table and plots.
8. **Exports** the fine-tuned model to GGUF (4-bit quantized) for use with llama.cpp-compatible inference engines.

## Requirements

- A CUDA-capable GPU (the notebook was built for Colab; an A100/L4/T4-class GPU is recommended for the 11B model).
- Python 3.10+
- See `requirements.txt` for Python package dependencies.

## Usage

Open the notebook in Jupyter, Colab, or any environment with GPU access, and run cells top to bottom. Key parameters you may want to adjust:

- `N_EVAL_SAMPLES` — number of held-out samples used for evaluation (default: 25).
- LoRA rank `r` and target modules in the `FastVisionModel.get_peft_model` call.
- Training hyperparameters in the `SFTConfig` (batch size, learning rate, epochs/steps).

## Output

- A comparison table and plots of before/after fine-tuning metrics.
- A GGUF-exported model saved to `radiology_model_gguf/`.

## Notes

- This is intended as an educational / experimental fine-tuning example, not a validated clinical tool. Do not use outputs for real diagnostic decisions.
- Evaluation metrics (BLEU/ROUGE/METEOR/BERTScore) are text-similarity proxies and don't capture clinical correctness.
