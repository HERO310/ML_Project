# ML_Project

# ICE-LoRA: Efficient Knowledge Editing using In-Context Learning with Low-Rank Adaptation

**Team:** Bhutoria

**Authors:** Sarthak Gopal, Aryan Verma

**Date:** September 4, 2025

---

## Project Overview

ICE-LoRA is a parameter-efficient implementation of In-Context Editing (ICE) that integrates the ICE loss function with LoRA (Low-Rank Adaptation) to enable fast, memory-efficient knowledge editing in transformer language models. Our approach reduces the number of trainable parameters by over 95% compared to full-layer fine-tuning while preserving the ability to perform accurate, local, and portable knowledge edits.

## Motivation

Maintaining up-to-date and accurate factual knowledge in large language models without full retraining is a critical research problem. Existing methods either suffer from catastrophic forgetting or are computationally expensive. ICE introduces a distribution-based loss that mitigates brittleness caused by deterministic targets, and LoRA provides a compact, efficient mechanism to adapt model parameters. ICE-LoRA marries these techniques to achieve high-quality edits on consumer hardware.

## Key Features

* **ICE loss integration:** Implements the ICE objective that minimizes the KL divergence between context-conditioned and context-free model outputs.
* **LoRA adaptation:** Applies low-rank updates to transformer layers (rank-16 recommended) to drastically reduce trainable parameter count.
* **Combined objective:** Training uses a combined loss `L_total = L_FT + \lambda * L_ICE` for stable and effective edits.
* **Benchmarks:** Evaluation on ZsRE and WikiData Recent for edit success, locality, portability, and fluency.

---

## Repository Structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── zsre/
│   └── wikidata_recent/
├── src/
│   ├── icelora/
│   │   ├── icelora_editor.py       # core implementation (ICELoRAEditor)
│   │   ├── losses.py               # ICE loss and combined objectives
│   │   └── lora_utils.py           # LoRA integration helpers
│   ├── data_utils/
│   │   ├── dataset.py              # custom dataset handlers & preprocessing
│   │   └── templates.py            # context/template generation
│   ├── train.py                    # training entrypoint
│   └── eval.py                     # evaluation scripts & metrics
├── experiments/
│   └── logs/                       # training logs & checkpoints
└── reports/
    └── evaluation_report.pdf
```

---



## Quickstart: Training

1. Prepare dataset folders under `data/` (see *Datasets* below).
2. Edit `configs/train_config.yaml` (learning rate, batch size, LoRA rank, \lambda for L_ICE).
3. Run training:

```bash
python src/train.py --config configs/train_config.yaml --output_dir experiments/run01
```

Training produces checkpoints and logs in `experiments/run01/`.

---

## Quickstart: Evaluation

Evaluate a checkpoint on ZsRE and WikiData Recent:

```bash
python src/eval.py --checkpoint experiments/run01/checkpoint.pt --dataset zsre --output experiments/run01/eval_zsre.json
```

Evaluation metrics reported:

* **Edit Success** (accuracy on edited facts)
* **Locality Score** (preservation of unrelated facts)
* **Portability Score** (ability to apply edits in new contexts)
* **Fluency** (perplexity or human-evaluated naturalness)

---

## Datasets

We use the following datasets for training and evaluation:

* **ZsRE** — Zero-shot relation extraction benchmark (~1,037 pairs)
* **WikiData Recent** — Recent factual claims from Wikipedia (~1,200 samples)

Preprocessing steps (implemented in `src/data_utils/dataset.py`):

* Tokenization using Hugging Face tokenizers
* Template-based context generation for ICE contexts
* Creation of related/unrelated queries for locality testing

---

## Model & Hyperparameters

* **Base model:** GPT-2 medium (configurable)
* **LoRA rank:** 16 (default)
* **Trainable params:** ~1–2M (versus ~117M for full layer updates in baseline)
* **Loss:** `L_total = L_FT + \lambda * L_ICE` (choose \lambda via validation)

Suggested starting hyperparameters (example):

```yaml
learning_rate: 5e-5
batch_size: 16
num_epochs: 5
lora_rank: 16
lambda_ice: 0.5
```

---

## Implementation Notes

* The `ICELoRAEditor` class implements forward passes to compute both cross-entropy and ICE KL-divergence terms. It handles context generation and LoRA adapter updates.
* LoRA updates are applied to targeted transformer projection matrices (e.g., query/value/feedforward) using `src/icelora/lora_utils.py`.
* To ensure correctness of mirrored or synthetic contexts, we provide deterministic template functions and seeding in `src/data_utils/templates.py`.

---

## Results & Expected Outcomes

We aim for the following target metrics (on evaluation sets):

* **Edit Success:** > 0.80
* **Locality:** > 0.90
* **Portability:** > 0.70
* **Fluency:** > 0.75

A full evaluation report and comparative study with traditional fine-tuning will be included in `reports/`.

---

## Reproducibility

* All experiments are run with fixed random seeds (configurable in `configs/`).
* Environment and package versions are captured in `requirements.txt` and `environment.yml` (if using conda).

---

## Contributing

Contributions are welcome. Please open issues or pull requests for bug reports, feature requests, or code improvements. Follow the code style guidelines in `.github/CONTRIBUTING.md`.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## References

* Mitchell, E., et al. “In-Context Editing: Learning Knowledge from Self-Induced Distributions.”
* Hu, E. J., et al. “LoRA: Low-Rank Adaptation of Large Language Models.”
* Meng, K., et al. “MEMIT: Mass-Editing Memory in a Transformer.”
* CounterFact and ZsRE datasets (Knowledge Editing Benchmarks)

---



*Generated from the uploaded Statement of Purpose (CS550: Machine Learning Project).*
