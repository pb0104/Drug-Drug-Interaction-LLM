
# Drug–Drug Interaction Prediction using Fine-Tuned Language Models

**Course Project — Duke MIDS | Intelligent Agents**  
**Team:** Aesha Gandhi, Pranshul Bhatnagar, Gaurav Law

---

## Overview

Drug–drug interactions (DDIs) are one of the most common and preventable sources of patient harm. Clinicians in high-pressure settings often rely on slow manual lookups that return raw tabular data — not actionable clinical guidance.

This project fine-tunes **Microsoft Phi-2** using **QLoRA** on the [DDInter dataset](https://ddinter.scbdd.com/download/) to build an LLM system that answers natural language questions about drug interactions, returning structured severity classifications and plain-language clinical explanations.

---

## Repository Structure

```
├── DrugInteractionsLLM.ipynb     # Data pipeline: loads DDInter, builds QA pairs
├── Phi_FinetuningLLM.ipynb       # Model fine-tuning with QLoRA on Phi-2
├── DDI_Scoring.ipynb             # Evaluation: BERTScore, ROUGE-L, Token-F1
├── data/
│   ├── ddinter_downloads_code_A.csv   # Alimentary tract & metabolism
│   ├── ddinter_downloads_code_B.csv   # Blood & blood-forming organs
│   ├── ddinter_downloads_code_D.csv   # Dermatologicals
│   ├── ddinter_downloads_code_H.csv   # Systemic hormonal preparations
│   ├── ddinter_downloads_code_L.csv   # Antineoplastic & immunomodulating agents
│   ├── ddinter_downloads_code_P.csv   # Antiparasitic products
│   ├── ddinter_downloads_code_R.csv   # Respiratory system
│   └── ddinter_downloads_code_V.csv   # Various
├── drugs.csv                     # Concatenated, processed dataset with QA pairs
├── results.csv                   # Base and fine-tuned model predictions
└── requirements.txt
```

---

## Dataset

**Source:** [DDInter](https://ddinter.scbdd.com/download/)  
**Size:** ~220,000 drug pair interaction records across 8 ATC pharmacological categories  

| Column | Description |
|--------|-------------|
| `Drug_A` | First drug name |
| `Drug_B` | Second drug name |
| `Level` | Interaction severity: Major, Moderate, Minor, or Unknown |
| `Category` | ATC pharmacological category label |

---

## Notebooks

### 1. `DrugInteractionsLLM.ipynb` — Data Pipeline

Loads all 8 DDInter CSV files, assigns category labels, concatenates into a single dataframe, and converts each record into an instruction-style QA pair using the following template:

**Prompt:**
```
### Instruct:
You are a drug interaction classifier. You will be given two drugs and a disease category.
Classify their interaction for that category, and provide a short, structured response.

Drug 1: <Drug_A>
Drug 2: <Drug_B>
Category: <Category>

Respond conversationally (1-3 sentences). If you cannot determine the interaction, say you need more data.

### Output
```

**Completion:**
```
<Drug_A> and <Drug_B> are reported to have a <level> interaction.
This interaction is classified under <category>.
<Severity explanation>
```

Severity explanations are hardcoded per level (Major / Moderate / Minor / Unknown) to ensure clinically consistent language in training targets. Output saved as `drugs.csv`.

---

### 2. `Phi_FinetuningLLM.ipynb` — Model Fine-Tuning

Fine-tunes `microsoft/phi-2` using QLoRA on the processed dataset. Runs on Google Colab with an NVIDIA A100 GPU (80GB VRAM).

**Fine-Tuning Configuration:**

| Setting | Value |
|---------|-------|
| Base Model | `microsoft/phi-2` (2.7B parameters) |
| Method | QLoRA (LoRA + 4-bit quantization) |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| Compute dtype | FP16 |
| LoRA Rank (r) | 32 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `dense` |
| Task Type | Causal LM |
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-4 |
| LR Scheduler | Cosine |
| Warmup Ratio | 0.03 |
| Max Sequence Length | 512 |
| Optimizer | Paged AdamW (8-bit) |
| Training Framework | HuggingFace Transformers + PEFT + TRL |

The prompt and completion are split at `### Output` so that loss is computed only on the completion tokens (prompt tokens masked with `-100`).

The fine-tuned model is saved to Google Drive and loaded for inference using `PeftModel.from_pretrained`.

---

### 3. `DDI_Scoring.ipynb` — Evaluation

Evaluates base Phi-2 vs. fine-tuned Phi-2 predictions against ground-truth answers using a composite scoring metric.

**Scoring Weights:**

| Metric | Weight | Description |
|--------|--------|-------------|
| BERTScore (F1) | 0.70 | Semantic similarity via `distilbert-base-uncased` |
| ROUGE-L | 0.10 | Longest common subsequence overlap |
| Token F1 | 0.20 | Token-level precision/recall overlap |

**Results:**

| | BERTScore | ROUGE-L + Token F1 | Composite |
|---|---|---|---|
| Baseline Phi-2 | 0.77 | 0.18 | 0.5981 |
| Fine-Tuned Phi-2 | 0.88 | 0.35 | **0.7192** |

---

## Setup & Usage

All notebooks are designed to run on **Google Colab** with Drive mounted.

### 1. Mount Drive and set project root

```python
from google.colab import drive
drive.mount('/content/drive')
project_root = '/content/drive/MyDrive/Intelligent-Agents-Project/code'
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or directly:

```bash
pip install bitsandbytes transformers peft accelerate datasets scipy einops evaluate trl rouge_score bert_score
```

### 3. Run notebooks in order

```
DrugInteractionsLLM.ipynb  →  Phi_FinetuningLLM.ipynb  →  DDI_Scoring.ipynb
```

---

## Example Output

| Question | Ground Truth | Base Model | Fine-Tuned Model |
|----------|-------------|------------|-----------------|
| How do Astemizole and Magnesium citrate interact? | Moderate | Moderate interaction for Minor diseases | Magnesium citrates and Astemizoles have a **Moderate** interaction for interaction involving blood and blood forming organ drugs diseases. |
| How do Etoposide and Dabrafenib interact? | Moderate | You should Consult Doctor | DabrafENIB and Etoposides have a **Moderate** interaction for interaction involving antiplasmal diseases. |

---

## References

- De Vito, A., Lehmann, J., & Gesese, G. F. (2025). Can large language models predict drug-drug interactions? *arXiv:2502.06890*. https://doi.org/10.48550/arXiv.2502.06890
- Huang, K., Xiao, C., Hoang, T., Glass, L., & Sun, J. (2020). BERTChem-DDI: Improving drug-drug interaction prediction with chemical language model. *arXiv:2011.02743*. https://doi.org/10.48550/arXiv.2011.02743
- Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. *EMNLP 2016*. https://doi.org/10.48550/arXiv.1606.05250

---

## Disclaimer

This tool is intended as a first-pass clinical decision-support aid only. It is not a substitute for expert clinical judgment and should not be used for direct patient care without further validation.
