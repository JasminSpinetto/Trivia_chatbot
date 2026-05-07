# Natural Language Processing Project - 2025/2026

### Course Information
- **University:** Politecnico di Milano
- **Course:** Natural Language Processing (NLP)
- **Academic Year:** 2025/2026
- **Professor:** Mark Carman James

### Team Members
- Jasmine Spinetto
- Olga Eskarous
- Luca Colangelo
- Hossam Eldin Ahmed Hussien
- Andrea Mikhaiel

## Project Overview
This repository contains the development and implementation of the NLP project for the A.Y. 2025/2026. The project focuses on the development of an autonomous chatbot to compete in the 'Who Wants to be a PoliMillionaire?' online quiz. The system leverages locally-hosted models to provide accurate answers within a 30-second timeout, exploring advanced NLP techniques such as RAG and Agentic AI.

![intro](media/teaser.png)

For a detailed description of the project goals, constraints, and evaluation criteria, please refer to the `assignment.pdf` file available in the root directory.

---

## Installation and Environment Setup
To ensure reproducibility, we recommend using a virtual environment. Follow these steps to set up the workspace:

### Clone the Repository
```bash
git clone https://github.com/JasminSpinetto/NLP_challenge
cd NLP_challenge
```

### Create a Conda virtual environment
```bash
conda env create -f environment.yml
conda activate nlp_project
```

### Hardware-Specific PyTorch Installation
> Run **only one** of the following sections depending on your hardware. These commands install PyTorch and related CUDA/compute libraries **after** activating the conda environment.
#### A. Users with NVIDIA GPU
 ```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
> CUDA 11.8 is compatible with the vast majority of NVIDIA GPUs. If installation fails, check your driver version with `nvidia-smi` and visit the [official PyTorch page](https://pytorch.org/get-started/locally/) for alternatives.
#### B. Users without a GPU (CPU only)
 ```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
> Running deep learning models on CPU is significantly slower and may be impractical for the larger models used in this project. If you do not have a compatible NVIDIA GPU, we strongly recommend using a cloud-based GPU environment.

### Environment Variables

To interact with the online quiz platform, rename the `.env.example` file in the root directory to `.env`. Then open `.env` and set your values:

```env
API_URL = 
USERNAME =
PASSWORD =
HF_TOKEN = 
```

## How to Run

The main entry point is `main.py`. Models are configured via YAML files in the `config/` folder and selected with `--config`.

### Basic usage

```bash
python main.py --config [config_file]
```

### Full benchmark — all competitions, multiple games, save results

```bash
python main.py --config [config_file] --test_all --multiplicity [N] --output_csv
```

Plays `N` games per competition and appends results to `results/results.csv`.

### Math-only benchmark

```bash
python main.py --config [config_file] --math --multiplicity [N] --output_csv
```

Runs `N` games on the Maths competition only. Useful for evaluating math-specific models.

### Offline mode — run against a local question dataset

Set `ONLINE = False` in `main.py`, then:

```bash
python main.py --config [config_file] --dataset data/math_questions.json --output_csv
```

Runs the model against the fixed JSON dataset (no server connection needed). All questions in the file are answered exactly once and accuracy is reported. The dataset in `data/math_questions.json` contains 29 curated math questions with verified correct answers. `--math` and `--multiplicity` are not needed in this mode.

### All flags

| Flag | Description |
|------|-------------|
| `--config` | YAML config filename (resolved from `config/`) or full path |
| `--test_all` | Run on all competitions (online mode) |
| `--math` | Run on the math competition only (online mode) |
| `--multiplicity N` | Number of games per competition, or runs per question in offline mode (default: 1) |
| `--verbose` | Print question-by-question logs |
| `--output_csv` | Append session results to `results/results.csv` |
| `--dataset` | Path to a JSON question file for offline mode |

Note: only the math category has a dedicated flag since it's the trickiest category to handle, so testing on it specifically is encouraged.

### Available configs

| Config file | Model | Quant | RAG | Code | Notes |
|-------------|-------|:-----:|:---:|:----:|-------|
| `random.yaml` | Random baseline | — | — | — | Uniformly random answer, lower bound |
| **Meta LLaMA** |||||
| `llama-1b.yaml` | LLaMA 3.2 1B Instruct | — | — | — | Smallest, requires re-download |
| `llama-3b.yaml` | LLaMA 3.2 3B Instruct | — | — | — | Requires re-download |
| `llama-3b-wiki.yaml` | LLaMA 3.2 3B Instruct | — | ✓ | — | Wikipedia → DuckDuckGo retrieval |
| `llama-8b.yaml` | LLaMA 3.1 8B Instruct | 4-bit | — | — | |
| `llama-8b-code.yaml` | LLaMA 3.1 8B Instruct | 4-bit | — | ✓ | Agentic AI code executor |
| `llama-8b-wiki.yaml` | LLaMA 3.1 8B Instruct | 4-bit | ✓ | — | |
| `llama-8b-wiki-code.yaml` | LLaMA 3.1 8B Instruct | 4-bit | ✓ | ✓ | RAG + code executor |
| `llama-8b-router.yaml` | LLaMA 3.1 8B Instruct | 4-bit | ✓ | ✓ | Keyword router selects code vs text path per question |
| **Qwen 2.5 Math** |||||
| `qwen-7b.yaml` | Qwen 2.5 Math 7B Instruct | 4-bit | — | — | Math-focused, chain-of-thought |
| `qwen-7b-code.yaml` | Qwen 2.5 Math 7B Instruct | 4-bit | — | ✓ | Code executor |
| `qwen-7b-wiki.yaml` | Qwen 2.5 Math 7B Instruct | 4-bit | ✓ | — | RAG retrieval |
| `qwen-7b-wiki-code.yaml` | Qwen 2.5 Math 7B Instruct | 4-bit | ✓ | ✓ | RAG + code executor |
| **DeepSeek R1** |||||
| `deepseek-r1-7b-wiki.yaml` | DeepSeek-R1-Distill 7B | 4-bit | ✓ | — | Long reasoning chains; experimental, slow |
| `deepseek-r1-7b-wiki-code.yaml` | DeepSeek-R1-Distill 7B | 4-bit | ✓ | ✓ | Long reasoning chains + code; experimental, slow |
| **Microsoft Phi-4-mini** |||||
| `phi-4-mini.yaml` | Phi-4-mini-instruct (3.8B) | — | — | — | Fast, no quantization |
| `phi-4-mini-code.yaml` | Phi-4-mini-instruct (3.8B) | — | — | ✓ | Code executor |
| `phi-4-mini-wiki.yaml` | Phi-4-mini-instruct (3.8B) | — | ✓ | — | RAG retrieval |
| `phi-4-mini-wiki-code.yaml` | Phi-4-mini-instruct (3.8B) | — | ✓ | ✓ | **Best single model (50.8% on 63Q)** |
| `phi-4-mini-wiki-code-4bit.yaml` | Phi-4-mini-instruct (3.8B) | 4-bit | ✓ | ✓ | Colab T4 compatible version of the above |
| `phi-4-mini-router.yaml` | Phi-4-mini-instruct (3.8B) | — | ✓ | ✓ | Keyword router selects code vs text path per question |
| **Mistral / Mathstral** |||||
| `mathstral-7b.yaml` | Mathstral-7B-v0.1 | 4-bit | — | — | Math-specialized Mistral fine-tune; baseline |
| `mathstral-7b-wiki-code.yaml` | Mathstral-7B-v0.1 | 4-bit | ✓ | ✓ | RAG + code executor |
| **Google Gemma 3** |||||
| `gemma-3-12b.yaml` | Gemma-3-12B-IT | 4-bit | — | — | Requires HF license acceptance |
| `gemma-3-12b-wiki-code.yaml` | Gemma-3-12B-IT | 4-bit | ✓ | ✓ | RAG + code executor; requires HF license acceptance |

### Adding a new experiment

Create a new YAML file in `config/` — no code changes needed. All model parameters (`model_name`, `temperature`, `max_new_tokens`, `system_prompt`, `quantization`, `use_retrieval`, etc.) are defined in the YAML. See existing configs for reference.

---

## RAG Models (Wikipedia + DuckDuckGo)

The `llama-3b-wiki` and `llama-8b-wiki` configs enable retrieval-augmented generation. Before each question, the model searches Wikipedia (with DuckDuckGo as fallback) and injects the retrieved context into the prompt.

### Running a RAG model

```bash
# 3B model with retrieval
python main.py --config llama-3b-wiki.yaml

# 8B model (4-bit) with retrieval
python main.py --config llama-8b-wiki.yaml
```

### Enabling any model with retrieval

Add `use_retrieval: true` to any existing config — no code changes needed:

```yaml
model_name: "meta-llama/Llama-3.2-3B-Instruct"
use_retrieval: true
max_new_tokens: 20
```

### Debug logging

Pass `--debug` to write a full trace of every question to a timestamped log file in `logs/`:

```bash
python main.py --config llama-8b-wiki.yaml --debug
```

Each entry logs the question, retrieved context, full prompt, raw model response, and final answer:

```
============================================================
QUESTION : What term describes the body of Roman citizens?
CONTEXT  : The Roman people was the body of Roman citizens...
PROMPT   :
[system] You are a quiz contestant...
[user] Context from web search: ...
RESPONSE : '0'
ANSWER   : 0
============================================================
```

### Viewing the log in Google Colab

```python
import glob

log_files = sorted(glob.glob("logs/*.log"))
with open(log_files[-1]) as f:
    print(f.read())
```

> **Note:** The quiz platform will be taken offline at the end of the course. Once unavailable, all model testing must be performed using offline datasets. In `main.py` change `ONLINE` variable to False.

## Acknowledgments
We would like to acknowledge the course Teaching Assistants for developing and providing the utility modules located in `utils/`. These components are essential for the client configuration and the seamless interaction with the online quiz platform.