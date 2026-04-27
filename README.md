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

The main entry point is `main.py`. It accepts a `--model` argument to select which model to use for answering questions.

### Usage

```bash
python main.py --model [model_name]
```
### Benchmarking usage
To benchmark models and save aggregate performance results, run:
```bash
python main.py --model [model_name] --test_all --multiplicity [number] --verbose
```
This command runs the selected model and plays ```[number]``` games for each competition, saving the results to a ```results.csv``` file.

### Available Models

| Model key | Description |
|-----------|-------------|
| `random` | Random baseline — picks a uniformly random answer (`models/random.py`) |
| `llama-1b` | Meta LLaMA 3.2 1B Instruct, locally hosted via HuggingFace (`models/LLM.py`) |
| `llama-3b` | Meta LLaMA 3.2 3B Instruct, locally hosted via HuggingFace (`models/LLM.py`) |
| `llama-8b` | Meta LLaMA 3.1 8B Instruct (4-bit), locally hosted via HuggingFace (`models/LLM.py`) |

> **Note:** The quiz platform will be taken offline at the end of the course. Once unavailable, all model testing must be performed using offline datasets. In `main.py` change `ONLINE` variable to False.

## Acknowledgments
We would like to acknowledge the course Teaching Assistants for developing and providing the utility modules located in `utils/`. These components are essential for the client configuration and the seamless interaction with the online quiz platform.