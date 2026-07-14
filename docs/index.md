# Ollama Classifier GUI

A desktop GUI application for text classification using LLMs. Supports **multiple inference backends**: Ollama, vLLM, SGLang, and llama.cpp.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Built with [Flet](https://flet.dev/) and powered by [ollama-classifier](https://github.com/paluigi/ollama-classifier).

---

## Features

- **Multiple backends**: Ollama, vLLM, SGLang, llama.cpp (local or remote)
- **Load data** from CSV or Excel files
- **Flexible classification schema**: define labels manually or load from a CSV/Excel file
- **Two classification methods**:
    - *Classify*: single-call prediction with confidence score
    - *Score*: multi-call evaluation with softmax probabilities for all labels
- **Output format options**:
    - *Top label only*: prediction + confidence
    - *All labels*: each label becomes a column with its probability
- **Batch size control**: process multiple texts concurrently for speed
- **Save results** to Excel (merged with original data)
- **Dark/Light mode** toggle
- **Secure storage** for API keys

## Quick start

```bash
# Via uvx (recommended — no installation needed)
uvx ollama-classifier-gui

# Or via pip
pip install ollama-classifier-gui
ollama-classifier-gui
```

Head to the [Installation](installation.md) page for prerequisites and detailed setup,
or the [Usage](usage.md) page for a walkthrough of each tab.
