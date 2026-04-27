# LLM Classifier GUI

A desktop GUI application for text classification using LLMs. Supports **multiple inference backends**: Ollama, vLLM, SGLang, and llama.cpp.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Built with [Flet](https://flet.dev/) and powered by [ollama-classifier](https://github.com/paluigi/ollama-classifier).

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

## Prerequisites

1. **Python 3.11+**
2. At least one inference backend:
   - [Ollama](https://ollama.com/download) (local)
   - [vLLM](https://docs.vllm.ai/) server
   - [SGLang](https://sglang.ai/) server
   - [llama.cpp server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)

### Linux system dependencies

| Dependency | Purpose | Install (Debian/Ubuntu) |
|---|---|---|
| `zenity` | File picker dialogs | `sudo apt install zenity` |
| `libsecret-1-0` | Secure storage backend | `sudo apt install libsecret-1-0` |
| `gnome-keyring` or `kwalletmanager` or `secret-service` | Keyring service | `sudo apt install gnome-keyring` |

## Installation

```bash
# Via uvx (recommended — no installation needed)
uvx ollama-classifier-gui

# Or via pip
pip install ollama-classifier-gui
ollama-classifier-gui
```

## Usage

### 1. Settings tab
Configure your inference backend:
- Select the **backend type** (Ollama, vLLM, SGLang, llama.cpp)
- Set the **endpoint URL** (auto-fills with defaults)
- Enter the **model name** (or use "Test Connection" to auto-detect available models)
- Optionally set an **API key** for authenticated remote servers
- Toggle **dark/light mode**
- Click **Save Settings**

### 2. Data Input tab
- Click **Select File** to load a CSV or Excel file
- For Excel files, select the **sheet** from the dropdown
- Select the **text column** that contains the text to classify

### 3. Schema tab
Define your classification labels:
- **Manual Labels**: click "Add Label" to add labels with optional descriptions
- **Load from File**: select a CSV/Excel file containing labels (and optionally descriptions)

Then choose:
- **Classification Method**: Classify (fast) or Score (accurate probabilities)
- **Output Format**: Top label only or All labels (each as a column)
- Optionally override the **system prompt**

### 4. Results tab
- Click **Run Classification** to start
- Monitor progress in real-time
- Click **Save Results** to export to Excel (original data + classification columns)

## Configuration

Settings are stored in a JSON config file:

| OS | Path |
|---|---|
| Linux | `$XDG_CONFIG_HOME/ollama-classifier-gui/config.json` (fallback `~/.config`) |
| macOS | `~/Library/Application Support/ollama-classifier-gui/config.json` |
| Windows | `%APPDATA%\ollama-classifier-gui\config.json` |

API keys are stored separately using the OS native secure storage (Keychain, Credential Manager, libsecret).

## Default Endpoints

| Backend | Default URL |
|---|---|
| Ollama | `http://localhost:11434` |
| vLLM | `http://localhost:8000/v1` |
| SGLang | `http://localhost:30000/v1` |
| llama.cpp | `http://localhost:8080/v1` |

## License

MIT
