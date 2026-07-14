# Installation

## Prerequisites

1. **Python 3.11+**
2. At least one inference backend:
    - [Ollama](https://ollama.com/download) (local)
    - [vLLM](https://docs.vllm.ai/) server
    - [SGLang](https://sglang.ai/) server
    - [llama.cpp server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server)

## Linux system dependencies

| Dependency | Purpose | Install (Debian/Ubuntu) |
|---|---|---|
| `zenity` | File picker dialogs | `sudo apt install zenity` |
| `libsecret-1-0` | Secure storage backend | `sudo apt install libsecret-1-0` |
| `gnome-keyring` or `kwalletmanager` or `secret-service` | Keyring service | `sudo apt install gnome-keyring` |

The application checks for these on startup and warns you if any are missing.

## Install the application

### Via `uvx` (recommended — no installation needed)

```bash
uvx ollama-classifier-gui
```

### Via `pip`

```bash
pip install ollama-classifier-gui
ollama-classifier-gui
```

## Run from source

```bash
git clone https://github.com/paluigi/ollama-classifier-gui.git
cd ollama-classifier-gui
uv sync
uv run python -m ollama_classifier_gui.main
```
