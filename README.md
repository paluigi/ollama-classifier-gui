# Ollama Classifier GUI

A desktop GUI application for text classification using [Ollama](https://ollama.com/) and [ollama-classifier](https://pypi.org/project/ollama-classifier/).

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Load data** from CSV, Excel, or JSON files
- **Flexible classification schema**: define labels manually or load from a file
- **Two classification methods**:
  - *Classify*: single-call prediction with confidence score
  - *Score*: multi-call evaluation with softmax probabilities for all labels
- **Model selection**: connect to local or cloud Ollama instances, browse available models
- **Dark/Light mode** toggle
- **Save results** to CSV and JSON (with optional per-label probabilities)
- **Desktop shortcuts** via `pyshortcuts` (requires `uv`)
- **Secure storage** for API keys

## Prerequisites

1. **Python 3.11+**
2. **[Ollama](https://ollama.com/download)** installed and running
3. At least one model pulled (e.g., `ollama pull llama3.2`)

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

1. **Settings tab**: Configure your Ollama host URL, select a model, and optionally set an API key. Click *Test Connection* to verify and browse available models.
2. **Data Input tab**: Load a CSV, Excel, or JSON file containing the text you want to classify. Select the column that contains the text.
3. **Schema tab**: Define your classification labels — either manually (label name + optional description) or by loading a labels file. Choose the classification method.
4. **Results tab**: Click *Run Classification* to start. Monitor progress, then save results to CSV/JSON.

## Creating a Desktop Shortcut

The app includes a button in the Settings tab to create desktop and start menu shortcuts. This requires [uv](https://docs.astral.sh/uv/) to be installed. The shortcut runs `uvx ollama-classifier-gui`.

## Configuration

Settings are stored in a JSON config file:

| OS | Path |
|---|---|
| Linux | `$XDG_CONFIG_HOME/ollama-classifier-gui/config.json` (fallback `~/.config`) |
| macOS | `~/Library/Application Support/ollama-classifier-gui/config.json` |
| Windows | `%APPDATA%\ollama-classifier-gui\config.json` |

API keys are stored separately using the OS native secure storage (Keychain, Credential Manager, libsecret).

## License

MIT
