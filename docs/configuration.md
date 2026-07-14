# Configuration

## Settings file

Settings are stored in a JSON config file:

| OS | Path |
|---|---|
| Linux | `$XDG_CONFIG_HOME/ollama-classifier-gui/config.json` (fallback `~/.config`) |
| macOS | `~/Library/Application Support/ollama-classifier-gui/config.json` |
| Windows | `%APPDATA%\ollama-classifier-gui\config.json` |

API keys are stored separately using the OS native secure storage (Keychain, Credential Manager, libsecret).

### Default config

```json
{
  "backend_type": "ollama",
  "endpoint": "http://localhost:11434",
  "model": "llama3.2",
  "theme": "system",
  "batch_size": "1",
  "output_format": "top_label"
}
```

## Default endpoints

| Backend | Default URL |
|---|---|
| Ollama | `http://localhost:11434` |
| vLLM | `http://localhost:8000/v1` |
| SGLang | `http://localhost:30000/v1` |
| llama.cpp | `http://localhost:8080/v1` |
