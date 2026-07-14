# API Reference

Public helpers exposed by `ollama_classifier_gui.utils`.

## Constants

### `BACKEND_TYPES`

```python
BACKEND_TYPES = ["ollama", "vllm", "sglang", "llamacpp"]
```

The list of supported inference backend identifiers.

### `DEFAULT_ENDPOINTS`

```python
DEFAULT_ENDPOINTS = {
    "ollama":   "http://localhost:11434",
    "vllm":     "http://localhost:8000/v1",
    "sglang":   "http://localhost:30000/v1",
    "llamacpp": "http://localhost:8080/v1",
}
```

Mapping of backend type → default endpoint URL.

## Functions

### `check_system_dependencies`

```python
def check_system_dependencies() -> list[str]
```

Returns a list of missing system binaries/libraries on Linux.

Returns an empty list on non-Linux platforms or when everything is present.

### `config_path`

```python
def config_path() -> Path
```

Returns the full path to the JSON config file for the current OS.

### `load_config`

```python
def load_config() -> dict[str, str]
```

Loads the configuration from disk, falling back to defaults if the file is
missing or corrupt.

### `save_config`

```python
def save_config(data: dict[str, str]) -> None
```

Persists the given configuration dictionary to the config file as JSON.

### `app_version`

```python
def app_version() -> str
```

Returns the installed application version (read from package metadata via
`importlib.metadata`). Falls back to `"0.0.0"` if unavailable.

### `app_repository_url`

```python
def app_repository_url() -> str
```

Returns the GitHub repository URL from package metadata, falling back to the
Homepage URL if the Repository entry is absent.

### `show_missing_dependencies`

```python
async def show_missing_dependencies(page, missing: list[str])
```

Displays a modal Flet dialog listing missing system requirements.
No-op when `missing` is empty.
