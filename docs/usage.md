# Usage

The application is organized into tabs. Work through them left-to-right.

## 1. Settings tab

Configure your inference backend:

- Select the **backend type** (Ollama, vLLM, SGLang, llama.cpp)
- Set the **endpoint URL** (auto-fills with defaults)
- Enter the **model name** (or use "Test Connection" to auto-detect available models)
- Optionally set an **API key** for authenticated remote servers
- Toggle **dark/light mode**
- Click **Save Settings**

## 2. Data Input tab

- Click **Select File** to load a CSV or Excel file
- For Excel files, select the **sheet** from the dropdown
- Select the **text column** that contains the text to classify

## 3. Schema tab

Define your classification labels:

- **Manual Labels**: click "Add Label" to add labels with optional descriptions
- **Load from File**: select a CSV/Excel file containing labels (and optionally descriptions)

Then choose:

- **Classification Method**: Classify (fast) or Score (accurate probabilities)
- **Output Format**: Top label only or All labels (each as a column)
- Optionally override the **system prompt**

### Classification methods

| Method | Description | Output |
|---|---|---|
| **Classify** | Single LLM call per text | Predicted label + confidence score |
| **Score** | One call per label | Softmax probability for every label |

## 4. Results tab

- Click **Run Classification** to start
- Monitor progress in real-time
- Click **Save Results** to export to Excel (original data + classification columns)

## 5. Info tab

- Shows the **application version** and a link to the **GitHub repository** (both read from the package metadata defined in `pyproject.toml`)
- Click "Open in Browser" to open the repository in your default browser
