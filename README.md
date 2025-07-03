#  - TOXIC PROMPT Classification

This project focuses on building a transformer-based classification system to detect and flag the **toxic prompts** during customer engagements.

---

## - Objective

To fine-tune a transformer model that can accurately classify prompts as **safe** or **unsafe**, enabling effective **prompt guardrails** in GENAI systems.

---

## - Explanation

All the explanation about the setup, dataset used, model and metrics are clearly given in the notebook.
NOTE: While uploading to github if this error is found "Invalid Notebook"(Attached the faulty notebook as well.) please run these.
This removes any metadata, widgets etc. Better used in development stage only.

```Colab cell
from google.colab import files
uploaded = files.upload()
```

```Colab cell
import nbformat

filename = "GENAI_Assessment(Prathama).ipynb"  # change if needed
notebook = nbformat.read(filename, as_version=4)

# Remove widgets metadata if it exists
if "widgets" in notebook.metadata:
    del notebook.metadata["widgets"]
    print("Removed metadata.widgets")

# Save the cleaned notebook
nbformat.write(notebook, filename)
print(f"Cleaned and saved: {filename}")
```

```Colab cell
files.download(filename)
```


## - Summary and Discussion

### 1. Results

- Accuracy: 96.8%
- Precision : 0.87
- Recall : 0.8
- F1-score: 0.84

### 2. Observations and challenges

- Flags the unsafe comments with high accuracy and precision.
- However low recall indicates that a few unsafe comments are missed which might be a tradeoff because of high optimization of other metrics.

### 3. Improvements

- Training for more epochs(Currently have done it for 1)
- Better hyperparams optimization with respect to other metrics.
- Changing max_length for better tokenization.

### 4. Real-world Integration strategies

- In a real-time system, the model should act like an API microservice that gets user prompt before they are sent to the Generative LLM. Each prompt that is intercepted will go through the classifier and if when flagged, the response can be handled accordingly.


