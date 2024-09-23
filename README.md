# Beyond-ML-Task

## Installation and Setup

### Pre-requisites
- Python >= 3.9

### Create a virtual environment
To create a virtual environment, run the following command:
```
python -m venv .bynd
```

### Activate the virtual environment
Activate the virtual environment using the appropriate command for your operating system:

**For Windows:**
```
.bynd\Scripts\activate
```

**For macOS and Linux:**
```
source .bynd/bin/activate
```

### Install dependencies
Install the required dependencies using `pip`:
```
pip install -r requirements.txt
```

## Usage

### Preprocess the Data
To preprocess the data, you can use the `Iris` class from `src/preprocess.py`:
```python:src/preprocess.py
startLine: 1
endLine: 35
```

### Train the Model
To train the K-Nearest Neighbors (KNN) model, you can use the `KNN` class from `src/KNN.py`:
```python:src/KNN.py
startLine: 1
endLine: 11
```

### Example
An example of loading the Iris dataset and training a KNN model can be found in `notebook/ml_experiment.ipynb`:
```python:src/model.py
startLine: 31
endLine: 49
```

## Testing
To run the tests, use `pytest`:
```
pytest
```

## Configuration
### `.gitignore`
The `.gitignore` file is configured to ignore the virtual environment and other common directories:
```plaintext:.gitignore
startLine: 1
endLine: 2
```

### `ruff.toml`
The `ruff.toml` file is configured for linting and formatting:
```toml:ruff.toml
startLine: 1
endLine: 49
```

### VSCode Settings
The VSCode settings are configured for Python development:
```json:.vscode/settings.json
startLine: 1
endLine: 24
```

## License
This project is licensed under the MIT License.