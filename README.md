# circles_ml

Circle-counting experiments using TensorFlow/Keras on 32x32 binary images.

## Setup (Windows)

This project is configured to use Python 3.12 for TensorFlow compatibility.

1. Create the virtual environment:

```powershell
py -3.12 -m venv .venv
```

2. Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.12` is not available, install Python 3.12 first, then re-run the commands above.

Optional quick check:

```powershell
.\.venv\Scripts\python.exe -c "import sys, tensorflow as tf; print(sys.version); print(tf.__version__)"
```

3. Run the model sanity check:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Project layout

- `models/model_dense.py`: dense baseline model
- `models/model_dense_two_hidden.py`: dense model with 2 hidden layers
- `models/model_cnn.py`: CNN model
- `models/model_cnn_extra_hidden.py`: CNN model with an extra hidden layer
- `main.py`: imports/builds model variants and can run predictions

## Train and save weights

Run training on `training_data/labels.csv` + `training_data/images/`:

```powershell
.\.venv\Scripts\python.exe train_models.py --epochs 100 --batch-size 8
```

This writes:

- `weights/dense.weights.h5`
- `weights/dense_two_hidden.weights.h5`
- `weights/cnn.weights.h5`
- `weights/cnn_extra_hidden.weights.h5`

Load the saved weights later:

```powershell
.\.venv\Scripts\python.exe main.py --load-weights
```
