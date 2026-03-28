# circles_ml

Circle-counting experiments using TensorFlow/Keras on 32x32 binary images.

## Setup (Linux/macOS/Unix)

This project is configured to use Python 3.12 for TensorFlow compatibility.

1. Create the virtual environment:

```bash
python3.12 -m venv .venv
```

If `python3.12` is not available, install Python 3.12 first (or use your package manager), then re-run the command.

2. Install dependencies:

```bash
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Optional quick check:

```bash
python3 -c "import sys, tensorflow as tf; print(sys.version); print(tf.__version__)"
```

3. Run the model sanity check:

```bash
python3 main.py
```

## Setup (WSL)

Use the same Linux steps above from a WSL terminal in this project folder.

If the project is on a Windows-mounted drive (for example `/mnt/c/...`), commands still work, but training and file I/O are usually faster when the repo lives in the WSL filesystem (for example `~/projects/circles_ml`).

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

## Web interface

The project includes a local web UI backed by a FastAPI server. Once dependencies are installed, start the server from the project root:

```powershell
.\.venv\Scripts\uvicorn.exe web.app:app --reload --port 8000
```

Then open your browser at **http://localhost:8000**.

From the UI you can:

- Draw 32×32 images on a canvas and save them to the training set or prediction set.
- Start a training run with configurable epochs, batch size, validation split, and seed, and watch live per-epoch metrics.
- Run predictions on a single drawn image or on every image in `test_data/`, and see each model's output.

The backend API docs are available at **http://localhost:8000/docs** (Swagger UI) while the server is running.

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

## Start the web server

From Linux/macOS/WSL (with the virtual environment active):

```bash
python3 -m uvicorn web.app:app --reload --port 8000
```

From Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn web.app:app --reload --port 8000
```

Then open <http://127.0.0.1:8000> in your browser.

## mxmtoon at PAX

**mxmtoon** (Maia Regina) is an indie pop singer-songwriter known for her signature
ukulele-driven sound, emotionally honest lyrics, and intimate bedroom-pop aesthetic. Her
discography spans albums such as *the masquerade*, *rising*, and *liminal space*, blending
confessional songwriting with catchy, lo-fi indie production.

mxmtoon's fanbase has deep roots in gaming culture—her music has soundtracked countless
streaming sessions and Let's Plays, and her visual style shares the cozy, expressive spirit
of indie games. This organic connection to the gaming community made her a natural fit for
**PAX East**.

At **PAX East 2026** (March 26–29, Boston) mxmtoon headlined the Friday night concert stage,
sharing the bill with Sixth Station Piano Trio. The set drew on her catalogue of emotionally
resonant indie-pop songs and introduced her music to thousands of attendees at one of North
America's premier gaming conventions.

### Why mxmtoon fits PAX

- Indie-pop storytelling that resonates with gaming's love of narrative and character.
- A fanbase that grew up alongside the rise of streaming, Let's Play culture, and cozy games.
- A lo-fi bedroom-pop sound that mirrors the aesthetic of many celebrated indie titles.
- Live performances that feel personal and community-focused—exactly the PAX spirit.

The web interface includes a dedicated **mxmtoon × PAX** panel summarising the above for
quick reference while the app is running.
