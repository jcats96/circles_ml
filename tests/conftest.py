"""
tests/conftest.py – shared pytest fixtures and setup for the test suite.

TensorFlow is mocked at the sys.modules level so that the project modules
can be imported and tested without a real TF installation.
"""

import sys
import types


def _install_tf_mock():
    """Install a lightweight tensorflow stub into sys.modules if TF is absent."""
    if "tensorflow" in sys.modules:
        return  # real TF is present; nothing to do

    tf_mock = types.ModuleType("tensorflow")

    # ── keras.callbacks ──────────────────────────────────────────────────────
    keras_callbacks_mock = types.ModuleType("tensorflow.keras.callbacks")

    class _FakeCallback:
        """Minimal stand-in for keras.callbacks.Callback."""

        def __init__(self):
            self.model = None

        def on_epoch_end(self, epoch, logs=None):
            pass

    keras_callbacks_mock.Callback = _FakeCallback

    # ── keras.Model / Sequential / layers ────────────────────────────────────
    keras_layers_mock = types.ModuleType("tensorflow.keras.layers")
    keras_mock = types.ModuleType("tensorflow.keras")
    keras_mock.callbacks = keras_callbacks_mock
    keras_mock.layers = keras_layers_mock

    # Minimal Sequential / Model stubs
    class _FakeSequential:
        def __init__(self, *args, **kwargs):
            pass

        def compile(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def predict(self, *a, **kw):
            import numpy as np
            return [[0.0]]

        def load_weights(self, *a, **kw):
            pass

        def save_weights(self, *a, **kw):
            pass

        def count_params(self):
            return 0

        @property
        def name(self):
            return "mock_model"

    keras_mock.Sequential = _FakeSequential
    keras_mock.Model = _FakeSequential

    tf_mock.keras = keras_mock

    # ── io / image stubs ─────────────────────────────────────────────────────
    tf_io_mock = types.ModuleType("tensorflow.io")
    tf_image_mock = types.ModuleType("tensorflow.image")
    tf_mock.io = tf_io_mock
    tf_mock.image = tf_image_mock

    # Register everything
    sys.modules["tensorflow"] = tf_mock
    sys.modules["tensorflow.keras"] = keras_mock
    sys.modules["tensorflow.keras.callbacks"] = keras_callbacks_mock
    sys.modules["tensorflow.keras.layers"] = keras_layers_mock
    sys.modules["tensorflow.io"] = tf_io_mock
    sys.modules["tensorflow.image"] = tf_image_mock


_install_tf_mock()
