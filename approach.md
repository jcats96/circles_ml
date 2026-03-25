# Approaches to Circle Counting in 32×32 Binary Images

## Problem Summary

We want to train a neural network to count the number of circles present in a
32×32 grayscale image where every pixel is either black (0) or white (1).

---

## Input Representation

Each image is 32 pixels wide × 32 pixels high = **1,024 pixels** in total.
Because the images are binary (two possible pixel values), we can represent
each image as a flat vector of 1,024 numbers or as a (32, 32, 1) tensor.

**The first (input) layer must therefore accept 32×32 = 1,024 values.**

---

## Approach 1 – Fully-Connected (Dense) Network

This is the simplest starting point and is already implemented in `model.py`.

```
Input (32×32×1)
  └─ Flatten → 1,024 neurons
       └─ Dense(128, relu)   ← one hidden layer
            └─ Dense(1, relu) ← output: predicted count
```

### Why it works
Every pixel is fed directly to the hidden layer. The network can learn
arbitrary combinations of pixels that correspond to circular shapes.

### Limitations
The network has no built-in spatial awareness; it treats a circle in the
top-left corner differently from the same circle in the bottom-right corner
(not translation-invariant).

### When to use it
Good first experiment. Establishes a performance baseline with minimal
complexity.

---

## Approach 2 – Convolutional Neural Network (CNN)

CNNs are the standard approach for image tasks because convolutional filters
slide over the entire image and detect local patterns (edges, curves) regardless
of where they appear.

```
Input (32×32×1)
  └─ Conv2D(32, 3×3, relu) + MaxPooling(2×2)
       └─ Conv2D(64, 3×3, relu) + MaxPooling(2×2)
            └─ Flatten
                 └─ Dense(64, relu)
                      └─ Dense(1, relu) ← output: predicted count
```

### Why it works
- Small filters learn to detect arc/edge segments.
- Pooling layers provide some translation invariance.
- Stacking two conv blocks on a 32×32 image is usually sufficient before
  feature maps become too small.

### When to use it
Once the dense baseline is established, a CNN typically yields significantly
better accuracy for spatial tasks like this one.

---

## Approach 3 – Treating it as Classification

Instead of predicting a continuous count (regression), frame the problem as
multi-class classification where class *k* means "there are *k* circles".

- Replace the final `Dense(1, relu)` with `Dense(max_circles + 1, softmax)`.
- Use `sparse_categorical_crossentropy` as the loss function.
- The label for each image is the integer circle count (0, 1, 2, …).

### Trade-offs
| | Regression | Classification |
|---|---|---|
| Output | Any non-negative float | Probability per class |
| Loss | MSE / MAE | Categorical cross-entropy |
| Good when | Range is large / continuous | Range is small and discrete |

For small, fixed maximum counts (e.g. 0–5 circles), classification can
train faster and give cleaner integer predictions.

---

## Approach 4 – Data Augmentation

Because the images are binary and circles are simple geometric shapes, you
can programmatically generate an unlimited training set:

1. Draw *k* non-overlapping circles of random radii at random positions on a
   blank 32×32 canvas using a library like `Pillow` or `OpenCV`.
2. Save as PNG and record the count *k* in the annotation file.
3. Augment with random noise (flip a small fraction of pixels) to improve
   robustness.

This removes the need for manual annotation entirely.

---

## Recommended Starting Point

1. Start with **Approach 1** (dense network, already in `model.py`) to verify
   the training pipeline end-to-end.
2. Switch to **Approach 2** (CNN) once everything works, and compare accuracy.
3. If the count range is small (e.g. 0–5), try **Approach 3** (classification).
4. Use **Approach 4** to generate as much labelled data as needed cheaply.
