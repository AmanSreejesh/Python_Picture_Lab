"""
Warm vs. Cool Color Classifier: a small neural network from scratch (NumPy only)

Lesson goal: show that a *linear* rule (R > B) works fine for a simple,
hand-picked definition of "warm/cool" but breaks down once we switch to a
true hue-wheel definition (which wraps around and isn't linearly separable
in raw RGB space). A tiny 2-layer network (3 -> 8 -> 1) can learn the hue
boundary directly from labeled examples instead.

Architecture: 3 inputs (R, G, B) -> 8 hidden neurons -> 1 output (sigmoid)
Loss: Binary Cross-Entropy
Training: full-batch gradient descent, vectorized backprop

This version uses the "narrow" hue definition of warm (red/orange/yellow
only), which is the intuitive, real-world definition -- and, like a lot of
real classification problems (fraud, rare disease, etc.), it's genuinely
imbalanced: only about 25% of random saturated colors count as "warm."
Three strategies for handling that imbalance are built in and switchable
below (IMBALANCE_STRATEGY).
"""

import numpy as np
import colorsys

# ============================================================
# CONFIG - flip these to compare approaches
# ============================================================
HIDDEN_ACTIVATION = "relu"       # "relu" or "sigmoid"
IMBALANCE_STRATEGY = "none"      # "none", "oversample", or "class_weight"

# ============================================================
# 1. GENERATE LABELED DATA
# ============================================================
# Labels come from true HSV hue, not from a hand-written R>B rule.
# This makes the decision boundary nonlinear/circular in RGB space.
#
# NOTE: the cutoff below is 1/6 (60 degrees), which is pure yellow's exact
# hue. An earlier version of this used 0.16 (57.6 degrees), which put pure
# yellow just OUTSIDE the warm band by a hair -- a labeling bug, not a
# model problem. Worth showing students: a threshold that's off by a tiny
# amount silently mislabels real examples, and no amount of training fixes
# a wrong label.

WARM_HUE_CUTOFF_LOW = 1/6 + 0.001   # just past pure yellow, so yellow -> warm
WARM_HUE_CUTOFF_HIGH = 0.92         # red-violet boundary on the other side

def hue_label(r, g, b):
    """Return 1 (warm) or 0 (cool) based on hue, or None for near-gray pixels."""
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if s < 0.15:
        return None  # too little color to call warm or cool
    return 1 if (h <= WARM_HUE_CUTOFF_LOW or h >= WARM_HUE_CUTOFF_HIGH) else 0


def build_dataset(n_samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(n_samples, 3))

    X_list, y_list = [], []
    for r, g, b in rgb:
        label = hue_label(r, g, b)
        if label is not None:
            X_list.append([r, g, b])
            y_list.append(label)

    X = np.array(X_list, dtype=float) / 255.0      # normalize to 0-1
    y = np.array(y_list, dtype=float).reshape(-1, 1)
    return X, y, rgb


X, y, rgb_raw = build_dataset()
print(f"Dataset: {len(y)} labeled pixels (dropped low-saturation/gray ones)")
print(f"  Warm: {int(y.sum())}   Cool: {int(len(y) - y.sum())}   "
      f"(warm = {y.mean()*100:.1f}% -- genuinely imbalanced, by design)")

# ============================================================
# 2. BASELINE: the simple hand-written rule (R > B)
# ============================================================
kept_mask = np.array([hue_label(r, g, b) is not None for r, g, b in rgb_raw])
r_gt_b_preds = (rgb_raw[kept_mask][:, 0] > rgb_raw[kept_mask][:, 2]).astype(float).reshape(-1, 1)
baseline_acc = (r_gt_b_preds == y).mean()
majority_acc = max(y.mean(), 1 - y.mean())  # accuracy of always guessing the bigger class
print(f"\nBaseline 'R > B' rule accuracy:      {baseline_acc:.3f}")
print(f"'Always guess majority class' acc:   {majority_acc:.3f}")
print("(the baseline needs to beat majority-class guessing to prove it's")
print(" actually learning something about color, not just riding the imbalance)")

# ============================================================
# 3. HANDLE IMBALANCE (per IMBALANCE_STRATEGY)
# ============================================================

def oversample(X, y, seed=1):
    """Duplicate random minority-class examples until both classes match.
    Doesn't add new information -- just makes the network see the rare
    class more often per epoch."""
    rng = np.random.default_rng(seed)
    warm_idx = np.where(y[:, 0] == 1)[0]
    cool_idx = np.where(y[:, 0] == 0)[0]
    minority_idx, majority_idx = (warm_idx, cool_idx) if len(warm_idx) < len(cool_idx) else (cool_idx, warm_idx)
    n_needed = len(majority_idx) - len(minority_idx)
    extra = rng.choice(minority_idx, size=n_needed, replace=True)
    X_bal = np.vstack([X, X[extra]])
    y_bal = np.vstack([y, y[extra]])
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


class_weights = None  # per-class weight dict, only used if strategy == "class_weight"

if IMBALANCE_STRATEGY == "oversample":
    n_before = len(y)
    X, y = oversample(X, y)
    print(f"\nOversampled minority class: {n_before} -> {len(y)} samples "
          f"(now {int(y.sum())} warm / {int(len(y)-y.sum())} cool)")

elif IMBALANCE_STRATEGY == "class_weight":
    # Weight each class inversely to its frequency, so a rare warm example
    # contributes as much total gradient signal as several common cool ones.
    n_total = len(y)
    n_warm = y.sum()
    n_cool = n_total - n_warm
    class_weights = {0: n_total / (2 * n_cool), 1: n_total / (2 * n_warm)}
    print(f"\nUsing class weights: cool={class_weights[0]:.2f}  warm={class_weights[1]:.2f}")

# ============================================================
# 4. NEURAL NETWORK: 3 -> 8 -> 1, from scratch
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def init_params(n_in, n_hidden, n_out, seed=42):
    rng = np.random.default_rng(seed)
    if HIDDEN_ACTIVATION == "relu":
        W1 = rng.normal(0, 1, (n_in, n_hidden)) * np.sqrt(2 / n_in)   # He init
    else:
        W1 = rng.normal(0, 1, (n_in, n_hidden)) * np.sqrt(1 / n_in)   # Xavier init
    b1 = np.zeros((1, n_hidden))
    W2 = rng.normal(0, 1, (n_hidden, n_out)) * np.sqrt(1 / n_hidden)
    b2 = np.zeros((1, n_out))
    return W1, b1, W2, b2


def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1) if HIDDEN_ACTIVATION == "relu" else sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)  # output layer always sigmoid - we want a 0-1 probability
    return Z1, A1, Z2, A2


def bce_loss(A2, y, eps=1e-9):
    return -np.mean(y * np.log(A2 + eps) + (1 - y) * np.log(1 - A2 + eps))


def backward(X, y, Z1, A1, A2, W2):
    n = X.shape[0]

    # Output layer: dL/dZ2 simplifies to (A2 - y) because sigmoid + BCE
    # derivatives cancel out cleanly (same simplification as the
    # shirt/not-shirt classifier). If class_weight is active, each
    # example's error gets scaled by its class's weight before it flows
    # backward -- that's the ENTIRE mechanism behind class weighting.
    if class_weights is not None:
        sample_weights = np.where(y == 1, class_weights[1], class_weights[0])
        dZ2 = sample_weights * (A2 - y) / n
    else:
        dZ2 = (A2 - y) / n

    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)

    # Hidden layer: chain rule back through W2, then through the hidden
    # activation's derivative.
    #   sigmoid'(Z1) = A1 * (1 - A1)
    #   ReLU'(Z1)    = 1 where Z1 > 0, else 0   (a step function)
    dA1 = dZ2 @ W2.T
    if HIDDEN_ACTIVATION == "relu":
        dZ1 = dA1 * (Z1 > 0)
    else:
        dZ1 = dA1 * A1 * (1 - A1)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0, keepdims=True)

    return dW1, db1, dW2, db2


def train(X, y, n_hidden=8, lr=0.5, epochs=3000, print_every=500):
    n_in, n_out = X.shape[1], y.shape[1]
    W1, b1, W2, b2 = init_params(n_in, n_hidden, n_out)

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        loss = bce_loss(A2, y)
        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, A2, W2)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        if epoch % print_every == 0:
            preds = (A2 > 0.5).astype(float)
            acc = (preds == y).mean()
            print(f"  epoch {epoch:4d}   loss={loss:.4f}   acc={acc:.3f}")

    return W1, b1, W2, b2


print(f"\nTraining 3 -> 8 -> 1 network "
      f"(hidden activation: {HIDDEN_ACTIVATION}, imbalance strategy: {IMBALANCE_STRATEGY})...")
W1, b1, W2, b2 = train(X, y)

# ============================================================
# 5. EVALUATE -- against the TRUE real-world (imbalanced) population,
#    regardless of what strategy was used to train, so results are
#    comparable across all three strategies.
# ============================================================

def evaluate_on_true_population(W1, b1, W2, b2, n_test=5000, seed=99):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(n_test, 3))
    X_list, y_list = [], []
    for r, g, b in rgb:
        lab = hue_label(r, g, b)
        if lab is not None:
            X_list.append([r, g, b])
            y_list.append(lab)
    X_test = np.array(X_list, dtype=float) / 255.0
    y_test = np.array(y_list, dtype=float).reshape(-1, 1)

    _, _, _, A2 = forward(X_test, W1, b1, W2, b2)
    preds = (A2 > 0.5).astype(float)
    acc = (preds == y_test).mean()

    warm_mask = y_test[:, 0] == 1
    warm_recall = (preds[warm_mask] == 1).mean() if warm_mask.sum() > 0 else float("nan")
    cool_recall = (preds[~warm_mask] == 0).mean() if (~warm_mask).sum() > 0 else float("nan")
    return acc, warm_recall, cool_recall


acc, warm_recall, cool_recall = evaluate_on_true_population(W1, b1, W2, b2)
print(f"\nEvaluated on the TRUE real-world color distribution "
      f"(not the training strategy's resampled/reweighted one):")
print(f"  Overall accuracy: {acc:.3f}")
print(f"  Warm recall:      {warm_recall:.3f}   (of truly-warm pixels, % correctly caught)")
print(f"  Cool recall:      {cool_recall:.3f}   (of truly-cool pixels, % correctly caught)")
print("  (compare these three numbers across IMBALANCE_STRATEGY settings --")
print("   oversampling/class_weight typically trade a little cool_recall for warm_recall)")


# ============================================================
# 6. TRY IT ON A SINGLE COLOR
# ============================================================
def classify_color(r, g, b, W1, b1, W2, b2):
    x = np.array([[r, g, b]], dtype=float) / 255.0
    _, _, _, pred = forward(x, W1, b1, W2, b2)
    label = "warm" if pred[0, 0] > 0.5 else "cool"
    return label, float(pred[0, 0])


if __name__ == "__main__":
    test_colors = {
        "pure red":    (255, 0, 0),
        "pure blue":   (0, 0, 255),
        "yellow":      (255, 255, 0),
        "green":       (0, 255, 0),
        "purple":      (150, 0, 200),
        "orange":      (255, 140, 0),
    }
    print("\nSpot-checking a few named colors:")
    for name, (r, g, b) in test_colors.items():
        label, score = classify_color(r, g, b, W1, b1, W2, b2)
        print(f"  {name:10s} RGB{(r, g, b)}  ->  {label}  (score={score:.3f})")