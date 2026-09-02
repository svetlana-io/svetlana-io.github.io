---
title: "Best Practices for Seeding"
author: "Svetlana"
pubDatetime: 2026-07-17T18:45:00Z
slug: "best-practices-for-seeding"
featured: true
draft: false
tags:
  - python
  - debugging
  - best-practices
description: "How hidden RNG reseeding can silently corrupt randomness across your data and training pipelines."
---

> **Key takeaway:** Seed shared RNGs at the application level, not inside utilities. If a component needs independent reproducible randomness, give it its own local generator.


Libraries often rely on a **global state** for randomness. When you call `random.seed(42)` or `np.random.seed(42)`, you aren't creating a scoped random number generator; you are resetting the shared RNG state. 

This is why setting seeds deep inside sub-libraries or utility functions is a terrible practice. A single `random.seed()` hidden in a helper function can silently overwrite the randomness of your entire pipeline.

The catch is that your code won't crash. You may think you have a perfectly random pipeline until some module kicks in mid-processing and resets the RNG state. From that point on, your code may start replaying a previously seen random sequence, possibly shifted by however many values the module consumes.

# Bug in Action

To understand why this is so dangerous, let's look at a concrete example. Imagine you have a utility module that tries to be "helpful" by enforcing reproducibility locally, completely unaware it is resetting shared RNG state.

```python
import random

def sneaky_utility(seed=42):
    # BAD PRACTICE: Setting the global seed deep in a utility module!
    random.seed(seed) 
    
    # It consumes one random number for its own internal logic
    utility_val = random.randint(1, 100)
    print(f"   [bad_module] Consumed 1 number: {utility_val}")
```

Now, watch what happens to the sequence in your main execution script after bad_module.py is called.

```python
import random
import bad_module

random.seed(42)

# Main pipeline uses randomness for object initialization
position = random.randint(1, 100)
size = random.randint(1, 100)

print(f"Object position: {position}")
print(f"Object size: {size}")

print("\n--- Calling external utility ---")
bad_module.sneaky_utility()
print("--------------------------------\n")

# Later, the same RNG is used for transformations
rotation = random.randint(1, 100)
brightness = random.randint(1, 100)

print(f"Rotation: {rotation}")
print(f"Brightness: {brightness}")
```

The output:
```
Object position: 82
Object size: 15

--- Calling external utility ---
   [bad_module] Consumed 1 number: 82
--------------------------------

Rotation: 15
Brightness: 4
```
The pipeline did not produce an obvious duplicate output. Instead, the same underlying RNG sequence was reused for different operations. Your "random" main pipeline generated `82, 15, 15, 4`. Because one random value might control initialization while another controls a transformation, the outputs can look unrelated even though they come from the same repeated sequence.

That is what makes this bug dangerous in real pipelines: the duplication happens at the RNG-state level, not necessarily in the visible output. In randomized data processing or model training, this kind of hidden dependence can quietly change the statistical behavior of your experiment.

# How to Fix It

## Conceptual

To avoid these silent failures, follow two architectural rules.

### Fix 1: Centralize Global Seeding
Do not let sub-modules, utilities, or data loaders reset shared RNG state. If your application needs global reproducibility, configure those seeds at the top level of your program, where the randomness policy is explicit and visible (e.g., right after the imports in `main.py`).

### Fix 2: Use Local Random Generators
If a utility needs reproducible randomness independent of the rest of the program, give it its own generator instead of resetting shared RNG state.
```python
import random
import numpy as np

def safe_utility():
    # Python standard library
    local_rng = random.Random(42)
    val1 = local_rng.randint(1, 100)
    
    # Modern Numpy approach
    np_rng = np.random.default_rng(42)
    val2 = np_rng.integers(1, 100)
```

## Practical

### PyTorch
If you are writing raw PyTorch, your top-level `main.py` needs a comprehensive seeding function:
```python
import torch
import numpy as np
import random

def set_global_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_global_seeds(42)
```
If you also need deterministic operations, PyTorch provides:
```python
torch.use_deterministic_algorithms(True)
```
This makes PyTorch use deterministic implementations where available and raise an error when an operation has no deterministic alternative. It can also reduce performance.

### PyTorch Lightning
If you are using PyTorch Lightning, you do not need to write the seeding boilerplate yourself. Lightning provides a built-in utility for seeding Python, NumPy, and PyTorch consistently:
```python
from lightning.pytorch import seed_everything

seed_everything(42, workers=True)
```
The `workers=True` flag also helps initialize DataLoader workers with distinct, reproducible random states.

### TensorFlow
TensorFlow also provides top-level utilities for reproducible training. A simple setup is:

```python
import tensorflow as tf 

tf.keras.utils.set_random_seed(42) 
tf.config.experimental.enable_op_determinism()
```

`set_random_seed()` seeds Python, NumPy, and TensorFlow, while `enable_op_determinism()` asks TensorFlow to use deterministic implementations of operations where possible.
Deterministic execution can reduce performance, and reproducibility is still best understood within a fixed software and hardware environment rather than as an absolute guarantee.