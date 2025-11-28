# Generator

Event generation module for three-body decays with VEGAS importance sampling.

## Overview

The `generator` module provides the main interface for generating Monte Carlo events:

- **`weighted_generation()`**: Main event generation function
- **`save_data()`**: Save events to disk in NumPy format
- **`main()`**: Command-line interface

All functions support negative weights from destructive quantum interference.

---

::: three_body_decay.generator.weighted_generation
    options:
      show_source: true

---

::: three_body_decay.generator.save_data
    options:
      show_source: false

---

::: three_body_decay.generator.main
    options:
      show_source: false

## Usage Examples

### Basic Generation

```python
from three_body_decay import weighted_generation

# Generate 10k events with default resonances
events, weights, s12, s23 = weighted_generation(
    M=1.864,      # Parent mass (GeV)
    m1=0.139,     # Daughter 1 mass
    m2=0.139,     # Daughter 2 mass
    m3=0.139,     # Daughter 3 mass
    n_events=10000
)

# Each event is a dict with four-vectors
print(events[0])
# {'p1': array([E1, px1, py1, pz1]),
#  'p2': array([E2, px2, py2, pz2]),
#  'p3': array([E3, px3, py3, pz3])}
```

### With Custom Resonances

```python
import numpy as np

# Define three resonances
my_resonances = [
    {'mass': 0.770, 'width': 0.150, 'weight': 1.0,
     'channel': 's12', 'phase': 0.0, 'sign': +1},
    {'mass': 0.980, 'width': 0.070, 'weight': 0.8,
     'channel': 's23', 'phase': np.pi/4, 'sign': +1},
    {'mass': 1.270, 'width': 0.185, 'weight': 0.5,
     'channel': 's13', 'phase': np.pi/2, 'sign': +1},
]

events, weights, s12, s23 = weighted_generation(
    M=1.864, m1=0.139, m2=0.139, m3=0.139,
    resonances=my_resonances,
    n_events=10000,
    nitn=15,      # More VEGAS iterations
    neval=20000   # More evaluations per iteration
)
```

### With Negative Weights

```python
# Destructive interference configuration
resonances = [
    {'mass': 0.770, 'width': 0.150, 'weight': 1.0,
     'channel': 's12', 'phase': 0.0, 'sign': +1},   # Constructive
    {'mass': 0.980, 'width': 0.070, 'weight': 0.5,
     'channel': 's23', 'phase': np.pi/2, 'sign': -1},  # Destructive!
]

events, weights, s12, s23 = weighted_generation(
    M=1.864, m1=0.139, m2=0.139, m3=0.139,
    resonances=resonances,
    n_events=10000
)

# Analyze negative weights
n_negative = np.sum(weights < 0)
print(f"Negative weight events: {n_negative}/{len(weights)}")
# Typical output: ~20-30% negative weights
```

### Saving Data

```python
from three_body_decay.generator import save_data

# Save events to disk
save_data(events, label='signal', output_path='./my_data')

# Creates:
#   my_data/threebodydecay_signal.npy      (N, 3, 4) array
#   my_data/threebodydecay_labels_signal.npy  (N,) array
```

## VEGAS Parameters

### Choosing `nitn` and `neval`

The VEGAS adaptation quality depends on:

- **`nitn`** (iterations): More iterations = better adaptation
  - Default: 10 (fast, good for testing)
  - Recommended: 15-20 (production)
  - High precision: 30+ (slow)

- **`neval`** (evaluations per iteration): More evaluations = finer grid
  - Default: 10000 (fast)
  - Recommended: 20000-50000 (production)
  - High precision: 100000+ (very slow)

!!! tip "Performance vs Accuracy"
    - For testing: `nitn=10, neval=10000` (~5 seconds)
    - For production: `nitn=15, neval=20000` (~15 seconds)
    - For publication: `nitn=20, neval=50000` (~60 seconds)

### VEGAS Sampling

The generator uses VEGAS in two stages:

1. **Integration**: Adapt grid and compute total width Γ
2. **Sampling**: Generate events from adapted grid
3. **Re-evaluation**: Compute signed weights (for negative weights)

!!! note "Why Re-evaluate?"
    VEGAS importance weights don't capture the **sign** of the amplitude. We must re-evaluate the signed integrand at each point to get correct negative weights.

## Weight Normalization

Weights are normalized so that `sum(|weights|) = 1.0`:

```python
# Before normalization
sum(weights) = Γ * V  # V = phase space volume

# After normalization
sum(|weights|) = 1.0
sum(weights) ≈ 1.0  # If no negative weights
sum(weights) < 1.0  # If negative weights present
```

!!! warning "Physical Constraint"
    Before normalization, `sum(weights)` must be positive. If negative, your configuration is **unphysical** - destructive interference exceeds constructive.

## Event Structure

Each event is a dictionary with three four-vectors:

```python
event = {
    'p1': np.array([E1, px1, py1, pz1]),  # Particle 1
    'p2': np.array([E2, px2, py2, pz2]),  # Particle 2
    'p3': np.array([E3, px3, py3, pz3]),  # Particle 3
}

# Four-momentum conservation (parent at rest)
p1 + p2 + p3 = [M, 0, 0, 0]

# Invariant masses
s12 = (p1 + p2)²
s23 = (p2 + p3)²
s13 = (p1 + p3)²
```

## Common Issues

### Too Few Events Generated

If VEGAS only generates a few events:

```python
# Bad: Default parameters with complex resonances
events, _, _, _ = weighted_generation(
    M=1.864, m1=0.139, m2=0.139, m3=0.139,
    resonances=complex_resonances,
    n_events=10000  # Might only get 100!
)

# ✓ Good: Increase VEGAS parameters
events, _, _, _ = weighted_generation(
    M=1.864, m1=0.139, m2=0.139, m3=0.139,
    resonances=complex_resonances,
    n_events=10000,
    nitn=20,       # More iterations
    neval=50000    # More evaluations
)
```

### Negative Total Weight

If you see `sum(weights) < 0`:

```python
# Unphysical configuration
resonances = [
    {'mass': 0.770, 'width': 0.150, 'weight': 1.0,
     'channel': 's12', 'phase': 0.0, 'sign': +1},
    {'mass': 0.980, 'width': 0.070, 'weight': 0.9,  # Too strong!
     'channel': 's23', 'phase': np.pi, 'sign': -1},
]

# Fix: Reduce destructive weight
resonances[1]['weight'] = 0.5  # ✓ Now physical
```

## See Also

- [Likelihood](likelihood.md): Compute event probabilities
- [Plotter](plotter.md): Visualize generated events
- [Integrator](integrator.md): Physics calculations
- [Examples: N-Resonances](../examples/n_resonances.md)
- [Examples: Negative Weights](../examples/negative_weights.md)
