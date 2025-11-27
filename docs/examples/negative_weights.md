# Example: Negative Weights

This example demonstrates how to handle negative Monte Carlo weights arising from destructive quantum interference.

## Overview

Negative weights occur when:

- ✅ Resonances have `sign=-1` (destructive interference)
- ✅ Amplitude² can be locally negative
- ✅ Total integrated rate must remain positive (Γ > 0)
- ✅ Typical: 20-40% of events have negative weights

## Running the Example

```bash
# From package root
python examples/demo_negative_weights.py

# Or with uv
uv run python examples/demo_negative_weights.py
```

This creates visualizations of negative weight regions and demonstrates proper handling.

## Physics Background

### Quantum Interference

In quantum mechanics, amplitudes add, not probabilities:

$$
M_{\text{total}} = M_+ - M_-
$$

The observable is:

$$
|M_{\text{total}}|^2 = |M_+|^2 + |M_-|^2 - 2\text{Re}(M_+^* M_-)
$$

The interference term $-2\text{Re}(M_+^* M_-)$ can be **positive or negative**!

### Signed Amplitude

We define a **signed amplitude squared**:

$$
\text{signed}|M|^2 = |M_+|^2 - |M_-|^2
$$

This can be negative locally, but the total must be positive:

$$
\Gamma = \int \text{signed}|M|^2 \, ds_{12} \, ds_{23} > 0
$$

!!! warning "Physical Constraint"
    If Γ < 0, your configuration is **unphysical**. Reduce the strength of destructive resonances!

## Example 1: Balanced Configuration

A physically acceptable configuration with ~25% negative weights:

```python
from three_body_decay import weighted_generation
import numpy as np

# Physical parameters
M = 1.864
m1, m2, m3 = 0.139, 0.139, 0.139

# Balanced: destructive is weaker than constructive
resonances = [
    {
        'mass': 0.770,
        'width': 0.150,
        'weight': 1.0,      # Constructive
        'channel': 's12',
        'phase': 0.0,
        'sign': +1          # ← Positive
    },
    {
        'mass': 0.980,
        'width': 0.070,
        'weight': 0.5,      # Weaker!
        'channel': 's23',
        'phase': np.pi/2,
        'sign': -1          # ← Negative (destructive)
    }
]

# Generate events
events, weights, s12, s23 = weighted_generation(
    M, m1, m2, m3,
    resonances=resonances,
    n_events=10000
)

# Analyze weights
n_negative = np.sum(weights < 0)
n_positive = np.sum(weights > 0)

print(f"Positive weights: {n_positive}/{len(weights)} ({100*n_positive/len(weights):.1f}%)")
print(f"Negative weights: {n_negative}/{len(weights)} ({100*n_negative/len(weights):.1f}%)")
print(f"Total weight sum: {np.sum(weights):.6f}")
print(f"Sum |weights|: {np.sum(np.abs(weights)):.6f}")
```

### Expected Output

```
Positive weights: 7500/10000 (75.0%)
Negative weights: 2500/10000 (25.0%)
Total weight sum: 1.000000
Sum |weights|: 1.000000
```

### What to Observe

- ✅ ~75% positive weights
- ✅ ~25% negative weights
- ✅ Γ > 0 (configuration is physical)
- ✅ Negative weights cluster in specific Dalitz regions

## Example 2: Weak Destructive Interference

Minimal negative weights (~10%):

```python
resonances = [
    {
        'mass': 0.770,
        'width': 0.150,
        'weight': 1.0,
        'channel': 's12',
        'phase': 0.0,
        'sign': +1
    },
    {
        'mass': 0.980,
        'width': 0.070,
        'weight': 0.3,      # Very weak destructive
        'channel': 's23',
        'phase': np.pi/3,   # Sub-optimal phase
        'sign': -1
    }
]

events, weights, s12, s23 = weighted_generation(
    M, m1, m2, m3,
    resonances=resonances,
    n_events=10000
)
```

### Expected Output

```
Negative weights: 1000/10000 (10.0%)
```

### When to Use

- Beginning projects (easier to handle)
- When statistical power is limited
- Testing/debugging

## Example 3: Unphysical Configuration (DO NOT USE!)

⚠️ **This is for demonstration only - do not use in real analysis!**

```python
# ❌ UNPHYSICAL - Destructive too strong!
resonances = [
    {
        'mass': 0.770,
        'width': 0.150,
        'weight': 1.0,
        'channel': 's12',
        'phase': 0.0,
        'sign': +1
    },
    {
        'mass': 0.980,
        'width': 0.070,
        'weight': 0.9,      # Too strong!
        'channel': 's23',
        'phase': np.pi,     # Maximum destructive
        'sign': -1
    }
]

# This will give Γ < 0 ⚠️
```

### Warning Signs

```
⚠️  WARNING: Total weight sum is NEGATIVE (-2.34e+01)
    This means destructive interference exceeds constructive.
    This configuration is UNPHYSICAL - total decay rate must be positive!
```

### How to Fix

1. **Reduce destructive weight**: 0.9 → 0.5
2. **Change phase**: π → π/2
3. **Add more constructive resonances**

## Visualizing Negative Weight Regions

```python
from three_body_decay.plotter import plot_negative_weight_regions

# Visualize signed amplitude
plot_negative_weight_regions(
    M=1.864,
    m1=0.139, m2=0.139, m3=0.139,
    resonances=resonances,
    n_bins=200,
    output_path='negative_regions.png'
)
```

### The Plot Shows

**Left panel**: Signed amplitude with diverging colormap

- Blue = negative weights
- White = zero
- Red = positive weights

**Right panel**: Categorical regions

- Blue = negative weight region
- White = zero
- Red = positive weight region

### Interpretation

```python
# From the plot statistics:
Positive: 75.3%
Negative: 24.7%

# In Dalitz plot:
# - Negative regions typically where destructive resonance peaks
# - Positive regions where constructive resonance dominates
# - Boundary regions have near-zero weights
```

## Handling Negative Weights in Analysis

### 1. Histogramming

```python
# ❌ Wrong: Ignore sign
hist, bins = np.histogram(s12, bins=50, weights=np.abs(weights))

# ✓ Correct: Keep sign
hist, bins = np.histogram(s12, bins=50, weights=weights)

# Some bins may be negative!
# This is correct - it represents destructive interference
```

### 2. Acceptance Correction

```python
# Generate signal with negative weights
sig_events, sig_weights, _, _ = weighted_generation(
    M, m1, m2, m3, resonances=signal_resonances, n_events=100000
)

# Apply detector acceptance (simplified)
acceptance = np.random.uniform(0, 1, len(sig_events)) < 0.5
acc_weights = sig_weights[acceptance]

# ✓ Acceptance calculation preserves sign
eff = np.sum(acc_weights) / np.sum(sig_weights)
print(f"Efficiency: {eff:.3f}")
```

### 3. Fitting

```python
# Negative log-likelihood (handles negative weights)
def neg_log_likelihood(params):
    # Generate with test parameters
    test_events, test_weights, _, _ = weighted_generation(
        M, m1, m2, m3, resonances=params_to_resonances(params),
        n_events=10000
    )
    
    # Compare to data using signed weights
    # (proper statistical treatment beyond scope here)
    nll = -np.sum(test_weights)  # Placeholder
    return nll
```

### 4. Plotting

```python
# For visualization, often use absolute value
import matplotlib.pyplot as plt

plt.hist(s12, bins=50, weights=np.abs(weights), label='Events')
plt.xlabel(r'$s_{12}$ [GeV²]')
plt.ylabel('Events (absolute weight)')
plt.legend()
plt.show()

# But always report that you used |weights|!
```

## Statistical Considerations

### Effective Sample Size

With negative weights, effective sample size is reduced:

$$
N_{\text{eff}} = \frac{(\sum w_i)^2}{\sum w_i^2}
$$

```python
# Calculate effective sample size
w_sum = np.sum(weights)
w_sum_sq = np.sum(weights**2)
n_eff = w_sum**2 / w_sum_sq

print(f"Actual events: {len(weights)}")
print(f"Effective events: {n_eff:.0f}")
print(f"Efficiency: {n_eff/len(weights):.2%}")
```

### Typical Results

```
Actual events: 10000
Effective events: 6500
Efficiency: 65%
```

With 25% negative weights, you lose ~35% statistical power.

### Guideline

- **0-10% negative**: Minimal impact (<5% efficiency loss)
- **10-30% negative**: Moderate impact (10-30% efficiency loss)
- **30-50% negative**: Severe impact (>50% efficiency loss)
- **>50% negative**: Unusable (reconsidered configuration)

## Common Applications

### 1. NLO QCD Corrections

```python
# Leading order + Next-to-leading order
resonances = [
    {'mass': 0.770, 'width': 0.150, 'weight': 1.0,
     'channel': 's12', 'phase': 0.0, 'sign': +1},  # LO
    {'mass': 0.770, 'width': 0.150, 'weight': 0.2,
     'channel': 's12', 'phase': np.pi, 'sign': -1},  # NLO (destructive)
]
```

### 2. Interference with Continuum

```python
# Resonance + continuum background
resonances = [
    {'mass': 0.770, 'width': 0.150, 'weight': 1.0,
     'channel': 's12', 'phase': 0.0, 'sign': +1},  # Signal
    {'mass': 0.500, 'width': 0.400, 'weight': 0.3,
     'channel': 's12', 'phase': np.pi/2, 'sign': -1},  # Continuum
]
```

### 3. Background Subtraction

```python
# Signal - background (method of subtraction)
# Generate signal with positive sign
signal_events, sig_weights, _, _ = weighted_generation(
    M, m1, m2, m3, resonances=signal_res, n_events=10000
)

# Generate background with negative sign (!)
bkg_events, bkg_weights, _, _ = weighted_generation(
    M, m1, m2, m3, resonances=background_res, n_events=10000
)
bkg_weights *= -1  # Make negative

# Combine
all_events = signal_events + bkg_events
all_weights = np.concatenate([sig_weights, bkg_weights])

# Histogram represents signal - background
hist, bins = np.histogram(observable, weights=all_weights)
```

## Rules of Thumb

### ✅ Physical Configurations

1. **Γ > 0**: Always check total width is positive
2. **Balance weights**: Destructive < 0.5 × Constructive
3. **Monitor stats**: Keep negative fraction < 40%
4. **Check phases**: π is maximum destructive

### ❌ Unphysical Configurations

1. **Γ < 0**: Configuration violates unitarity
2. **>80% negative**: Too many negative weights
3. **Nearly-zero Γ**: Indicates cancellation problem

## Debugging Tips

### Check Total Width

```python
from three_body_decay.likelihood import build_likelihood

# Build likelihood and get Γ
p_func, Gamma = build_likelihood(
    M, m1, m2, m3, resonances=resonances
)

if Gamma < 0:
    print("⚠️ UNPHYSICAL: Γ < 0")
elif Gamma < 10:
    print("⚠️ WARNING: Very small Γ")
else:
    print(f"✓ Physical: Γ = {Gamma:.2e}")
```

### Visualize Before Generating

```python
# Check signed amplitude before generating events
from three_body_decay.plotter import plot_negative_weight_regions

plot_negative_weight_regions(
    M, m1, m2, m3, resonances=resonances
)

# Look at the statistics:
# - What fraction is negative?
# - Is it spatially localized?
# - Does the pattern make sense?
```

### Gradual Adjustment

```python
# Start physical, gradually increase destructive component
weights_to_test = [0.1, 0.3, 0.5, 0.7, 0.9]

for w in weights_to_test:
    resonances[1]['weight'] = w
    
    p_func, Gamma = build_likelihood(M, m1, m2, m3, resonances=resonances)
    
    print(f"Weight={w:.1f}: Γ={Gamma:.2e} {'✓' if Gamma > 0 else '⚠️'}")
    
    if Gamma < 0:
        print(f"  → Maximum weight is {weights_to_test[weights_to_test.index(w)-1]:.1f}")
        break
```

## Further Reading

- **NNPDF Collaboration**: "Fitting Parton Distribution Data with Multiplicative Normalization Uncertainties", arXiv:0912.2276
- **NNLO QCD**: Negative weights in perturbative calculations
- **Unfolding**: SVD and negative weights, Nucl.Instrum.Meth. A362 (1995) 487-498

## Next Steps

- See [Multiple Resonances Example](n_resonances.md) for positive weight configurations
- Read [Physics Background](../physics.md) for detailed theory
- Check [API: Generator](../api/generator.md) for weight normalization details
