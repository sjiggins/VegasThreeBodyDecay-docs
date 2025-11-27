# Likelihood

Likelihood calculation module for computing event probabilities.

## Overview

The `likelihood` module provides functions for computing normalized probability densities:

- **`build_likelihood()`**: Create callable p(s₁₂, s₂₃) function
- **`event_likelihoods()`**: Compute likelihoods for events in unit hypercube
- **`event_likelihood_px()`**: Compute likelihoods from four-momenta
- **`p_from_unit()`**: Transform unit hypercube to Dalitz coordinates

These functions are used for:

- Probability density visualization
- Maximum likelihood fitting
- Reweighting events
- Statistical analysis

---

::: three_body_decay.likelihood.build_likelihood
    options:
      show_source: true

---

::: three_body_decay.likelihood.event_likelihoods
    options:
      show_source: true

---

::: three_body_decay.likelihood.event_likelihood_px
    options:
      show_source: false

---

::: three_body_decay.likelihood.p_from_unit
    options:
      show_source: false

---

::: three_body_decay.likelihood.invariant_mass2
    options:
      show_source: false

## Usage Examples

### Build Likelihood Function

```python
from three_body_decay.likelihood import build_likelihood
import numpy as np

# Physical parameters
M = 1.864
m1, m2, m3 = 0.139, 0.139, 0.139

# Resonance configuration
resonances = [
    {'mass': 0.770, 'width': 0.150, 'weight': 1.0,
     'channel': 's12', 'phase': 0.0, 'sign': +1},
]

# Build normalized likelihood function
p_func, Gamma = build_likelihood(
    M, m1, m2, m3,
    resonances=resonances,
    nitn=15,
    neval=20000
)

print(f"Total width Γ = {Gamma:.3e}")

# Evaluate at specific Dalitz coordinates
s12_point = 0.6  # GeV²
s23_point = 0.8  # GeV²
prob = p_func(s12_point, s23_point)
print(f"p({s12_point}, {s23_point}) = {prob:.6e}")
```

### Compute Event Likelihoods

```python
from three_body_decay.likelihood import event_likelihoods
import numpy as np

# Generate random points in unit hypercube
n_events = 1000
xs = np.random.uniform(0, 1, (n_events, 2))

# Compute likelihoods
results = event_likelihoods(
    xs,
    M=1.864,
    m1=0.139, m2=0.139, m3=0.139,
    resonances=resonances
)

print(f"Γ = {results['Gamma']:.3e}")
print(f"s12 range: [{results['s12'].min():.3f}, {results['s12'].max():.3f}]")
print(f"s23 range: [{results['s23'].min():.3f}, {results['s23'].max():.3f}]")
print(f"Probability range: [{results['p'].min():.3e}, {results['p'].max():.3e}]")
print(f"Log-prob range: [{results['logp'].min():.2f}, {results['logp'].max():.2f}]")
```

### From Four-Momenta

```python
from three_body_decay.likelihood import event_likelihood_px
from three_body_decay import weighted_generation

# Generate events
events, weights, s12, s23 = weighted_generation(
    M=1.864, m1=0.139, m2=0.139, m3=0.139,
    resonances=resonances,
    n_events=100
)

# Compute likelihood for each event
likelihoods = []
for event in events:
    p_val = event_likelihood_px(
        event['p1'], event['p2'], event['p3'],
        M=1.864, m1=0.139, m2=0.139, m3=0.139,
        resonances=resonances
    )
    likelihoods.append(p_val)

likelihoods = np.array(likelihoods)
print(f"Likelihood range: [{likelihoods.min():.3e}, {likelihoods.max():.3e}]")
```

### Grid Evaluation

```python
# Evaluate likelihood on a grid for visualization
s12_min = (m1 + m2)**2
s12_max = (M - m3)**2
s12_grid = np.linspace(s12_min, s12_max, 100)
s23_grid = np.linspace(s12_min, s12_max, 100)

# Build likelihood function
p_func, Gamma = build_likelihood(
    M, m1, m2, m3, resonances=resonances
)

# Evaluate on grid
prob_grid = np.zeros((100, 100))
for i, s12 in enumerate(s12_grid):
    for j, s23 in enumerate(s23_grid):
        prob_grid[j, i] = p_func(s12, s23)

# Plot (requires matplotlib)
import matplotlib.pyplot as plt
plt.imshow(np.log10(prob_grid), origin='lower', aspect='auto',
           extent=[s12_min, s12_max, s23_grid[0], s23_grid[-1]])
plt.colorbar(label=r'$\log_{10} p(s_{12}, s_{23})$')
plt.xlabel(r'$s_{12}$ [GeV²]')
plt.ylabel(r'$s_{23}$ [GeV²]')
plt.title('Probability Density')
plt.show()
```

## Normalization

The likelihood functions return **normalized probability densities**:

$$
\int p(s_{12}, s_{23}) \, ds_{12} \, ds_{23} = 1
$$

The normalization constant is the total width Γ:

$$
p(s_{12}, s_{23}) = \frac{|M(s_{12}, s_{23})|^2}{\Gamma}
$$

where:

$$
\Gamma = \int |M(s_{12}, s_{23})|^2 \, ds_{12} \, ds_{23}
$$

### Integration Measure

The integration is over the Dalitz plot with measure:

$$
ds_{12} \, ds_{23}
$$

When computing from four-momenta, include the angular phase space:

$$
d\Phi_3 = \frac{1}{(2\pi)^5} \frac{1}{2M} ds_{12} \, ds_{23} \, d\phi \, d\cos\theta
$$

## Unit Hypercube Transform

The `event_likelihoods()` function works with points in [0,1]² which are transformed to physical Dalitz coordinates:

```python
# Input: x = (x₁, x₂) ∈ [0,1]²

# Transform to s₁₂
s12 = s12_min + x₁ * (s12_max - s12_min)

# Compute s₂₃ limits at this s₁₂
s23_min = s23_limits(s12, M, m1, m2, m3)[0]
s23_max = s23_limits(s12, M, m1, m2, m3)[1]

# Transform to s₂₃
s23 = s23_min + x₂ * (s23_max - s23_min)
```

This is the same transformation used in VEGAS integration.

## Log-Probability

For numerical stability, `event_likelihoods()` also returns log-probabilities:

```python
results = event_likelihoods(xs, M, m1, m2, m3, resonances=resonances)

# Linear probability
p = results['p']  # May underflow for small values

# Log-probability (more stable)
logp = results['logp']  # = log(p), handles small values better

# Convert back if needed
p_reconstructed = np.exp(logp)
```

!!! tip "Numerical Stability"
    Use `logp` for:
    - Maximum likelihood fitting
    - Chi-squared calculations
    - Statistical inference
    
    Use `p` for:
    - Visualization (linear scale)
    - Reweighting
    - Probability ratios

## Reweighting Events

You can reweight events from one hypothesis to another:

```python
# Generate events under hypothesis A
events, weights_A, s12, s23 = weighted_generation(
    M, m1, m2, m3, resonances=resonances_A, n_events=10000
)

# Build likelihood under hypothesis B
p_B, Gamma_B = build_likelihood(M, m1, m2, m3, resonances=resonances_B)
p_A, Gamma_A = build_likelihood(M, m1, m2, m3, resonances=resonances_A)

# Compute reweighting factors
weights_B = np.array([
    p_B(s12[i], s23[i]) / p_A(s12[i], s23[i])
    for i in range(len(events))
])

# Normalize
weights_B /= weights_B.sum()

print(f"Effective sample size: {(weights_B.sum())**2 / (weights_B**2).sum():.0f}")
```

## Common Use Cases

### Maximum Likelihood Fit

```python
# Observed data (e.g., from detector)
s12_obs = np.array([...])  # Observed s12 values
s23_obs = np.array([...])  # Observed s23 values

# Hypothesis to test
def neg_log_likelihood(params):
    # Build resonances from params
    resonances = [
        {'mass': params[0], 'width': params[1], 'weight': 1.0,
         'channel': 's12', 'phase': 0.0, 'sign': +1},
    ]
    
    # Build likelihood
    p_func, _ = build_likelihood(M, m1, m2, m3, resonances=resonances)
    
    # Compute negative log-likelihood
    nll = 0
    for i in range(len(s12_obs)):
        p_val = p_func(s12_obs[i], s23_obs[i])
        if p_val > 0:
            nll -= np.log(p_val)
        else:
            nll += 1e10  # Penalty for invalid
    return nll

# Minimize (requires scipy)
from scipy.optimize import minimize
result = minimize(neg_log_likelihood, x0=[0.77, 0.15])
print(f"Best fit: mass={result.x[0]:.3f}, width={result.x[1]:.3f}")
```

### Goodness-of-Fit Test

```python
# Generated events
events, weights, s12_gen, s23_gen = weighted_generation(
    M, m1, m2, m3, resonances=resonances, n_events=10000
)

# Build likelihood
p_func, Gamma = build_likelihood(M, m1, m2, m3, resonances=resonances)

# Compute expected probabilities
p_expected = np.array([p_func(s12_gen[i], s23_gen[i]) for i in range(len(events))])

# Compare with weights (should be proportional)
correlation = np.corrcoef(weights, p_expected)[0, 1]
print(f"Correlation: {correlation:.4f}")  # Should be ~1.0
```

## See Also

- [Generator](generator.md): Generate events with weights
- [Plotter](plotter.md): Visualize probability densities
- [Integrator](integrator.md): Compute amplitudes
