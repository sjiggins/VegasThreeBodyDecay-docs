# Integrator

Physics calculation module for three-body decay amplitudes.

## Overview

The `integrator` module implements the core physics calculations:

- **`amplitude_squared()`**: Compute |M|² for given Dalitz coordinates
- **`BW()`**: Breit-Wigner resonance function
- **`integrand()`**: VEGAS integrand with Jacobian
- **`s23_limits()`**: Kinematic boundaries for s₂₃
- **`kinematic_constraint()`**: Compute s₁₃ from s₁₂ and s₂₃
- **`two_body_momentum()`**: Two-body decay momentum
- **`get_default_resonances()`**: Default resonance configuration

These form the foundation for event generation and likelihood calculations.

---

::: three_body_decay.integrator.amplitude_squared
    options:
      show_source: true

---

::: three_body_decay.integrator.BW
    options:
      show_source: true

---

::: three_body_decay.integrator.integrand
    options:
      show_source: false

---

::: three_body_decay.integrator.s23_limits
    options:
      show_source: false

---

::: three_body_decay.integrator.kinematic_constraint
    options:
      show_source: false

---

::: three_body_decay.integrator.two_body_momentum
    options:
      show_source: false

---

::: three_body_decay.integrator.get_default_resonances
    options:
      show_source: false

## Usage Examples

### Compute Amplitude

```python
from three_body_decay.integrator import amplitude_squared, kinematic_constraint

# Dalitz coordinates
s12 = 0.6  # (GeV)²
s23 = 0.8  # (GeV)²

# Parent and daughter masses
M = 1.864
m1, m2, m3 = 0.139, 0.139, 0.139

# Compute s13 from kinematics
s13 = kinematic_constraint(M, m1, m2, m3, s12, s23)

# Resonance configuration
resonances = [
    {'mass': 0.770, 'width': 0.150, 'weight': 1.0,
     'channel': 's12', 'phase': 0.0, 'sign': +1},
]

# Compute squared amplitude
amp2 = amplitude_squared(s12, s23, s13=s13, resonances=resonances)
print(f"|M|² = {amp2:.6e}")
```

### Breit-Wigner Resonance

```python
from three_body_decay.integrator import BW
import numpy as np
import matplotlib.pyplot as plt

# Resonance parameters
m_res = 0.770  # ρ(770) mass
gamma = 0.150  # Width

# Energy scan
s_vals = np.linspace(0.5, 1.5, 1000)

# Compute BW shape
bw_vals = [BW(s, m_res, gamma) for s in s_vals]

# Plot
plt.plot(s_vals, bw_vals)
plt.xlabel(r'$s$ [GeV²]')
plt.ylabel(r'BW$(s)$')
plt.title(f'Breit-Wigner: m={m_res} GeV, Γ={gamma} GeV')
plt.grid(True, alpha=0.3)
plt.show()
```

### Kinematic Boundaries

```python
from three_body_decay.integrator import s23_limits

# Physical parameters
M = 1.864
m1, m2, m3 = 0.139, 0.139, 0.139

# s12 values to check
s12_min = (m1 + m2)**2
s12_max = (M - m3)**2

s12_test = np.linspace(s12_min, s12_max, 100)

# Find s23 boundaries at each s12
s23_boundaries = []
for s12 in s12_test:
    s23_min, s23_max = s23_limits(s12, M, m1, m2, m3)
    s23_boundaries.append((s23_min, s23_max))

# Plot Dalitz boundary
import matplotlib.pyplot as plt
s23_min_arr = [b[0] for b in s23_boundaries]
s23_max_arr = [b[1] for b in s23_boundaries]

plt.fill_between(s12_test, s23_min_arr, s23_max_arr, alpha=0.3)
plt.xlabel(r'$s_{12}$ [GeV²]')
plt.ylabel(r'$s_{23}$ [GeV²]')
plt.title('Dalitz Plot Boundary')
plt.grid(True, alpha=0.3)
plt.show()
```

### Scan Dalitz Plot

```python
from three_body_decay.integrator import amplitude_squared, s23_limits, kinematic_constraint

# Setup grid
s12_vals = np.linspace(s12_min, s12_max, 100)
s23_vals = np.linspace(s12_min, s12_max, 100)

# Compute amplitude on grid
Z = np.zeros((100, 100))
for i, s12 in enumerate(s12_vals):
    s23_min, s23_max = s23_limits(s12, M, m1, m2, m3)
    for j, s23 in enumerate(s23_vals):
        if s23_min <= s23 <= s23_max:
            s13 = kinematic_constraint(M, m1, m2, m3, s12, s23)
            Z[j, i] = amplitude_squared(s12, s23, s13=s13, resonances=resonances)
        else:
            Z[j, i] = np.nan

# Plot
plt.imshow(Z, origin='lower', aspect='auto',
           extent=[s12_min, s12_max, s23_vals[0], s23_vals[-1]],
           cmap='viridis')
plt.colorbar(label=r'$|M|^2$')
plt.xlabel(r'$s_{12}$ [GeV²]')
plt.ylabel(r'$s_{23}$ [GeV²]')
plt.show()
```

### Two-Body Momentum

```python
from three_body_decay.integrator import two_body_momentum

# Decay: M → m1 + m2
M_parent = 1.864
m1_daughter = 0.770  # ρ(770)
m2_daughter = 0.139  # π

# Compute momentum in parent rest frame
p = two_body_momentum(M_parent, m1_daughter, m2_daughter)
print(f"Momentum: {p:.4f} GeV")

# Energy of daughter 1
E1 = np.sqrt(m1_daughter**2 + p**2)
E2 = np.sqrt(m2_daughter**2 + p**2)
print(f"E1 = {E1:.4f} GeV, E2 = {E2:.4f} GeV")
print(f"Total = {E1 + E2:.4f} GeV (should equal {M_parent})")
```

## Physics Formulas

### Isobar Model

The amplitude is a coherent sum over resonances:

$$
M_{\text{total}} = \sum_{i} c_i \, e^{i\phi_i} \, \text{BW}_i(s_i)
$$

where:

- $c_i$ = coupling strength (`weight`)
- $\phi_i$ = relative phase (`phase`)
- $s_i$ = invariant mass squared in channel $i$

The squared amplitude is:

$$
|M_{\text{total}}|^2 = \left| \sum_{i} c_i \, e^{i\phi_i} \, \text{BW}_i(s_i) \right|^2
$$

This naturally includes interference terms!

### Breit-Wigner Function

The relativistic Breit-Wigner:

$$
\text{BW}(s, m_0, \Gamma) = \frac{1}{(s - m_0^2)^2 + m_0^2 \Gamma^2}
$$

where:

- $s$ = invariant mass squared
- $m_0$ = resonance pole mass
- $\Gamma$ = resonance width

### Kinematic Constraint

For three-body decay M → 1 + 2 + 3, the Dalitz variables satisfy:

$$
s_{12} + s_{23} + s_{13} = M^2 + m_1^2 + m_2^2 + m_3^2
$$

Given $s_{12}$ and $s_{23}$, we compute:

$$
s_{13} = M^2 + m_1^2 + m_2^2 + m_3^2 - s_{12} - s_{23}
$$

### Phase Space Boundaries

For fixed $s_{12}$, the allowed range of $s_{23}$ is:

$$
s_{23}^{\min/\max} = m_2^2 + m_3^2 + \frac{1}{2s_{12}} \left[ (s_{12} + m_2^2 - m_1^2)(M^2 + m_3^2 - s_{12}) \mp \lambda^{1/2}(s_{12}, m_1^2, m_2^2) \lambda^{1/2}(M^2, s_{12}, m_3^2) \right]
$$

where $\lambda(a,b,c) = a^2 + b^2 + c^2 - 2ab - 2ac - 2bc$ is the Källén function.

### Two-Body Momentum

In the rest frame of parent M decaying to daughters with masses $m_1$ and $m_2$:

$$
p = \frac{1}{2M} \sqrt{\lambda(M^2, m_1^2, m_2^2)}
$$

## Negative Weights Implementation

For destructive interference, resonances have `sign=-1`:

```python
resonances = [
    {'mass': 0.770, 'width': 0.150, 'weight': 1.0,
     'channel': 's12', 'phase': 0.0, 'sign': +1},  # Constructive
    {'mass': 0.980, 'width': 0.070, 'weight': 0.5,
     'channel': 's23', 'phase': 0.0, 'sign': -1},  # Destructive
]
```

The implementation separates positive and negative contributions:

$$
M_+ = \sum_{i: \text{sign}_i=+1} c_i e^{i\phi_i} \text{BW}_i
$$

$$
M_- = \sum_{i: \text{sign}_i=-1} c_i e^{i\phi_i} \text{BW}_i
$$

The signed amplitude squared is:

$$
\text{signed}|M|^2 = |M_+|^2 - |M_-|^2
$$

!!! note "Cross Terms"
    This formulation neglects the cross term $2\text{Re}(M_+^* M_-)$. For a full treatment including cross terms, all resonances should have `sign=+1` and interference is handled through the phases.

## Default Resonances

```python
from three_body_decay.integrator import get_default_resonances

# Get default configuration (2 resonances)
resonances = get_default_resonances()

for i, res in enumerate(resonances):
    print(f"Resonance {i+1}:")
    print(f"  mass: {res['mass']} GeV")
    print(f"  width: {res['width']} GeV")
    print(f"  channel: {res['channel']}")
    print(f"  phase: {res['phase']} rad")
    print(f"  weight: {res['weight']}")
    print(f"  sign: {res['sign']:+d}")
```

## Resonance Dictionary Format

Each resonance is a dictionary with required keys:

```python
resonance = {
    'mass': 0.770,      # Pole mass (GeV)
    'width': 0.150,     # Width (GeV)
    'weight': 1.0,      # Coupling strength (dimensionless)
    'channel': 's12',   # Channel: 's12', 's23', or 's13'
    'phase': 0.0,       # Relative phase (radians)
    'sign': +1          # +1 (constructive) or -1 (destructive)
}
```

### Channel Mapping

- `'s12'`: Resonance in (particle 1, particle 2) system
- `'s23'`: Resonance in (particle 2, particle 3) system
- `'s13'`: Resonance in (particle 1, particle 3) system

For identical particles, some channels are equivalent by symmetry.

## Performance Tips

### Vectorization

For many points, vectorize calculations:

```python
# ❌ Slow: Loop over points
results = [amplitude_squared(s12[i], s23[i], ...) for i in range(n)]

# ✓ Fast: Vectorize where possible
# (Note: Current implementation requires loop, but Numba/Cython can help)
```

### Caching

For repeated calculations with same resonances:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_amplitude(s12, s23, resonances_tuple):
    # Convert tuple back to list
    resonances = [dict(r) for r in resonances_tuple]
    return amplitude_squared(s12, s23, resonances=resonances)

# Usage (convert list to tuple for hashing)
res_tuple = tuple(tuple(r.items()) for r in resonances)
amp = cached_amplitude(0.6, 0.8, res_tuple)
```

## Common Issues

### Invalid Kinematics

```python
# Check if point is in Dalitz plot
s12 = 0.6
s23 = 0.8

s23_min, s23_max = s23_limits(s12, M, m1, m2, m3)

if s23_min <= s23 <= s23_max:
    # Valid point
    amp2 = amplitude_squared(s12, s23, ...)
else:
    # Outside kinematic boundary
    amp2 = 0.0  # or np.nan
```

### NaN Results

```python
# Check for NaN in amplitude
amp2 = amplitude_squared(s12, s23, ...)
if np.isnan(amp2):
    print("⚠️ NaN amplitude!")
    print(f"  s12={s12}, s23={s23}")
    print(f"  Check kinematics and resonance parameters")
```

## See Also

- [Generator](generator.md): Uses integrator for event generation
- [Likelihood](likelihood.md): Uses integrator for probability
- [Physics Background](../physics.md): Detailed physics explanation
