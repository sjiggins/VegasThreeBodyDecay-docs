# Example: Multiple Resonances

This example demonstrates how to work with arbitrary numbers of resonances in the isobar model.

## Overview

The package supports:

- ✅ Any number of resonances (2, 3, 4, ...)
- ✅ Each resonance in a specific channel (s₁₂, s₂₃, or s₁₃)
- ✅ Individual phases and coupling strengths
- ✅ Complex interference patterns

## Running the Example

```bash
# From package root
python examples/demo_n_resonances.py

# Or with uv
uv run python examples/demo_n_resonances.py
```

This creates several visualizations showing resonance structure and interference patterns.

## Example 1: Two Resonances

The simplest non-trivial case with two resonances:

```python
from three_body_decay import weighted_generation
import numpy as np

# Physical parameters (D⁰ → π⁺π⁻π⁰)
M = 1.864
m1, m2, m3 = 0.139, 0.139, 0.139

# Two resonances with 90° phase difference
resonances = [
    {
        'mass': 0.770,      # ρ(770) in s₁₂
        'width': 0.150,
        'weight': 1.0,
        'channel': 's12',
        'phase': 0.0,
        'sign': +1
    },
    {
        'mass': 0.980,      # f₀(980) in s₂₃
        'width': 0.070,
        'weight': 1.0,
        'channel': 's23',
        'phase': np.pi/2,   # 90° phase
        'sign': +1
    }
]

# Generate events
events, weights, s12, s23 = weighted_generation(
    M, m1, m2, m3,
    resonances=resonances,
    n_events=10000
)
```

### What to Observe

- ✅ Two distinct bands in the Dalitz plot
- ✅ Interference pattern where bands cross
- ✅ Total rate Γ ≈ 180 GeV⁻¹

## Example 2: Three Resonances

Adding a third resonance creates richer structure:

```python
# Three resonances in different channels
resonances = [
    {
        'mass': 0.770,      # ρ(770) in s₁₂
        'width': 0.150,
        'weight': 1.0,
        'channel': 's12',
        'phase': 0.0,
        'sign': +1
    },
    {
        'mass': 0.980,      # f₀(980) in s₂₃
        'width': 0.070,
        'weight': 0.8,      # Slightly weaker
        'channel': 's23',
        'phase': np.pi/4,
        'sign': +1
    },
    {
        'mass': 1.270,      # f₁(1270) in s₁₃
        'width': 0.185,
        'weight': 0.5,      # Weakest
        'channel': 's13',
        'phase': np.pi/2,
        'sign': +1
    }
]

events, weights, s12, s23 = weighted_generation(
    M, m1, m2, m3,
    resonances=resonances,
    n_events=10000,
    nitn=15,
    neval=20000
)
```

### What to Observe

- ✅ Three overlapping resonance bands
- ✅ Complex three-way interference
- ✅ Hot spots where all three overlap
- ✅ Modified phase space distribution

## Example 3: Four Resonances

Maximum complexity with four resonances:

```python
# Four resonances including multiple in same channel
resonances = [
    {
        'mass': 0.770,      # ρ(770) in s₁₂
        'width': 0.150,
        'weight': 1.0,
        'channel': 's12',
        'phase': 0.0,
        'sign': +1
    },
    {
        'mass': 1.270,      # ρ(1270) also in s₁₂!
        'width': 0.185,
        'weight': 0.7,
        'channel': 's12',   # Same channel
        'phase': np.pi/3,
        'sign': +1
    },
    {
        'mass': 0.980,      # f₀(980) in s₂₃
        'width': 0.070,
        'weight': 0.8,
        'channel': 's23',
        'phase': np.pi/4,
        'sign': +1
    },
    {
        'mass': 1.450,      # f₀(1450) in s₁₃
        'width': 0.260,
        'weight': 0.4,
        'channel': 's13',
        'phase': np.pi/2,
        'sign': +1
    }
]

events, weights, s12, s23 = weighted_generation(
    M, m1, m2, m3,
    resonances=resonances,
    n_events=10000,
    nitn=20,      # Need more iterations!
    neval=30000   # And more evaluations!
)
```

### What to Observe

- ✅ Very complex interference structure
- ✅ Two resonances in s₁₂ create double-peak structure
- ✅ Multiple hot spots from 4-way interference
- ✅ Requires longer integration time (increase nitn/neval)

## Phase Scan

Exploring the effect of relative phase:

```python
import matplotlib.pyplot as plt
from three_body_decay.plotter import plot_dalitz_density

# Loop over phases
phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, phi in enumerate(phases):
    resonances = [
        {'mass': 0.770, 'width': 0.150, 'weight': 1.0,
         'channel': 's12', 'phase': 0.0, 'sign': +1},
        {'mass': 0.980, 'width': 0.070, 'weight': 1.0,
         'channel': 's23', 'phase': phi, 'sign': +1},  # Variable phase
    ]
    
    ax = axes.flatten()[idx]
    
    # Plot density at this phase
    plot_dalitz_density(
        M, m1, m2, m3,
        Gamma=180.0,  # Approximate
        resonances=resonances,
        output_path=None,  # Don't save
        ax=ax  # Plot on this axis
    )
    
    ax.set_title(f'φ = {phi:.2f} rad ({phi*180/np.pi:.0f}°)')

axes[-1].axis('off')  # Hide last subplot
plt.tight_layout()
plt.savefig('phase_scan.png', dpi=300)
plt.show()
```

### What to Observe

- ✅ φ = 0: Constructive interference, bright spots
- ✅ φ = π/2: Partial interference, moderate intensity
- ✅ φ = π: Maximum destructive interference (but still |M|² ≥ 0)

## Visualizing Individual Contributions

See how each resonance contributes:

```python
from three_body_decay.integrator import amplitude_squared, s23_limits, kinematic_constraint

# Setup grid
s12_min, s12_max = (m1 + m2)**2, (M - m3)**2
s12_vals = np.linspace(s12_min, s12_max, 100)
s23_vals = np.linspace(0, s12_max, 100)

# Compute total and individual contributions
Z_total = np.zeros((100, 100))
Z_individual = []

for res in resonances:
    Z_res = np.zeros((100, 100))
    single_res = [res]  # Single resonance
    
    for i, s12 in enumerate(s12_vals):
        s23_min, s23_max = s23_limits(s12, M, m1, m2, m3)
        for j, s23 in enumerate(s23_vals):
            if s23_min <= s23 <= s23_max:
                s13 = kinematic_constraint(M, m1, m2, m3, s12, s23)
                
                # Individual
                Z_res[j, i] = amplitude_squared(s12, s23, s13=s13, resonances=single_res)
                
                # Total
                Z_total[j, i] = amplitude_squared(s12, s23, s13=s13, resonances=resonances)
            else:
                Z_res[j, i] = np.nan
                Z_total[j, i] = np.nan
    
    Z_individual.append(Z_res)

# Plot
n_res = len(resonances)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (Z_res, res) in enumerate(zip(Z_individual[:3], resonances[:3])):
    ax = axes.flatten()[idx]
    im = ax.imshow(Z_res, origin='lower', aspect='auto', cmap='viridis',
                   extent=[s12_min, s12_max, s23_vals[0], s23_vals[-1]])
    ax.set_title(f"Resonance {idx+1}: m={res['mass']:.3f} GeV")
    plt.colorbar(im, ax=ax)

# Total
ax = axes.flatten()[-1]
im = ax.imshow(Z_total, origin='lower', aspect='auto', cmap='viridis',
               extent=[s12_min, s12_max, s23_vals[0], s23_vals[-1]])
ax.set_title(f'Total ({n_res} resonances)')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('individual_contributions.png', dpi=300)
plt.show()
```

## Common Resonances

Here are some common resonances used in practice:

### ρ Family (I=1, Vector)

```python
rho_770 = {
    'mass': 0.770, 'width': 0.150, 'weight': 1.0,
    'channel': 's12', 'phase': 0.0, 'sign': +1
}

rho_1270 = {
    'mass': 1.270, 'width': 0.185, 'weight': 0.5,
    'channel': 's12', 'phase': 0.0, 'sign': +1
}

rho_1450 = {
    'mass': 1.465, 'width': 0.400, 'weight': 0.3,
    'channel': 's12', 'phase': 0.0, 'sign': +1
}
```

### f₀ Family (I=0, Scalar)

```python
f0_500 = {  # σ/f₀(500) - very broad!
    'mass': 0.500, 'width': 0.400, 'weight': 0.5,
    'channel': 's23', 'phase': 0.0, 'sign': +1
}

f0_980 = {
    'mass': 0.980, 'width': 0.070, 'weight': 1.0,
    'channel': 's23', 'phase': 0.0, 'sign': +1
}

f0_1370 = {
    'mass': 1.350, 'width': 0.350, 'weight': 0.4,
    'channel': 's23', 'phase': 0.0, 'sign': +1
}
```

### f₁ and f₂ Families

```python
f1_1270 = {
    'mass': 1.281, 'width': 0.023, 'weight': 0.3,
    'channel': 's13', 'phase': 0.0, 'sign': +1
}

f2_1270 = {
    'mass': 1.275, 'width': 0.185, 'weight': 0.4,
    'channel': 's13', 'phase': 0.0, 'sign': +1
}
```

## Tips for Multiple Resonances

### 1. Start Simple

```python
# ❌ Don't start with 5 resonances
resonances = [res1, res2, res3, res4, res5]  # Too complex!

# ✓ Start with 2, verify, then add more
resonances = [res1, res2]  # Test first
# ... verify this works ...
resonances.append(res3)  # Add third
# ... verify ...
```

### 2. Increase VEGAS Parameters

```python
# For N resonances, use roughly:
n_resonances = len(resonances)

nitn = 10 + 2 * n_resonances   # 10 + 2N
neval = 10000 + 5000 * n_resonances  # 10k + 5kN

# Examples:
# N=2: nitn=14, neval=20000
# N=3: nitn=16, neval=25000  
# N=4: nitn=18, neval=30000
```

### 3. Balance Weights

```python
# ❌ One resonance dominates
resonances = [
    {'weight': 1.0, ...},  # Strong
    {'weight': 0.01, ...}, # Too weak - won't see it
]

# ✓ Keep weights within 10x of each other
resonances = [
    {'weight': 1.0, ...},   # Primary
    {'weight': 0.5, ...},   # Secondary
    {'weight': 0.3, ...},   # Tertiary
]
```

### 4. Monitor Integration

```python
# The integrator prints:
# Total width Γ = 1.823e+02

# Sanity checks:
# - Γ should be positive
# - Γ should be O(100-1000) for typical configs
# - If Γ is very small or large, check your resonances
```

## Full Example Script

The complete example is in `examples/demo_n_resonances.py`. It includes:

1. **Two resonances** - Basic interference
2. **Three resonances** - Three-way interference
3. **Four resonances** - Maximum complexity
4. **Phase scan** - Effect of relative phases
5. **Visualization** - Individual contributions

Run it to see all examples!

## Next Steps

- Try [Negative Weights Example](negative_weights.md) for destructive interference
- See [API: Generator](../api/generator.md) for detailed parameter documentation
- Read [Physics Background](../physics.md) for theoretical details

## Further Reading

- PDG Resonance Listings: [pdg.lbl.gov](https://pdg.lbl.gov/)
- Isobar Model: Chung et al., Ann.Phys. 4 (1995) 404-430
- VEGAS Algorithm: Lepage, J.Comput.Phys. 27 (1978) 192
