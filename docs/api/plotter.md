# Plotter

Visualization module for Dalitz plots and event distributions.

## Overview

The `plotter` module provides comprehensive visualization tools:

**Advanced Dalitz Functions:**

- **`plot_dalitz_density()`**: 2D probability density heatmap
- **`plot_dalitz_with_marginals()`**: Multi-panel with marginal distributions
- **`plot_dalitz_density_overlay()`**: Density with event scatter overlay
- **`plot_negative_weight_regions()`**: Signed amplitude visualization

**Basic Plotting Functions:**

- **`dalitz()`**: Quick seaborn jointplot
- **`plot_four_momenta()`**: Four-momentum component histograms
- **`corner_plot()`**: Corner plot matrix

---

## Advanced Dalitz Functions

::: three_body_decay.plotter.plot_dalitz_density
    options:
      show_source: false

---

::: three_body_decay.plotter.plot_dalitz_with_marginals
    options:
      show_source: false

---

::: three_body_decay.plotter.plot_dalitz_density_overlay
    options:
      show_source: false

---

::: three_body_decay.plotter.plot_negative_weight_regions
    options:
      show_source: false

---

## Basic Plotting Functions

::: three_body_decay.plotter.dalitz
    options:
      show_source: false

---

::: three_body_decay.plotter.plot_four_momenta
    options:
      show_source: false

---

::: three_body_decay.plotter.corner_plot
    options:
      show_source: false

## Usage Examples

### Quick Dalitz Plot

```python
from three_body_decay import weighted_generation
from three_body_decay.plotter import dalitz

# Generate events
events, weights, s12, s23 = weighted_generation(
    M=1.864, m1=0.139, m2=0.139, m3=0.139,
    n_events=10000
)

# Quick plot
dalitz(s12, s23, weights=weights, output_path='dalitz.png')
```

### Probability Density Heatmap

```python
from three_body_decay.plotter import plot_dalitz_density

# Create probability density visualization
plot_dalitz_density(
    M=1.864,
    m1=0.139, m2=0.139, m3=0.139,
    Gamma=182.5,  # From integration
    resonances=my_resonances,
    phi=0.0,
    n_bins=200,
    output_path='density_heatmap.png'
)
```

### With Marginals

```python
from three_body_decay.plotter import plot_dalitz_with_marginals
from three_body_decay.likelihood import event_likelihoods
import numpy as np

# Generate points
xs = np.random.uniform(0, 1, (1000, 2))
results = event_likelihoods(
    xs, M=1.864, m1=0.139, m2=0.139, m3=0.139,
    resonances=my_resonances
)

# Plot with marginal distributions
plot_dalitz_with_marginals(
    M=1.864,
    m1=0.139, m2=0.139, m3=0.139,
    Gamma=results['Gamma'],
    resonances=my_resonances,
    events=results,  # Pass the full results dict
    n_bins=200,
    output_path='dalitz_with_marginals.png'
)
```

### Negative Weight Regions

```python
from three_body_decay.plotter import plot_negative_weight_regions

# Visualize signed amplitudes
resonances = [
    {'mass': 0.770, 'width': 0.150, 'weight': 1.0,
     'channel': 's12', 'phase': 0.0, 'sign': +1},
    {'mass': 0.980, 'width': 0.070, 'weight': 0.5,
     'channel': 's23', 'phase': np.pi/2, 'sign': -1},  # Destructive
]

plot_negative_weight_regions(
    M=1.864,
    m1=0.139, m2=0.139, m3=0.139,
    resonances=resonances,
    n_bins=200,
    output_path='negative_regions.png'
)
```

### Four-Momentum Distributions

```python
from three_body_decay.plotter import plot_four_momenta
import numpy as np

# Extract four-momentum components
p1_array = np.array([e['p1'] for e in events])

# Plot energy distribution
plot_four_momenta(
    p1_array[:, 0],  # Energy component
    weights=weights,
    x_title='E₁ [GeV]',
    y_title='dN/dE',
    output_path='p1_energy.png'
)

# Plot momentum components
for i, label in enumerate(['px', 'py', 'pz']):
    plot_four_momenta(
        p1_array[:, i+1],
        weights=weights,
        x_title=f'{label}₁ [GeV]',
        output_path=f'p1_{label}.png'
    )
```

### Corner Plot

```python
from three_body_decay.plotter import corner_plot

# Convert events to array
p1_array = np.array([e['p1'] for e in events])

# Create corner plot
corner_plot(
    p1_array,
    weights=weights,
    output_path='p1_corner.png'
)

# Corner plot shows all pairwise correlations:
# - Diagonal: 1D histograms
# - Off-diagonal: 2D histograms
```

## Plot Customization

### Custom Colormaps

```python
# Available colormaps for different purposes

# For probability density (light = high probability)
plot_dalitz_density(..., cmap='viridis')  # Default
plot_dalitz_density(..., cmap='plasma')
plot_dalitz_density(..., cmap='hot')

# For signed amplitudes (blue = negative, red = positive)
plot_negative_weight_regions(...)  # Uses RdBu_r automatically
```

### Resolution Control

```python
# Low resolution (fast, for testing)
plot_dalitz_density(..., n_bins=50, dpi=100)

# Medium resolution (default)
plot_dalitz_density(..., n_bins=200, dpi=150)

# High resolution (publication quality)
plot_dalitz_density(..., n_bins=500, dpi=300)
```

### Multiple Plots

```python
import matplotlib.pyplot as plt

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot different configurations
for idx, phi in enumerate([0, np.pi/4, np.pi/2, 3*np.pi/4]):
    ax = axes.flatten()[idx]
    
    # Compute density at this phase
    # ... (compute Z grid)
    
    im = ax.imshow(Z, ...)
    ax.set_title(f'φ = {phi:.2f} rad')
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('phase_comparison.png', dpi=300)
plt.show()
```

## Output Formats

All plotting functions support multiple output formats:

```python
# PNG (default, good for web)
dalitz(..., output_path='plot.png')

# PDF (vector graphics, publication quality)
dalitz(..., output_path='plot.pdf')

# SVG (vector graphics, editable)
dalitz(..., output_path='plot.svg')

# EPS (vector graphics, legacy journals)
dalitz(..., output_path='plot.eps')

# High-DPI PNG
dalitz(..., output_path='plot.png', dpi=300)
```

## Interactive Plotting

For interactive exploration (Jupyter notebooks):

```python
import matplotlib.pyplot as plt

# Don't save to file
plot_dalitz_density(..., output_path=None)

# Show interactive plot
plt.show()

# Enable interactive mode
plt.ion()
```

## Batch Processing

Generate multiple plots programmatically:

```python
# Loop over resonance configurations
configs = [
    {'name': 'rho_only', 'resonances': [rho]},
    {'name': 'f0_only', 'resonances': [f0]},
    {'name': 'both', 'resonances': [rho, f0]},
]

for config in configs:
    # Generate events
    events, weights, s12, s23 = weighted_generation(
        M, m1, m2, m3,
        resonances=config['resonances'],
        n_events=10000
    )
    
    # Plot
    dalitz(
        s12, s23, weights=weights,
        output_path=f"dalitz_{config['name']}.png"
    )
    
    print(f"✓ Created plot for {config['name']}")
```

## Plotting Tips

### Memory Efficiency

For large datasets:

```python
# Memory intensive
dalitz(s12, s23, weights=weights, bins=(1000, 1000))

# ✓ Memory efficient
dalitz(s12, s23, weights=weights, bins=(200, 200))  # Default

# Or subsample events
n_plot = 10000
idx = np.random.choice(len(s12), size=n_plot, replace=False)
dalitz(s12[idx], s23[idx], weights=weights[idx])
```

### Handling Negative Weights

```python
# For basic dalitz plot, use absolute weights
dalitz(s12, s23, weights=np.abs(weights))

# For signed visualization, use specialized function
plot_negative_weight_regions(
    M, m1, m2, m3, resonances=resonances
)
```

### Color Scales

```python
# Linear scale (default)
plot_dalitz_density(..., log_scale=False)

# Log scale (better for wide dynamic range)
plot_dalitz_density(..., log_scale=True)

# The function automatically handles this, showing log₁₀(p)
```

## Common Issues

### Plot Not Showing

```python
# Forgot to show plot
plot_dalitz_density(...)

# Add plt.show() if not saving
plot_dalitz_density(..., output_path=None)
plt.show()

# Or save to file
plot_dalitz_density(..., output_path='plot.png')
```

### Empty Plots

```python
# Check if weights are all zero
if np.allclose(weights, 0):
    print("️ All weights are zero!")

# Check if data is in range
print(f"s12 range: [{s12.min():.3f}, {s12.max():.3f}]")
print(f"s23 range: [{s23.min():.3f}, {s23.max():.3f}]")
```

### Slow Rendering

```python
# Reduce resolution for speed
plot_dalitz_density(..., n_bins=100)  # Instead of 500

# Use lower DPI
dalitz(..., dpi=100)  # Instead of 300

# Plot fewer events
n_subsample = 10000
idx = np.random.choice(len(events), n_subsample)
dalitz(s12[idx], s23[idx], weights[idx])
```

## See Also

- [Generator](generator.md): Generate events to plot
- [Likelihood](likelihood.md): Compute probability densities
- [Examples: N-Resonances](../examples/n_resonances.md): Visualization examples
- [Examples: Negative Weights](../examples/negative_weights.md): Negative weight visualization
