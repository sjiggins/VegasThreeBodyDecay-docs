# Three-Body Decay Monte Carlo

A Python package for Monte Carlo event generation in three-body particle decays with support for multiple resonances, quantum interference, and negative weights.

## Features

✨ **Flexible Resonance Configuration**
- Arbitrary number of Breit-Wigner resonances
- Individual phases and coupling strengths
- Support for s₁₂, s₂₃, and s₁₃ channels

⚡ **Advanced Physics**
- VEGAS adaptive importance sampling
- Negative weights from destructive interference
- Proper treatment of quantum mechanical interference

📊 **Rich Visualization**
- Dalitz plot generation with marginals
- Four-momentum distributions
- Negative weight region analysis
- Corner plots for correlations

## Quick Start

### Installation

```bash
# With pip
pip install three-body-decay

# With uv
uv pip install three-body-decay

# From source
git clone https://github.com/yourusername/three-body-decay
cd three-body-decay
pip install -e .
```

### Basic Usage

```python
from three_body_decay import weighted_generation
import numpy as np

# Generate events for D⁰ → π⁺π⁻π⁰
events, weights, s12, s23 = weighted_generation(
    M=1.864,      # D⁰ mass (GeV)
    m1=0.139,     # π⁺ mass (GeV)
    m2=0.139,     # π⁻ mass (GeV)
    m3=0.139,     # π⁰ mass (GeV)
    n_events=10000
)

print(f"Generated {len(events)} events")
print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
```

### Custom Resonances

```python
# Define custom resonance configuration
my_resonances = [
    {
        'mass': 0.770,      # ρ(770) mass in GeV
        'width': 0.150,     # Width in GeV
        'weight': 1.0,      # Coupling strength
        'channel': 's12',   # Couples to π⁺π⁻
        'phase': 0.0,       # Phase in radians
        'sign': +1          # Constructive interference
    },
    {
        'mass': 0.980,      # f₀(980) mass
        'width': 0.070,     # Width
        'weight': 0.5,      # Weaker coupling
        'channel': 's23',   # Couples to π⁻π⁰
        'phase': np.pi/2,   # 90° phase
        'sign': -1          # Destructive interference
    }
]

events, weights, s12, s23 = weighted_generation(
    M=1.864, m1=0.139, m2=0.139, m3=0.139,
    resonances=my_resonances,
    n_events=10000
)
```

### Plotting

```python
from three_body_decay.plotter import dalitz, plot_dalitz_with_marginals

# Quick Dalitz plot
dalitz(s12, s23, weights=weights, output_path='dalitz.png')

# Advanced plot with probability density
plot_dalitz_with_marginals(
    M=1.864, m1=0.139, m2=0.139, m3=0.139,
    Gamma=182.5,  # Total width from integration
    resonances=my_resonances,
    events={'s12': s12, 's23': s23},
    n_bins=200
)
```

## Key Concepts

### Negative Weights

This package properly handles **negative Monte Carlo weights** that arise from destructive quantum interference:

- When resonances have `sign=-1`, they contribute destructively
- Events in destructive regions get negative weights
- Total integrated rate (Γ) must remain positive
- Typical configurations yield 20-40% negative weight events

!!! warning "Physical Constraint"
    While individual events can have negative weights, the **total sum must be positive**. The generator will warn you if your configuration is unphysical (Γ < 0).

### VEGAS Importance Sampling

The package uses [VEGAS](https://vegas.readthedocs.io/) adaptive Monte Carlo:

1. **Adaptation phase**: VEGAS learns where amplitude is large
2. **Sampling phase**: Events generated preferentially in important regions
3. **Re-evaluation**: Signed integrand re-evaluated for correct negative weights

## Physics Applications

This package is useful for:

- **Amplitude analysis** in particle physics
- **Dalitz plot studies** of three-body decays
- **Interference effects** between resonances
- **NLO corrections** with negative weights
- **Machine learning** training data generation

## Examples

See the [Examples](examples/n_resonances.md) section for detailed walkthroughs:

- [Multiple Resonances](examples/n_resonances.md): Working with 2, 3, 4+ resonances
- [Negative Weights](examples/negative_weights.md): Handling destructive interference

## API Reference

Complete API documentation:

- [Generator](api/generator.md): Event generation functions
- [Likelihood](api/likelihood.md): Probability calculations
- [Plotter](api/plotter.md): Visualization tools
- [Integrator](api/integrator.md): Physics calculations

## Command-Line Interface

The package includes a CLI for quick generation:

```bash
# Generate sample dataset with default parameters
three-body-decay

# This creates:
# - Dalitz plots
# - Four-momentum distributions
# - Corner plots
# - particle_data/*.npy files
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{three_body_decay,
  title = {Three-Body Decay Monte Carlo},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/three-body-decay}
}
```

## License

MIT License - see [LICENSE](https://github.com/yourusername/three-body-decay/blob/main/LICENSE) file.

## Contributing

Contributions welcome! Please see the [GitHub repository](https://github.com/yourusername/three-body-decay) for:

- Issue tracker
- Pull request guidelines
- Development setup

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/three-body-decay/issues)
- **Documentation**: [https://yourusername.github.io/three-body-decay/](https://yourusername.github.io/three-body-decay/)
- **PyPI**: [https://pypi.org/project/three-body-decay/](https://pypi.org/project/three-body-decay/)
