# Physics Background

A comprehensive guide to the physics implemented in the three-body decay package.

## Three-Body Decay Kinematics

### Decay Process

We consider the decay of a parent particle M into three daughters:

$$
M \to m_1 + m_2 + m_3
$$

In the rest frame of M, the decay is characterized by:

- Parent mass: $M$
- Daughter masses: $m_1, m_2, m_3$
- Total energy-momentum conservation

### Dalitz Variables

The kinematics are described by invariant mass squared variables:

$$
s_{ij} = (p_i + p_j)^2 = (E_i + E_j)^2 - (\vec{p}_i + \vec{p}_j)^2
$$

where $i, j \in \{1, 2, 3\}$ and $i \neq j$.

### Kinematic Constraint

The three Dalitz variables are not independent:

$$
s_{12} + s_{23} + s_{13} = M^2 + m_1^2 + m_2^2 + m_3^2
$$

This reduces the phase space from 3D to 2D. We typically use $(s_{12}, s_{23})$ as independent variables.

### Dalitz Plot Boundaries

For fixed $s_{12}$, the allowed range of $s_{23}$ is:

$$
s_{23}^{\min} \leq s_{23} \leq s_{23}^{\max}
$$

where:

$$
s_{23}^{\min/\max} = m_2^2 + m_3^2 + \frac{1}{2s_{12}} \left[ (s_{12} + m_2^2 - m_1^2)(M^2 + m_3^2 - s_{12}) \mp \lambda^{1/2}(s_{12}, m_1^2, m_2^2) \lambda^{1/2}(M^2, s_{12}, m_3^2) \right]
$$

The Källén function is defined as:

$$
\lambda(a, b, c) = a^2 + b^2 + c^2 - 2ab - 2ac - 2bc
$$

### Overall Boundaries

The $s_{12}$ variable has fixed limits:

$$
(m_1 + m_2)^2 \leq s_{12} \leq (M - m_3)^2
$$

## Isobar Model

### Motivation

Direct three-body matrix elements are complex. The **isobar model** simplifies by:

1. Assuming decay proceeds through intermediate resonances
2. Each resonance decays to two particles
3. Resonances described by Breit-Wigner functions

### Amplitude Structure

The total amplitude is a coherent sum:

$$
M_{\text{total}}(s_{12}, s_{23}, s_{13}) = \sum_i c_i e^{i\phi_i} \, \text{BW}_i(s_i) \, F_i(s_i)
$$

where:

- $c_i$ = coupling constant (strength)
- $\phi_i$ = relative phase
- $\text{BW}_i(s)$ = Breit-Wigner resonance function
- $F_i(s)$ = form factor (often approximated as 1)
- $s_i$ = invariant mass squared in the appropriate channel

### Breit-Wigner Function

The relativistic Breit-Wigner:

$$
\text{BW}(s, m_0, \Gamma) = \frac{1}{(s - m_0^2)^2 + m_0^2 \Gamma^2}
$$

Properties:

- Peaks at $s = m_0^2$ (resonance mass)
- Width at half-maximum: $\Gamma$ (resonance width)
- Integral over all $s$ is $\propto 1/\Gamma$

!!! note "Alternative Parametrizations"
    The package uses this simple form. More sophisticated models use energy-dependent widths: $\Gamma(s) = \Gamma_0 \left(\frac{p(s)}{p(m_0^2)}\right)^{2L+1}$ where $L$ is orbital angular momentum.

### Observable: Squared Amplitude

The experimentally observable quantity is:

$$
\frac{d\Gamma}{ds_{12} \, ds_{23}} \propto |M_{\text{total}}(s_{12}, s_{23}, s_{13})|^2
$$

## Quantum Interference

### Coherent Sum

Because amplitudes add (not probabilities), we get interference:

$$
|M_1 + M_2|^2 = |M_1|^2 + |M_2|^2 + 2\text{Re}(M_1^* M_2)
$$

The cross term $2\text{Re}(M_1^* M_2)$ is the **interference term**.

### Constructive vs Destructive

- **Constructive**: $\phi_1 - \phi_2 = 0$ → $+2|M_1||M_2|$
- **Destructive**: $\phi_1 - \phi_2 = \pi$ → $-2|M_1||M_2|$

### Example: Two Resonances

$$
M_{\text{total}} = c_1 e^{i\phi_1} \text{BW}_1(s_{12}) + c_2 e^{i\phi_2} \text{BW}_2(s_{23})
$$

Squared amplitude:

$$
|M_{\text{total}}|^2 = c_1^2 |\text{BW}_1|^2 + c_2^2 |\text{BW}_2|^2 + 2c_1 c_2 \cos(\phi_2 - \phi_1) \, \text{Re}(\text{BW}_1^* \text{BW}_2)
$$

The last term creates:

- **Hot spots** where both resonances overlap (if $\phi_2 - \phi_1 \approx 0$)
- **Dead zones** where they cancel (if $\phi_2 - \phi_1 \approx \pi$)

## Negative Weights

### Motivation

In certain applications, we want to represent **signed** amplitudes:

1. NLO QCD corrections (can be negative)
2. Background subtraction
3. Importance sampling schemes

### Implementation

We separate resonances by sign:

$$
M_+ = \sum_{i: \text{sign}_i = +1} c_i e^{i\phi_i} \text{BW}_i(s_i)
$$

$$
M_- = \sum_{i: \text{sign}_i = -1} c_i e^{i\phi_i} \text{BW}_i(s_i)
$$

The **signed amplitude squared** is:

$$
\text{signed}|M|^2 = |M_+|^2 - |M_-|^2
$$

### Physical Constraint

Locally, signed$|M|^2$ can be negative, but the total must be positive:

$$
\Gamma = \int_{\text{Dalitz}} \text{signed}|M|^2 \, ds_{12} \, ds_{23} > 0
$$

If $\Gamma < 0$, the configuration is **unphysical** and violates probability conservation.

### Interpretation

Events in negative regions represent:

- **Destructive interference** exceeding constructive locally
- Regions where amplitude from $M_-$ dominates
- Statistical fluctuations that must be accounted for properly

## VEGAS Importance Sampling

### The Problem

Direct sampling from arbitrary $|M|^2$ is inefficient:

- Most of phase space has $|M|^2 \approx 0$
- Resonances create narrow peaks
- Uniform sampling wastes most events

### VEGAS Solution

VEGAS (Peter Lepage, 1978) adaptively learns the function:

1. **Grid adaptation**: Divide space into strata
2. **Weight strata**: More subdivisions where $|M|^2$ is large
3. **Sample**: Generate points preferentially in important regions

### Algorithm

1. **Initialization**: Start with uniform grid
2. **Integration**: Evaluate $|M|^2$ at random points
3. **Adaptation**: Refine grid based on variance
4. **Iteration**: Repeat steps 2-3 for `nitn` iterations
5. **Sampling**: Generate events from adapted grid

### Parameters

- **nitn**: Number of adaptation iterations (more = better adaptation)
- **neval**: Evaluations per iteration (more = finer grid)

Typical values:

- Testing: nitn=10, neval=10000 (~5 seconds)
- Production: nitn=15, neval=20000 (~15 seconds)
- High precision: nitn=30, neval=100000 (~5 minutes)

### Importance Weights

VEGAS returns importance weights $w_{\text{VEGAS}}$, but for negative weights we **re-evaluate**:

$$
w_{\text{final}}(x) = \text{signed}|M|^2(s_{12}(x), s_{23}(x))
$$

This is necessary because VEGAS doesn't capture the **sign** of the amplitude!

## Integration and Normalization

### Total Width

The total decay width is:

$$
\Gamma = \int_{\text{Dalitz}} |M(s_{12}, s_{23})|^2 \, ds_{12} \, ds_{23}
$$

This is computed by VEGAS during the adaptation phase.

### Probability Density

The normalized probability density is:

$$
p(s_{12}, s_{23}) = \frac{|M(s_{12}, s_{23})|^2}{\Gamma}
$$

This satisfies:

$$
\int_{\text{Dalitz}} p(s_{12}, s_{23}) \, ds_{12} \, ds_{23} = 1
$$

### Event Weights

After generation, weights are normalized:

$$
w_i^{\text{final}} = \frac{w_i^{\text{raw}}}{\sum_j |w_j^{\text{raw}}|}
$$

Note the absolute value in the denominator - this preserves the **sign** of individual weights while normalizing the total.

## Phase Space Volume

### Differential Phase Space

The three-body phase space element is:

$$
d\Phi_3 = \frac{1}{(2\pi)^5} \frac{1}{2M} \, ds_{12} \, ds_{23} \, d\Omega
$$

where $d\Omega = d\phi \, d\cos\theta$ is the angular part.

### Integration

For the unpolarized decay rate:

$$
\Gamma = \int d\Phi_3 \, |M|^2 = \frac{1}{(2\pi)^5} \frac{1}{2M} \int ds_{12} \, ds_{23} \, |M|^2 \int d\Omega
$$

If $|M|^2$ doesn't depend on angles:

$$
\Gamma = \frac{4\pi}{(2\pi)^5} \frac{1}{2M} \int ds_{12} \, ds_{23} \, |M|^2 = \frac{1}{32\pi^3 M} \int ds_{12} \, ds_{23} \, |M|^2
$$

## Applications

### Amplitude Analysis

Given experimental data $(s_{12}^i, s_{23}^i)$, fit resonance parameters:

$$
\mathcal{L} = \prod_i p(s_{12}^i, s_{23}^i | \{m_j, \Gamma_j, c_j, \phi_j\})
$$

Maximize likelihood to extract:

- Resonance masses and widths
- Coupling constants
- Relative phases

### Machine Learning

Generate training data for ML models:

```python
# Generate large sample
events, weights, s12, s23 = weighted_generation(
    M, m1, m2, m3, resonances=config, n_events=1000000
)

# Train neural network
model = train_network(events, weights)

# Use for fast event generation
new_events = model.generate(n_events=1000000)
```

### Detector Simulation

Include acceptance effects:

```python
# Generate physics events
events, weights, _, _ = weighted_generation(...)

# Apply detector acceptance
for i, event in enumerate(events):
    if detector_accepts(event):
        accepted_events.append(event)
        accepted_weights.append(weights[i])

# Compute efficiency
eff = sum(accepted_weights) / sum(weights)
```

## Limitations and Extensions

### Current Limitations

1. **Constant width**: Uses constant $\Gamma$ (could use $\Gamma(s)$)
2. **No spin**: Doesn't include spin projections
3. **No form factors**: Could add form factors $F(s)$
4. **Relative BW**: Could use alternative line shapes

### Possible Extensions

1. **Energy-dependent widths**:
   $$
   \Gamma(s) = \Gamma_0 \left(\frac{p(s)}{p(m_0^2)}\right)^{2L+1}
   $$

2. **Spin projections**: Include angular distributions
   $$
   |M|^2 = \sum_{m_s} |M_{m_s}|^2 \, Y_{LM}(\theta, \phi)
   $$

3. **K-matrix formalism**: For coupled channels

4. **Barrier factors**: For centrifugal effects

## References

### Textbooks

- **Particle Physics**: Griffiths, "Introduction to Elementary Particles"
- **QFT**: Peskin & Schroeder, "An Introduction to Quantum Field Theory"
- **Kinematics**: Byckling & Kajantie, "Particle Kinematics"

### Papers

- **Isobar Model**: Chung et al., Ann.Phys. 4 (1995) 404-430
- **VEGAS**: Lepage, J.Comput.Phys. 27 (1978) 192-203
- **Dalitz Plots**: Dalitz, Phil.Mag. 44 (1953) 1068-1080
- **Negative Weights**: NNPDF, arXiv:0912.2276

### Data

- **PDG**: Particle Data Group, pdg.lbl.gov
- **BESIII**: Three-body D meson decays
- **LHCb**: B meson Dalitz studies

## Mathematical Appendix

### Källén Function

The Källén lambda function:

$$
\lambda(a, b, c) = a^2 + b^2 + c^2 - 2ab - 2ac - 2bc
$$

Useful identity:

$$
\lambda(a, b, c) = [a - (\sqrt{b} + \sqrt{c})^2][a - (\sqrt{b} - \sqrt{c})^2]
$$

### Two-Body Momentum

For $M \to m_1 + m_2$:

$$
p = \frac{1}{2M} \sqrt{\lambda(M^2, m_1^2, m_2^2)}
$$

Alternatively:

$$
p = \frac{1}{2M} \sqrt{[M^2 - (m_1 + m_2)^2][M^2 - (m_1 - m_2)^2]}
$$

### Lorentz Boost

To boost four-momentum $p = (E, \vec{p})$ with velocity $\vec{\beta}$:

$$
E' = \gamma(E - \vec{\beta} \cdot \vec{p})
$$

$$
\vec{p}' = \vec{p} + \frac{\gamma - 1}{|\vec{\beta}|^2}(\vec{\beta} \cdot \vec{p})\vec{\beta} - \gamma E \vec{\beta}
$$

where $\gamma = 1/\sqrt{1 - |\vec{\beta}|^2}$.

## Glossary

- **Dalitz plot**: 2D plot of $(s_{12}, s_{23})$ showing decay kinematics
- **Isobar model**: Decay model using intermediate resonances
- **Breit-Wigner**: Standard resonance line shape
- **VEGAS**: Adaptive Monte Carlo integration algorithm
- **Importance sampling**: Sample where function is large
- **Negative weights**: Events with negative Monte Carlo weights
- **Signed amplitude**: $|M_+|^2 - |M_-|^2$ allowing local negativity
- **Phase space**: Allowed kinematic region
- **Källén function**: $\lambda(a,b,c)$ used in kinematics

## Further Study

### Recommended Order

1. **Basic QM**: Interference and amplitude addition
2. **Particle Physics**: Resonances and decays
3. **Monte Carlo**: VEGAS and importance sampling
4. **Dalitz Plots**: Three-body kinematics
5. **Advanced**: Negative weights and signed amplitudes

### Exercises

1. Derive the $s_{23}$ limits from energy-momentum conservation
2. Compute $|M_1 + M_2|^2$ showing interference term explicitly
3. Verify that $s_{12} + s_{23} + s_{13} = M^2 + \sum m_i^2$
4. Show that $\Gamma > 0$ implies probability conservation
5. Implement energy-dependent Breit-Wigner width
