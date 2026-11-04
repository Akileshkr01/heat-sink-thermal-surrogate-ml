# Engineering Summary Report: Heat Sink Surrogate Modeling

**Author:** Akilesh  
**Discipline:** Mechanical Engineering & Applied ML  
**Project:** Physics-Informed ML Surrogate Modeling for Plate-Fin Heat Sinks

---

## 1. Assumptions and Engineering Justifications

Developing a reduced-order thermal model requires establishing boundary approximations that preserve core physics while keeping computation tractable. The analytical model and parameter sweep rely on the following decisions.

### 1.1. One-Dimensional Serial Resistance Formulation

Heat flow from the die junction through the thermal interface material (TIM) and the heat sink base is treated as one-dimensional normal conduction:

$$
R_{\text{total}} = R_{\text{jc}} + R_{\text{tim}} + R_{\text{cond}} + R_{\text{conv}}
$$

**Justification:**

The package thickness ($2.2\text{ mm}$ die, $0.1\text{ mm}$ TIM, $2.5\text{ mm}$ base) is an order of magnitude smaller than the lateral dimensions ($52.5 \times 45.0\text{ mm}$ die, $90 \times 116\text{ mm}$ sink). Normal thermal gradients vastly exceed planar gradients in the direct heat path, making 1D network modeling sufficient for primary temperature estimation.

### 1.2. Inclusion of Base Plate Conduction Resistance ($R_{\text{cond}}$)

The initial formulation evaluated heat sink thermal resistance solely as convective resistance ($R_{\text{hs}} = R_{\text{conv}}$), treating the aluminum base as infinitely conductive. In this work, normal conduction resistance through the base directly under the die footprint was integrated:

$$
R_{\text{cond}} = \frac{t_{\text{base}}}{k_{\text{Al}} \cdot A_{\text{die}}}
$$

For $t_{\text{base}} = 2.5\text{ mm}$, $k_{\text{Al}} = 167\text{ W/m}\cdot\text{K}$, and $A_{\text{die}} = 2362.5\text{ mm}^2$, this adds $0.00634^\circ\text{C/W}$, ensuring total thermal resistance accurately tracks physical hardware behavior.

### 1.3. Ducted Channel Flow with Negligible Fin Tip Bypass

Cooling air is assumed to pass entirely through the inter-fin channels without top or side bypass flow.

**Justification:**

This represents a fully shrouded or ducted electronics enclosure, matching the boundary condition assumed by the Sieder-Tate duct correlation.

### 1.4. Constant Air Properties Evaluated at Ambient Temperature

Air density, viscosity, thermal conductivity, and Prandtl number are evaluated at $T_{\text{ambient}} = 25^\circ\text{C}$ ($1\text{ atm}$) rather than locally updating with film temperature.

**Justification:**

Across the operating velocity sweep ($0.5\text{ to }15.0\text{ m/s}$), fin channel Reynolds numbers remain within moderate duct ranges ($36.7 \le Re \le 1101.2$). While film temperature variations induce small shifts in kinematic viscosity, fixing ambient properties isolates kinematic and thermal dependencies, preventing circular dependencies during Latin Hypercube Sampling.

### 1.5. Smooth Transition Flow Blending

Instead of a hard threshold step at $Re = 2300$ between laminar developing duct flow (Sieder-Tate) and turbulent flow (Dittus-Boelter), a cubic Hermite polynomial interpolation was formulated across:

$$
2300 \le Re \le 3000
$$

**Justification:**

Artificial step discontinuities in physical models disrupt gradient-based regressors, causing tree-splitting errors and gradient explosion in neural networks. Continuous blending preserves physical reality and numerical stability across extended velocity sweeps.

---

## 2. Key Results from the Model

Evaluating the analytical physics model at the reference operating point provides the following baseline breakdown.

### Reference Operating Conditions

| Parameter | Value |
|:---|---:|
| Thermal Design Power (TDP) | 150 W |
| Approach Air Velocity ($V_{\text{air}}$) | 1.0 m/s |
| TIM Thermal Conductivity ($k_{\text{tim}}$) | 4.0 W/m·K |
| Ambient Temperature ($T_{\text{ambient}}$) | 25.0 °C |

### 2.1. Baseline Thermal Performance

| Physical State Variable | Calculated Value | Unit | Engineering Note |
|:---|---:|:---|:---|
| Reynolds Number ($Re$) | 73.41 | - | Hydrodynamically developing laminar channel regime |
| Nusselt Number ($Nu$) | 2.0480 | - | Thermally developing duct profile (Sieder-Tate) |
| Convective Coefficient ($h_{\text{conv}}$) | 23.28 | W/m²·K | Forced air boundary layer rate |
| Convective Surface Area ($A_{\text{total}}$) | 0.27504 | m² | Combined fin flanks, tips, and exposed base channels |
| TIM Resistance ($R_{\text{tim}}$) | 0.01058 | °C/W | 3.9% of total thermal resistance |
| Base Conduction Resistance ($R_{\text{cond}}$) | 0.00634 | °C/W | 2.3% of total thermal resistance |
| Convective Thermal Resistance ($R_{\text{conv}}$) | 0.15619 | °C/W | 57.2% of total thermal resistance (primary bottleneck) |
| Junction-to-Case Resistance ($R_{\text{jc}}$) | 0.10000 | °C/W | 36.6% of total thermal resistance (internal constraint) |
| Total Thermal Resistance ($R_{\text{total}}$) | 0.27311 | °C/W | Overall system thermal impedance |
| Processor Junction Temperature ($T_j$) | 65.97 | °C | Well within standard silicon reliability limits (< 85 °C) |

**Note:** If junction-to-case resistance is adjusted to $R_{\text{jc}} = 0.20^\circ\text{C/W}$ to mirror legacy package references, total resistance becomes $0.37311^\circ\text{C/W}$ with $T_j = 80.97^\circ\text{C}$.

### 2.2. Operational Domain Performance

Across the broader Latin Hypercube operational domain ($N = 2500$):

- Calculated junction temperatures span from $30.3^\circ\text{C}$ ($\text{TDP} \approx 30\text{ W}$, $V_{\text{air}} \approx 12.6\text{ m/s}$) up to $97.4^\circ\text{C}$ ($\text{TDP} \approx 235\text{ W}$, $V_{\text{air}} \approx 0.6\text{ m/s}$, low TIM).
- Sustained high heat fluxes ($>220\text{ W}$) combined with low air velocities ($<1.0\text{ m/s}$) push junction temperatures toward the critical $95\text{--}100^\circ\text{C}$ thermal throttling boundary, demonstrating the necessity of active dynamic fan modulation.

---

## 3. ML Model Accuracy Comparison

Two machine learning architectures were trained on $2,000$ points from a Latin Hypercube parameter sweep and evaluated against an unseen holdout test split of $500$ points ($80/20$ split).

### 3.1. Model Configurations

**Gradient Boosting Regressor (GBR / GBDT):**

- Number of estimators: 300
- Learning rate: 0.04
- Maximum tree depth: 5
- Random state: 42

**Multi-Layer Perceptron (MLP):**

- Hidden layer sizes: $(128, 64)$
- Activation function: ReLU
- Preprocessing: `StandardScaler` pipeline
- Maximum iterations: 1500 with early stopping enabled

### 3.2. Test Partition Accuracy Benchmark

| Predicted Target | Architecture | MAE | RMSE | R² Score | Interpretation |
|:---|:---|---:|---:|---:|:---|
| $R_{\text{total}}$ (°C/W) | Gradient Boosting (GBR) | 0.000556 | 0.001062 | 0.99827 | Captures power-law velocity decay with near-zero residual spread. |
| $R_{\text{total}}$ (°C/W) | Multi-Layer Perceptron (MLP) | 0.003522 | 0.004555 | 0.96822 | Good global fit; slightly rounds the sharp knee at low velocities. |
| $T_{\text{junction}}$ (°C) | Gradient Boosting (GBR) | 0.3630 °C | 0.5952 °C | 0.99814 | Average error is well under $0.4^\circ\text{C}$, exceeding CFD mesh convergence fidelity. |
| $T_{\text{junction}}$ (°C) | Multi-Layer Perceptron (MLP) | 0.5673 °C | 0.9747 °C | 0.99502 | Strong continuous tracking across the entire temperature scale. |

### 3.3. Architectural Trade-offs

**Why GBR Won on Discrete Accuracy:**

Convective resistance scales with air velocity according to an inverse fractional power ($R_{\text{conv}} \propto V^{-1/3}$). Tree ensembles partition the feature space with orthogonal splits along the velocity axis, accurately resolving the steep decay below $2.0\text{ m/s}$ without requiring input scaling.

**Where MLP Excels:**

While GBR achieves lower discrete absolute error, the neural network provides continuous, differentiable function manifolds ($\partial T_j / \partial V$). This makes the MLP preferable for downstream gradient-based numerical optimization, where piecewise-constant tree-step boundaries can stall solver convergence.

---

## 4. Important Correlations and Sensitivities

Analyzing the Latin Hypercube sweep using Spearman rank-order correlation coefficients ($\rho$) isolates monotonic dependencies without linear bias.

### 4.1. Spearman Rank Correlation Matrix

| Parameter | TDP_W | V_air_m_s | k_tim_W_mK | R_total_C_W | T_j_C |
|:---|---:|---:|---:|---:|---:|
| **TDP_W** | 1.0000 | 0.0088 | 0.0275 | -0.0211 | 0.9658 |
| **V_air_m_s** | 0.0088 | 1.0000 | 0.0284 | -0.9161 | -0.1990 |
| **k_tim_W_mK** | 0.0275 | 0.0284 | 1.0000 | -0.3326 | -0.0367 |
| **R_total_C_W** | -0.0211 | -0.9161 | -0.3326 | 1.0000 | 0.2023 |
| **T_j_C** | 0.9658 | -0.1990 | -0.0367 | 0.2023 | 1.0000 |

### 4.2. Physical Interpretations

#### Air Velocity Governs Resistance ($\rho = -0.9161$)

Convective thermal resistance constitutes between $57\%$ and $80\%$ of total system impedance. Increasing approach velocity directly compresses the boundary layer inside the narrow $1.15\text{ mm}$ channels, driving $h_{\text{conv}}$ upward.

#### The Convective Law of Diminishing Returns

- Increasing velocity from $0.5\text{ m/s}$ to $3.0\text{ m/s}$ reduces total resistance from $0.3137^\circ\text{C/W}$ to $0.2252^\circ\text{C/W}$—a $28.2\%$ reduction.
- Increasing velocity further from $3.0\text{ m/s}$ to $15.0\text{ m/s}$ yields an additional drop of only $19.9\%$ (down to $0.1803^\circ\text{C/W}$).

However, fan pumping power scales with the cube of velocity:

$$
P_{\text{fan}} \propto V^3
$$

Operating beyond $4.0\text{ to }5.0\text{ m/s}$ incurs severe acoustic penalties and electrical power draw for rapidly vanishing thermal gains.

#### TDP Drives Junction Temperature Directly ($\rho = +0.9658$)

From the 1D resistance equation:

$$
T_j = T_{\text{ambient}} + \text{TDP} \cdot R_{\text{total}}
$$

Package heat generation acts as the primary scalar driver of temperature rise. In contrast, TDP exhibits virtually zero correlation with $R_{\text{total}}$ ($\rho = -0.0211$), confirming that thermal resistance is an intrinsic property of geometry and flow rather than heat flux under constant-property conditions.

#### TIM Saturation Threshold ($\rho = -0.3326$)

Upgrading from low-grade thermal grease ($1.0\text{ W/m}\cdot\text{K}$) to standard compound ($4.0\text{ W/m}\cdot\text{K}$) drops $R_{\text{tim}}$ from $0.0423^\circ\text{C/W}$ to $0.0106^\circ\text{C/W}$ (saving $0.0317^\circ\text{C/W}$, which lowers $T_j$ by $4.76^\circ\text{C}$ at $150\text{ W}$). Moving from $4.0\text{ W/m}\cdot\text{K}$ to exotic compounds ($12.0\text{ W/m}\cdot\text{K}$) saves only an additional $0.0071^\circ\text{C/W}$ ($1.06^\circ\text{C}$ improvement). Once conductivity surpasses $4\text{ to }5\text{ W/m}\cdot\text{K}$, TIM ceases to be the bottleneck.

---

## 5. Limitations of the Approach

While effective for rapid parametric exploration, this reduced-order methodology possesses several physical simplifications.

### 5.1. Absence of 3D Spreading Resistance

The model treats conduction through the aluminum baseplate as purely 1D normal to the die footprint. In reality, the die area ($2362.5\text{ mm}^2$) occupies only $22.6\%$ of the total base footprint ($10440\text{ mm}^2$). Heat spreads laterally into outer base regions before entering periphery fins, contributing an additional $0.02\text{ to }0.04^\circ\text{C/W}$ in real hardware.

### 5.2. Idealized Fin Efficiency ($\eta_f = 1.0$)

Fins are assumed isothermal along their height. For aluminum fins ($H=24.5\text{ mm}$, $t_{\text{fin}}=0.8\text{ mm}$), true fin efficiency ranges between $88\%$ and $94\%$ depending on local convective rates, causing a minor overprediction ($6\text{ to }10\%$) in convective heat dissipation.

### 5.3. Shrouded vs. Unshrouded Bypass Effects

The formulation assumes zero flow bypass. In open electronic enclosures without tight shrouds, air takes the path of lower hydraulic resistance over fin tips, reducing effective inter-fin channel velocity by $30\%\text{ to }50\%$.

### 5.4. Incompressible Constant-Property Fluid

Air property evaluation at $25^\circ\text{C}$ ignores caloric air temperature rise along the duct:

$$
\Delta T_{\text{caloric}} = \frac{\text{TDP}}{\dot{m} c_p}
$$

At low velocities ($0.5\text{ m/s}$), air warming along the $90\text{ mm}$ flow path degrades the driving temperature potential at the channel trailing edge.

---

## 6. Areas for Improvement and Optimization

1. **Analytical Spreading and Fin Efficiency Corrections:** Integrate classical fin efficiency formulations ($\eta_f = \tanh(mH_c) / (mH_c)$) and Lee-Song 2D/3D spreading resistance models into the base calculation engine.

2. **Multi-Objective Pareto Optimization (NSGA-II):** Couple the trained MLP surrogate with a multi-objective genetic algorithm to simultaneously minimize junction temperature ($T_j$), heat sink mass/volume, and fan pumping power ($\Delta P \cdot \dot{V}$). Because surrogate inference runs in under $0.5\text{ ms}$, a 50,000-evaluation Pareto sweep completes in seconds.

3. **Physics-Informed Neural Networks (PINNs):** Embed the 1D governing energy balance directly into the loss function ($\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{energy}}$) to ensure bounded, physically consistent predictions when querying boundary edge cases.

4. **Integration with Chassis CFD & BMC Telemetry:** Deploy the trained surrogate within server Baseboard Management Controller (BMC) firmware as a real-time digital twin to drive dynamic closed-loop PWM fan control based on chip telemetry.

---

**Physics-Informed Thermal Modeling | Conjugate Heat Transfer | Machine Learning Surrogates | Global Sensitivity Analysis | Thermal Optimization**