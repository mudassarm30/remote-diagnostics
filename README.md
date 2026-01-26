# 🛠️ Turbine Diagnostics – Rule-Based & ML-Assisted Anomaly Detection

## Overview

This project demonstrates a **practical diagnostic analytics workflow** for rotating machinery using **publicly available turbine-like sensor data**.

The goal is **not** to build a black-box predictive model, but to show how **time-series sensor data** can be analyzed using a combination of:

- Explainable **rule-based diagnostics**
- **Machine-learning–assisted anomaly detection**
- Sound **engineering judgment**

The workflow and design choices are aligned with **industrial remote diagnostics practices**, similar to those used for **gas and steam turbines**.

---

## Motivation

In industrial environments (e.g. gas turbines, steam turbines, compressors):

- Assets operate for long periods under noisy conditions
- Initial wear and unit-to-unit variation are normal
- Failures develop **gradually**, not instantaneously
- Diagnostics must be **explainable and trusted by domain experts**

This project focuses on **early degradation detection** rather than failure prediction accuracy.

---

## Dataset

### NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001)

- Source: NASA Prognostics Center of Excellence
- Dataset: **FD001**
- Engines (train): 100
- Operating condition: Single (sea level)
- Fault mode: **High-Pressure Compressor (HPC) degradation**

Although the dataset originates from aviation, the **diagnostic principles are directly transferable** to industrial gas turbines:

- Compressor efficiency loss
- Gradual sensor drift
- Fleet-based time-series analysis

Each engine:

- Starts in a healthy condition
- Develops a fault gradually
- Exhibits noisy sensor behavior
- Has unknown initial wear (treated as normal)

---

## Project Structure

```text
turbine-diagnostics-demo/
│
├── data/
│   ├── train_FD001.txt              # Raw NASA data
│   └── train_FD001_clean.csv        # Cleaned, validated dataset
│
├── notebooks/
│   ├── 01_data_loading.ipynb        # Data loading & validation
│   ├── 02_sensor_analysis.ipynb     # Sensor understanding & selection
│   ├── 03_rule_based_diagnostics.ipynb
│   └── 04_ml_anomaly_detection.ipynb
│
├── src/
│   ├── data_utils.py                # Loading & preprocessing helpers
│   ├── rules.py                     # Rule-based diagnostic logic
│   ├── ml_models.py                 # ML-based anomaly detection
│
└── README.md
```

## Diagnostic Indicators – Conceptual Background

Diagnostic indicators are simple, engineering-interpretable summaries of how a sensor behaves as an asset progresses from healthy operation toward degradation. In this project, indicators are used to support explainable reasoning about what is changing, how strongly it is changing, and whether the change looks like drift or instability.

The three core concepts are:

- Mean Shift: has the typical operating level moved?
- Variance Increase: has the signal become less stable or more erratic?
- Trend Slope: is there a slow, consistent drift over life?

### 1. Mean Shift (Offset Drift)

A mean shift is a sustained change in the typical operating level of a measurement. Physically, this often indicates that the underlying thermodynamic or mechanical state has moved to a new equilibrium for the same operating context.

Typical mechanisms this can indicate include:

- Efficiency loss or performance deterioration (e.g., compressor fouling reducing pressure rise, increased losses changing temperatures)
- Sensor calibration drift (instrument bias gradually changing)
- Gradual changes in clearances, leakage, or actuator alignment that move the operating point

Why comparing early-life vs late-life behavior is meaningful:

- Early life provides a practical “as-installed” baseline for that specific engine, including normal unit-to-unit offsets
- Late life reflects accumulated degradation effects after the system has had time to drift away from its baseline
- The comparison focuses on sustained change rather than short-term noise

### 2. Variance Increase (Instability)

Variance increase means the signal becomes more scattered over time: measurements fluctuate more around their typical level. In physical systems, rising variability is often a sign that regulation is working harder, margins are shrinking, or a component is intermittently misbehaving.

Typical mechanisms this can indicate include:

- Vibration-related issues that introduce intermittent disturbances into measurements
- Combustion instability or irregular operating behavior that produces cycle-to-cycle scatter
- Control oscillations or hunting (the controller repeatedly overshoots and corrects)
- Intermittent sensor issues (aging electronics, loose connections) that increase noise

Why instability can appear before large mean shifts:

- Early-stage faults can be intermittent, increasing spread without a strong offset
- Control systems can partially mask mean shifts by adjusting actuators while variability still grows
- Degradation can reduce stability or robustness before it changes the average operating level

### 3. Trend Slope (Gradual Wear)

Trend slope represents a slow, monotonic drift across life. This is the diagnostic signature of wear processes that accumulate continuously rather than stepping abruptly.

Typical wear-related mechanisms this can indicate include:

- Erosion or surface wear gradually changing flow-path effectiveness
- Clearance increase over time shifting performance and sensor responses
- Aging effects that slowly alter component behavior
- Progressive fouling or deposition that accumulates with operation

Why life-normalized trends are used instead of raw time:

- Engines have different lifetimes, so raw cycle count does not align comparable health stages across the fleet
- Normalizing life creates a common “early / mid / late” axis that supports consistent fleet interpretation
- It avoids over-weighting engines that simply have longer recorded histories

### Why indicators are normalized by early-life variability

Indicator magnitude must be interpreted relative to the sensor’s healthy noise level because that determines whether a change is diagnostically meaningful.

- Sensors have different units and natural ranges; absolute changes are not directly comparable across sensors
- Early-life variability provides a practical estimate of “normal scatter” under healthy operation and expected operating variation
- A small absolute drift can still be significant if it is large compared to healthy variability
- Normalization helps avoid ranking sensors by numeric scale rather than diagnostic clarity

## Analysis Plan

### Step 1 – Data Loading & Validation

- Load raw FD001 data
- Assign explicit column names
- Validate:
  - Fleet size
  - Time ordering
  - Sensor consistency
  - Operating conditions
- Produce a clean, reproducible dataset

**Rationale**  
Diagnostics must start from validated data; incorrect assumptions at this stage propagate errors downstream.

---

### Step 2 – Sensor Understanding & Selection

- Explore sensor behavior across engines
- Identify sensors that:
  - Show gradual degradation
  - Exhibit variance changes
  - Are informative for diagnostics
  - Ignore sensors that are flat or noisy-only
- Select a **small subset (4–6 sensors)**

**Rationale**  
In real systems, diagnostics focus on a subset of trusted, informative sensors rather than all available signals.

---

### Step 3 – Rule-Based Diagnostics

- Define simple, explainable indicators:
  - Rolling mean drift
  - Rolling variance increase
  - Threshold-based alerts
- Establish baseline behavior from early life cycles

**Rationale**  
Rule-based analytics are transparent, interpretable, and form the backbone of industrial diagnostic systems.

---

### Step 4 – ML-Assisted Anomaly Detection

- Apply unsupervised methods (e.g. Isolation Forest)
- Detect multivariate deviations not captured by single rules
- Use ML as **support**, not replacement, for rules

**Rationale**  
Machine learning complements rules by identifying subtle patterns while preserving explainability.

---

### Step 5 – Combined Diagnostic Logic

- Combine rule-based and ML outputs:
  - Rule + ML → High confidence
  - Rule only → Monitor
  - ML only → Investigate

**Rationale**  
This approach reduces false positives and builds trust in analytics.

