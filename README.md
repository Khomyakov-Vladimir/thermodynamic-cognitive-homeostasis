**Reproducibility Package**  
# Thermodynamic Cognitive Homeostasis (TCH): A Subjective Physics Approach to Self‑Regulating AI  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17592100.svg)](https://doi.org/10.5281/zenodo.17592100)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
---

## Overview  
This package contains the reproducible simulation code and theoretical framework supporting the publication:  

*Vladimir Khomyakov (2025). Thermodynamic Cognitive Homeostasis (TCH): A Subjective Physics Approach to Self‑Regulating AI.*  

- **Version‑specific DOI:** [10.5281/zenodo.17607736](https://doi.org/10.5281/zenodo.17607736)  
- **Concept DOI (latest version):** [10.5281/zenodo.17592100](https://doi.org/10.5281/zenodo.17592100)  

---

## Description  

This repository provides the implementation and validation of the **Thermodynamic Cognitive Homeostasis (TCH)** model — a formal framework describing **self‑regulating cognitive systems** in thermodynamic terms.  
Each cognitive agent maintains informational equilibrium by controlling entropy, energy, and learning cost, achieving adaptive stability and phase transitions under incomplete or noisy information.  

The model extends **Subjective Physics** by integrating **homeostatic feedback**, **entropic regulation**, and **multi‑agent synchronization**, supported by a fully reproducible Python simulator implementing equations (3)–(5) from the paper.  
The code validates theoretical predictions on **cognitive equilibrium**, **collective phase synchronization**, and **entropy–energy exchange stability**.  

---

## Repository  
- **Source repository:** [https://github.com/Khomyakov-Vladimir/thermodynamic-cognitive-homeostasis](https://github.com/Khomyakov-Vladimir/thermodynamic-cognitive-homeostasis)  

---

## Package Structure  

```
thermodynamic-cognitive-homeostasis/  
│
├── README.md  
├── LICENSE
├── .zenodo.json
├── CITATION.cff
├── environment-tch.yml
├── requirements-tch.txt
│
├── tch_simulation_validation.py       # Main simulation script implementing TCH equations (3)–(5)  
│                                      # Reproduces figures and stability analysis from the paper  
├── scripts/
│   └── tch_simulation_validation.py   # Main simulation script implementing TCH equations (3)–(5)  
│                                      # Reproduces figures and stability analysis from the paper  
└── figures/  
    ├── phi_time_series_baseline.pdf     # Baseline collective order Φ(t)  
    ├── phi_alpha_comparison.pdf         # Φ(t) for various α_homeo values  
    ├── phi_phase_heatmap.pdf            # Phase diagram (Φ vs coupling_S, α_homeo)  
    ├── phi_heatmap_matrix.csv           # Numeric data for the phase heatmap  
    └── tch_summary_results.csv          # Summary table of selected simulation runs  
```

---

## Dependencies  

Install all dependencies using:  

```bash  
pip install -r requirements-tch.txt  
```

Pinned versions used for verification and figure reproduction are provided below.  

### requirements-tch.txt  
```
matplotlib==3.10.7  
numpy==2.3.4  
pandas==2.3.3  
```

> **Note on dependency verification**:  
> This archive includes minimal, empirically verified dependencies generated via an isolated Conda environment to ensure computational reproducibility. No unused or extraneous packages are included.  

---

## Python Environment (conda)  

File: `environment-tch.yml`  

```yaml  
name: tch-env
channels:
  - conda-forge
dependencies:
  - python>=3.8
  - matplotlib
  - numpy
  - pandas
```

Activate environment:  

```bash  
conda env create -f environment-tch.yml  
conda activate tch-env  
```

---

## Usage  

All scripts should be run from the repository root.  
Figures are generated in the `figures/` directory.  

### 1. Baseline simulation (collective homeostasis)  

```bash
python tch_simulation_validation.py  
```

or explicitly:  

```bash
python scripts/tch_simulation_validation.py  
```

or explicitly:  

```bash
python tch_simulation_validation.py --mode baseline  
```

Output:  
- `phi_time_series_baseline.pdf` — time evolution of collective order parameter Φ(t).  

---

### 2. Homeostatic strength comparison  

```bash
python tch_simulation_validation.py --mode alpha_comparison  
```

Produces:  
- `phi_alpha_comparison.pdf` — comparison of Φ(t) for different α_homeo values.  

---

### 3. Phase‑space sweep  

```bash  
python tch_simulation_validation.py --mode phase_heatmap  
```

Produces:  
- `phi_phase_heatmap.pdf` — final Φ values across (α_homeo, coupling_S).  
- `phi_heatmap_matrix.csv` — corresponding numeric matrix.  

---

## Theoretical Foundation  

The TCH model defines each cognitive agent by four thermodynamic variables:  

- Cognitive entropy **Sᶜᵒᵍᵢ**  
- Cognitive energy **Eᵢ**  
- Equilibrium entropy **Sᵉᵠᵢ**  
- Cognitive temperature **Tᶜᵒᵍᵢ**  

The homeostatic gradient acts as:  

∇𝑤ᵢ (Sᶜᵒᵍᵢ − Sᵉᵠᵢ)²  
ensuring relaxation toward informational equilibrium.  

In the multi-agent case, agents are coupled through entropy and energy fluxes:  

Ẋ Sᶜᵒᵍᵢ = f(Eᵢ, φᵢ) − g(ℒᵢ) + Σⱼ₍ⱼ ≠ ᵢ₎ κˢⁱʲ (Sᶜᵒᵍⱼ − Sᶜᵒᵍᵢ)  
Ẋ Eᵢ = −h(℘ᵢ) + Σⱼ₍ⱼ ≠ ᵢ₎ κᴱⁱʲ (Eⱼ − Eᵢ)  

Global stability is achieved when the total free energy **Fₜₒₜ** decreases monotonically,  
reflecting cognitive convergence and synchronized homeostasis.  

---

## Reproducibility  

All stochastic components use **fixed random seeds** to ensure deterministic reproducibility.  
Running the same simulation multiple times yields identical results and figures.  

Example verification:  

```bash  
python tch_simulation_validation.py --test_reproducibility  
```

Expected output:  
```
✓ Reproducibility verified: identical seeds produce identical results  
```

---

## Citation  

Vladimir Khomyakov (2025). *Thermodynamic Cognitive Homeostasis (TCH): A Subjective Physics Approach to Self‑Regulating AI.* Zenodo. https://doi.org/10.5281/zenodo.17592100  
