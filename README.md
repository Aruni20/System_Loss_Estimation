# Physics-Guided Unsupervised Latent Alignment for Unified Solar PV System Loss Estimation

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## 📌 Project Overview
This repository contains the complete research deliverable for estimating system-level losses in grid-connected solar photovoltaic (PV) installations across India. The study proposes a **Physics-Guided Unsupervised Latent Alignment** approach that:
1.  Fuses outputs from four conventional PV modeling frameworks (SAPM, CEC, SAM, PVWatts) into a **Latent Consensus Estimate (L5)**.
2.  Augments this with a **Physics-Guided Model (L6)** capturing soiling kinetics, temperature derating, and humidity-driven degradation.
3.  Analyzes **72 major Indian cities** across diverse climatic zones using IMD and Open-Meteo meteorological data.

## 📁 Repository Structure

### 📄 Manuscript & Research
*   **`Manuscript.tex`**: Complete research manuscript in LaTeX (Elsevier `cas-sc` format).
*   **`references.bib`**: BibTeX file with all cited academic and technical references.
*   **`Project_File_Description.docx`**: Professional file-by-file description for academic submission.

### 🐍 Reproducibility (Python)
*   **`run_all_frameworks.py`**: Main orchestration script. Fetches weather data, executes all 6 loss frameworks, and performs statistical tests.
*   **`simulation_engine.py`**: Physics-informed Monte Carlo engine for L6 (soiling, temperature, humidity models).
*   **`generate_figures.py`**: Script to regenerate all manuscript figures from computed results.
*   **`System_Losses_Extimation.ipynb`**: Interactive walkthrough of the entire pipeline.

### 📊 Datasets (CSV)
*   **`results_master_imd.csv`**: Master consolidated results for all 72 cities across all frameworks.
*   **`India Cities LatLng.csv`**: Coordinates and climatic metadata for the study sites.
*   **Framework Specifics**: Individual CSVs for SAPM, CEC, SAM, PVWatts, Latent Consensus (L5), and Physics (L6).

### 🖼️ Figures
The directory contains 5 publication-ready figures (PNG):
1.  **Architecture Overview**: System data flow and latent alignment logic.
2.  **Physics Mechanisms**: Conceptual panels for soiling, humidity, and temperature.
3.  **Analytics**: Shapley feature attribution and L5 vs L6 correlation.
4.  **Spatial Analysis**: Loss distribution and uncertainty maps across the Indian subcontinent.
5.  **Time Series**: Soiling kinetic recovery (sawtooth) profile for Delhi.

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy pvlib scipy matplotlib requests
```

### Quick Start
1.  **Compute Results**: 
    ```bash
    python run_all_frameworks.py
    ```
2.  **Generate Figures**: 
    ```bash
    python generate_figures.py
    ```
3.  **Explore Data**: 
    Open `System_Losses_Extimation.ipynb` in Jupyter.

## 🌓 Results Summary
The study identifies a systematic divergence ($p = 1.66 \times 10^{-13}$) between conventional consensus and physics-informed estimates. Key findings show that omitting temporal memory effects (soiling/humidity) leads to overestimation of energy yield, particularly in semi-arid and coastal regimes.

## ✍️ Authors
*   **Aruni Saxena** (Dhirubhai Ambani University)
*   **Arpit Rana** (Supervision)
*   **Sreeja Rajendran** (Validation)

---
*Developed for advanced solar energy research and system performance optimization.*
