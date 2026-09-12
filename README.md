# NSGA-II-and-MLP-Integration

MLP-assisted NSGA-II framework for geometric optimization of the HCA adaptive finger for cluttered waste sorting.

## Overview

This repository contains the implementation of a **Multi-Layer Perceptron (MLP)-assisted NSGA-II optimization framework** developed for the geometric optimization of an adaptive robotic finger used in the **Hybrid Compliance Activation (HCA) mechanism for cluttered waste sorting**.

The framework integrates a trained MLP surrogate model with the **Non-dominated Sorting Genetic Algorithm II (NSGA-II)** to efficiently explore the finger design space while reducing the computational cost associated with repeated finite element simulations.

The implementation consists of two main components:

1. **MLP Surrogate Model** – developed and trained using finite element simulation data.
2. **NSGA-II Multi-Objective Optimization** – integrates the trained MLP surrogate model with NSGA-II for efficient exploration of the finger design space.

---

# MLP Surrogate Model

The **Multi-Layer Perceptron (MLP)** surrogate model was developed and trained using **Google Colab**. The model learns the relationship between the finger design parameters, contact conditions, and corresponding mechanical responses obtained from finite element simulations.

The complete MLP development workflow consists of the following stages.

## Data Loading

The finite element simulation dataset is loaded into the Google Colab environment for initial inspection and subsequent processing.

```python
# Data loading code
# Add code here
