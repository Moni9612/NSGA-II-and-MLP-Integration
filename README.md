# NSGA-II and MLP Integration

An **MLP-assisted NSGA-II framework for geometric optimization of the HCA adaptive finger** for cluttered waste sorting.

<p align="center">
  <img src="architecture.jpeg" alt="MLP-assisted NSGA-II optimization framework" width="100%">
</p>

---

## Overview

The framework integrates a trained **Multi-Layer Perceptron (MLP)** model with the **Non-dominated Sorting Genetic Algorithm II (NSGA-II)** for geometric optimization of an adaptive robotic finger used in the **Hybrid Compliance Activation (HCA) mechanism for cluttered waste sorting**.

The trained MLP enables efficient exploration of the finger design space while reducing the computational cost associated with repeated finite-element simulations.

The implementation comprises two main components:

1. **MLP Model** – developed and trained using finite-element simulation data.
2. **NSGA-II Multi-Objective Optimization** – integrates the trained MLP with NSGA-II for efficient exploration of the finger design space.

---

# MLP Model and Usage

The **Multi-Layer Perceptron (MLP)** model is developed and trained in **Google Colab** using data obtained from finite-element simulations in COMSOL. The model learns the relationship between the finger design parameters, contact conditions, and corresponding mechanical responses.

The MLP workflow consists of three Python scripts executed sequentially within the same Google Colab session. Model development follows an iterative procedure, allowing the MLP configuration to be adjusted, retrained, and re-evaluated until satisfactory predictive performance is achieved.

<p align="center">
  <img src="MLP.png" alt="MLP creation and training" width="100%">
</p>

## 1. Data Preprocessing and MLP Creation

Copy the contents of [`Data_preprocessing_and_MLP_creation.py`](Data_preprocessing_and_MLP_creation.py) into the **first cell** of the Google Colab notebook and execute it.

The initial stage performs data preparation and model setup, including:

- Uploading and loading the simulation dataset
- Defining the MLP input features and output targets
- Data preprocessing and cleaning
- Splitting the dataset into training, validation, and test sets
- Scaling the input and output data
- Defining and creating the MLP architecture

```python
# Cell 1
# Use the contents of:
# Data_preprocessing_and_MLP_creation.py
```

The MLP configuration parameters are defined at this stage and can be modified during model tuning, including the network architecture, number and size of hidden layers, learning rate, and other model-related hyperparameters.

---

## 2. MLP Training

Copy the contents of [`MLP_training.py`](MLP_training.py) into the **second cell** and execute it after completing the data-preprocessing and model-creation stage.

```python
# Cell 2
# Use the contents of:
# MLP_training.py
```

The MLP is trained using the training and validation datasets prepared in the previous cell.

Model development is performed iteratively. After each training cycle, evaluate the model using the third cell. If further improvement is required, return to the first cell and adjust the relevant model or training parameters, such as:

- Number of hidden layers
- Number of neurons per layer
- Learning rate
- Number of training epochs
- Other model and training hyperparameters

After modifying the parameters, recreate the model and repeat the training process.

> **Note:** Re-executing the model-creation step initializes a new MLP. The model should therefore be retrained after modifying its architecture or relevant configuration parameters.

---

## 3. Model Evaluation and Saving

Copy the contents of [`MLP_evaluation_model_saving.py`](MLP_evaluation_model_saving.py) into the **third cell** and execute it after each training cycle.

```python
# Cell 3
# Use the contents of:
# MLP_evaluation_model_saving.py
```

The trained MLP is evaluated to assess and visualize its predictive performance. The resulting evaluation metrics can be used to determine whether further hyperparameter tuning and retraining are required.

Repeat the training and evaluation process, as illustrated in the workflow above, until satisfactory predictive performance is achieved.

Once the final model configuration has been selected, the same script can be used to save the **trained MLP model and associated data scalers**.

The saved model and scalers can subsequently be reloaded without retraining and used during the **MLP–NSGA-II integration and optimization stage**.

---

# NSGA-II and MLP Integration

The trained MLP is integrated with a constrained **NSGA-II-based multi-objective optimization framework** to optimize the adaptive Fin-Ray finger geometry. This stage is implemented in a separate **Google Colab** environment.

The optimization considers three objectives:

- Minimize the coefficient of variation of in-plane conformance, $CV_D$
- Minimize the coefficient of variation of contact force, $CV_F$
- Minimize the Z-dispersion index, $ZDI$

subject to the force constraint:

$$
\mu_F \geq 10~\text{N}
$$

where $\mu_F$ represents the required minimum mean contact force across the considered operating conditions.

The trained MLP and associated scalers generated during the previous stage are loaded and used to evaluate newly generated candidate geometries, avoiding repeated finite-element simulations during the optimization process.

---

## Environment Setup

Create a new **Google Colab** notebook for the NSGA-II optimization stage.

In the initial setup cell, install `pymoo` using:

```python
!pip install pymoo
```

This command only needs to be executed once at the beginning of the Colab session to establish the required multi-objective optimization environment.

> **Note:** The setup cell is not included as a Python file. After installing `pymoo`, the remaining five cells use the Python scripts provided with the framework.

The complete workflow is illustrated below.

<p align="center">
  <img src="NSGA-II.png" alt="NSGA-II-based multi-objective optimization framework" width="100%">
</p>

---

## 1. NSGA-II and MLP Initialization — Iteration 1

Copy the contents of [`NSGAII_MLP_initialization_and_iteration1.py`](NSGAII_MLP_initialization_and_iteration1.py) into **Cell 1** and execute it.

```python
# Cell 1
# Use the contents of:
# NSGAII_MLP_initialization_and_iteration1.py
```

Cell 1 initializes the complete constrained NSGA-II + MLP optimization framework and automatically performs **Iteration 1**.

During initialization, the trained MLP model and its associated input and output scalers are loaded into the Colab environment. The initial finite-element dataset is also loaded and used to construct the starting design population.

The initialization stage performs the following operations:

- Loads the trained MLP model and associated scalers
- Loads the finite-element simulation dataset
- Defines the five finger geometry variables
- Defines the operating/contact conditions
- Defines the three optimization objectives
- Applies the mean-force feasibility constraint
- Constructs the initial constrained Pareto front
- Defines the MLP-based geometry evaluation function
- Defines candidate-generation and duplicate-checking functions
- Generates new candidate geometries
- Evaluates the candidates using the trained MLP
- Adds the newly evaluated designs to the cumulative population
- Updates the constrained Pareto front

Each candidate geometry is evaluated under all considered operating conditions using the trained MLP.

The resulting displacement and force predictions are aggregated to calculate the three objectives:

$$
CV_D,\qquad CV_F,\qquad ZDI
$$

while the required minimum mean contact force is used to enforce:

$$
\mu_F \geq 10~\text{N}.
$$

Only designs satisfying this force requirement are considered feasible during constrained non-dominated sorting.

> **Note:** Cell 1 automatically performs **Iteration 1**.

---

## 2. Initialize Convergence History

After completing Iteration 1, copy the contents of [`Convergence_history_initialization.py`](Convergence_history_initialization.py) into **Cell 2** and execute it.

```python
# Cell 2
# Use the contents of:
# Convergence_history_initialization.py
```

Cell 2 initializes the convergence-history table before convergence tracking begins.

The following information is recorded during the optimization:

- Iteration number
- Cumulative number of evaluated designs
- Number of constrained Pareto designs
- Minimum $CV_D$
- Minimum $CV_F$
- Minimum $ZDI$
- Pareto-front hypervolume

This cell should be executed **once after Iteration 1 and before the first convergence evaluation**.

---

## 3. NSGA-II Convergence Tracking

Copy the contents of [`NSGAII_convergence_tracking.py`](NSGAII_convergence_tracking.py) into **Cell 3** and execute it after initializing the convergence history.

```python
# Cell 3
# Use the contents of:
# NSGAII_convergence_tracking.py
```

The convergence-tracking stage evaluates the current constrained Pareto front and records the progression of the optimization.

The primary convergence indicator is the **hypervolume of the constrained Pareto front**.

During the first execution, a fixed hypervolume reference point is established based on the evaluated objective space. The same reference point is retained throughout all subsequent iterations to ensure that the hypervolume values remain directly comparable.

For each iteration, the following information is recorded:

- Current iteration number
- Cumulative number of evaluated designs
- Number of Pareto-optimal designs
- Minimum $CV_D$
- Minimum $CV_F$
- Minimum $ZDI$
- Hypervolume

The convergence history is saved as:

```text
FinRay_Pareto_Convergence_History.csv
```

and the corresponding hypervolume convergence plot is saved as:

```text
Pareto_Hypervolume_Convergence.png
```

The plot presents the Pareto-front hypervolume against the **cumulative number of evaluated designs**, allowing the progression of the optimization to be monitored.

> **Important:** The same hypervolume reference point must be maintained throughout the complete optimization run.

After reviewing the convergence behaviour, proceed to the next iteration if the Pareto front has not yet converged.

---

## 4. Iterative NSGA-II + MLP Optimization

Copy the contents of [`NSGAII_MLP_iteration.py`](NSGAII_MLP_iteration.py) into **Cell 4**.

```python
# Cell 4
# Use the contents of:
# NSGAII_MLP_iteration.py
```

For the first execution, set:

```python
iteration_number = 2
```

Each execution of Cell 4 performs one additional constrained NSGA-II + MLP optimization iteration.

During each iteration:

- 20 new candidate geometries are generated
- Each candidate is evaluated using the trained MLP
- Candidate responses are predicted across all operating conditions
- $CV_D$, $CV_F$, and $ZDI$ are calculated
- Mean predicted contact force is calculated
- The $\mu_F \geq 10$ N feasibility constraint is evaluated
- Feasible and infeasible candidates are recorded
- Newly evaluated designs are appended to the cumulative population
- The constrained Pareto front is recalculated

After executing Cell 4, return to **Cell 3** and run the convergence-tracking code again.

If convergence has not yet been reached, increase only the iteration number by one before executing Cell 4 again.

Continue alternating between **Cell 4** and **Cell 3** until the Pareto-front hypervolume indicates convergence.

For the optimization performed in this work, convergence was observed at **Iteration 9**.

> **Note:** Iteration 9 represents the convergence point observed for the optimization conducted in this work. Other datasets, design spaces, or optimization configurations may require a different number of iterations.

---

## 5. Final Pareto Front and Optimal Design Selection

Once convergence has been reached, copy the contents of [`Pareto_front_and_optimal_design_selection.py`](Pareto_front_and_optimal_design_selection.py) into **Cell 5** and execute it.

```python
# Cell 5
# Use the contents of:
# Pareto_front_and_optimal_design_selection.py
```

The final stage analyzes the converged constrained Pareto front and identifies the highest-ranked candidate designs.

First, the final Pareto population is verified to ensure that all Pareto-optimal designs satisfy:

$$
\mu_F \geq 10~\text{N}.
$$

The three minimized objectives,

$$
CV_D,\qquad CV_F,\qquad ZDI,
$$

are then min-max normalized across the final constrained Pareto set.

Because all three objectives are minimized, the normalized utopia point is defined as:

$$
\mathbf{f}^{*} = [0,\;0,\;0].
$$

For each Pareto-optimal design, the Euclidean distance from its normalized objective vector to the utopia point is calculated.

The designs are ranked in ascending order of their distance to the utopia point, where a smaller distance represents a more balanced trade-off among the three optimization objectives.

The **five Pareto designs with the smallest utopia distances** are selected as the five highest-ranked candidate designs.

For each of these five designs, the following information is reported:

- Finger geometry
- $CV_D$
- $CV_F$
- $ZDI$
- Mean predicted contact force
- Utopia distance
- Predicted contact force across all 12 operating conditions
- Predicted $\delta_x$ across all 12 operating conditions
- Predicted $\delta_y$ across all 12 operating conditions
- Predicted $\delta_z$ across all 12 operating conditions

The final constrained three-objective Pareto front is also visualized in 3D.

The five highest-ranked Pareto designs are identified as:

```text
A → Rank 1
B → Rank 2
C → Rank 3
D → Rank 4
E → Rank 5
```

These five highest-ranked Pareto-optimal finger geometries can subsequently be carried forward for simulation and experimental validation.
