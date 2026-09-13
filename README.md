# NSGA-II-and-MLP-Integration

MLP-assisted NSGA-II framework for geometric optimization of the HCA adaptive finger for cluttered waste sorting.

<p align="center">
  <img src="architecture.jpeg" alt="MLP-assisted NSGA-II optimization framework" width="100%">
</p>

## Overview

This repository contains the implementation of a **Multi-Layer Perceptron (MLP)-assisted NSGA-II optimization framework** developed for the geometric optimization of an adaptive robotic finger used in the **Hybrid Compliance Activation (HCA) mechanism for cluttered waste sorting**.

The framework integrates a trained MLP model with the **Non-dominated Sorting Genetic Algorithm II (NSGA-II)** to efficiently explore the finger design space while reducing the computational cost associated with repeated finite element simulations.

The implementation consists of two main components:

1. **MLP Model** – developed and trained using finite element simulation data.
2. **NSGA-II Multi-Objective Optimization** – integrates the trained MLP model with NSGA-II for efficient exploration of the finger design space.

---

# MLP Model

The **Multi-Layer Perceptron (MLP)** model was developed and trained using **Google Colab**. The model learns the relationship between the finger design parameters, contact conditions, and corresponding mechanical responses obtained from finite element simulations.

The complete MLP development workflow consists of the following stages.
# Usage

The MLP development workflow is designed to be executed in **Google Colab** using the three Python scripts provided in this repository. The scripts should be executed sequentially in separate Colab cells, as described below.

## 1. Create a Google Colab Environment

Create a new Google Colab notebook and configure the runtime required for MLP training. The three Python scripts in this repository are intended to be used sequentially within the same Colab session.

The recommended execution order is:

```text
Cell 1: Data_preprocessing_and_MLP_creation.py
                         ↓
Cell 2: MLP_training.py
                         ↓
Cell 3: MLP_evaluation_model_saving.py
                         ↓
                  Model Evaluation
                         ↓
              ┌── Satisfactory? ──┐
              │                   │
             No                  Yes
              │                   │
              ↓                   ↓
      Adjust MLP Parameters    Save Model
              │                & Scalers
              └────→ Retrain
```

## 2. Data Preprocessing and MLP Creation

Copy the contents of [`Data_preprocessing_and_MLP_creation.py`](Data_preprocessing_and_MLP_creation.py) into the **first cell** of the Google Colab notebook and execute it.

This script performs the initial data preparation and model setup, including:

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

The MLP configuration parameters are defined at this stage. These parameters can be modified during model tuning, including the network architecture, number and size of hidden layers, learning rate, and other model-related hyperparameters.

## 3. MLP Training

Copy the contents of [`MLP_training.py`](MLP_training.py) into the **second cell** and execute it after completing the data preparation and model creation stage.

```python
# Cell 2
# Use the contents of:
# MLP_training.py
```

This script trains the MLP using the training and validation datasets prepared in the previous cell.

Model development should be performed iteratively. After each training cycle, evaluate the model using the third cell. If further improvement is required, return to the first cell and adjust the relevant model or training parameters, such as:

- Number of hidden layers
- Number of neurons per layer
- Learning rate
- Number of training epochs
- Other model and training hyperparameters

After modifying the parameters, recreate the model and repeat the training process.

> **Note:** Re-executing the model creation step initializes a new MLP. Therefore, the model should be retrained after modifying its architecture or relevant configuration parameters.

## 4. Model Evaluation and Saving

Copy the contents of [`MLP_evaluation_model_saving.py`](MLP_evaluation_model_saving.py) into the **third cell** and execute it after each training cycle.

```python
# Cell 3
# Use the contents of:
# MLP_evaluation_model_saving.py
```

This script is used to evaluate the predictive performance of the trained MLP and visualize its current performance. The evaluation results can then be used to determine whether further hyperparameter tuning and retraining are required.

The recommended iterative workflow is:

```text
Configure MLP
     ↓
Train MLP
     ↓
Evaluate Model
     ↓
Inspect Performance
     ↓
Adjust Hyperparameters (if required)
     ↓
Recreate and Retrain MLP
     ↓
Re-evaluate
```

Repeat this process until satisfactory predictive performance is achieved.

Once the final model configuration has been selected, the same script can be used to save the **trained MLP model and associated data scalers**.

These saved components allow the trained surrogate model to be reloaded without retraining and subsequently used during the **MLP–NSGA-II integration and optimization stage**.

## 5. Recommended Colab Workflow

The complete Google Colab notebook can therefore be organized into three primary cells:

| Cell | Script | Purpose |
| --- | --- | --- |
| **Cell 1** | [`Data_preprocessing_and_MLP_creation.py`](Data_preprocessing_and_MLP_creation.py) | Data loading, preprocessing, dataset splitting, scaling, and MLP creation |
| **Cell 2** | [`MLP_training.py`](MLP_training.py) | MLP training using the prepared training and validation datasets |
| **Cell 3** | [`MLP_evaluation_model_saving.py`](MLP_evaluation_model_saving.py) | Model evaluation, performance visualization, and saving the final trained model and scalers |

### Model Development Cycle

During model development, execute the cells in the following sequence:

```text
Cell 1 → Cell 2 → Cell 3
                  │
                  ↓
           Evaluate Results
                  │
          ┌───────┴───────┐
          │               │
    Tune Required     Model Finalized
          │               │
          ↓               ↓
   Modify Cell 1      Save Model
          │           and Scalers
          ↓
   Cell 1 → Cell 2 → Cell 3
```

Once the model achieves satisfactory performance, save the final trained model and scalers using Cell 3. The saved model can then be loaded during the subsequent **NSGA-II integration** stage for surrogate-based evaluation of candidate finger designs.
