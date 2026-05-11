# DCIL: DropConnect-based Causal Imitation Learning under Environment Heterogeneity

This repository contains the implementation of **DCIL**, as described in the paper:  
**"DCIL: DropConnect-based Causal Imitation Learning under Environment Heterogeneity"** by Xinyu Zhang, Shihua Li, and Rongjie Liu.

## Abstract

In many real-world scenarios, agents operate across heterogeneous environments where the underlying dynamics and data-generating processes vary. Standard reinforcement learning and imitation learning methods often fail in such settings as they typically assume stationarity and learn policies that overfit to environment-specific correlations. A key challenge is the presence of spurious correlations as observed states often contain both causal and non-causal features, with the latter introducing environment-specific biases that undermine generalization.

To address this problem, we propose **DropConnect-based Causal Imitation Learning (DCIL)**, a novel offline imitation learning framework designed to identify and exploit stable causal mechanisms across diverse environments. DCIL introduces a **gradient alignment constraint** to encourage the policy to align with causal structures shared across training environments. Additionally, it incorporates **DropConnect-based regularization** to inject stochastic perturbations into network weights, simulating parameter uncertainty and reducing reliance on unstable features.

We evaluate DCIL on synthetic benchmarks derived from OpenAI Gym control tasks, where non-causal features exhibiting spurious correlations are explicitly injected to simulate environmental heterogeneity. Experiments show that DCIL outperforms state-of-the-art imitation learning baselines, achieving superior generalization to unseen environments.

## Requirement

A `requirements.txt` with all the dependencies is provided.

* Make a conda environment and install dependencies: `pip install -r requirements.txt`
* Data can be self-generated or auto-downloaded.

## Repository Structure

```
DCIL/
│
├── GMM/                   # GMM synthetic data experiment
│   ├── data.py            # Data generation with spurious correlations
│   ├── plot.py            # Visualization utilities
│   ├── train_gmm.py       # Main script to run GMM experiments
│   └── results/           # Experiment results
│
└── IL/                    # DCIL on control tasks
    ├── agent/             # Policy and value network agents
    ├── experts_demo/      # Expert trajectories for imitation learning
    ├── network/           # Network architectures with DropConnect
    ├── parameters/        # Configuration parameters
    ├── results_DCIL/      # Experiment results
    ├── function.py        # Helper functions
    ├── loss.py            # DCIL loss functions
    ├── my_Buffer.py       # Replay buffer for offline learning
    ├── my_ENV.py          # Environment wrappers
    └── train.py           # Main script to run IL experiments
```

## Running and evaluating the model

### 1. Imitation Learning Experiments on OpenAI Gym Control Tasks

We conducted imitation learning experiments on several control tasks from OpenAI Gym, including **Acrobot**, **CartPole**, and **LunarLander**.
To create heterogeneous environments, we augmented each environment's original state space with additional non-causal variables. These spurious variables were designed to correlate with actions in a misleading way, making it challenging for algorithms to distinguish true causal relationships from spurious correlations.
This setup tests the robustness of imitation learning methods in learning stable policies that generalize across environments with varying spurious correlations.

```bash
cd IL
python train_IL.py
```

### Examples

Acrobot using 5 trajectory from each of 2 heterogeneous environments with 6-dimensional non-causal variables

```bash
cd IL
python train_IL.py env=acrobot buffer.traj_num=5 extra_dims=6 n_env=2
```

### 2. Synthetic Gaussian Mixture Model (GMM) Experiment

We evaluated the **DropConnect-based causal constraint** independently from imitation learning on a synthetic classification task based on the Gaussian Mixture Model (GMM).

* The **causal features** $x_1^e$ determine the labels $y^e$ stably across environments.
* The **non-causal features** $x_2^e$ induce spurious correlations that vary between environments.

This setup focuses on the effect of the DropConnect causal constraint alone. We compared our method (**DCIL**) against several baselines: **CoCo** (Constrained Causal Optimization), **IRM** (Invariant Risk Minimization), **ERM** (Empirical Risk Minimization), and **REx** (Risk Extrapolation).

```bash
cd GMM
python train_GMM.py --method DCIL
```
