
# Project Odyssey — Module v1.0: Foundational Autodiff Engine & Graph Mechanics

A clean-room reverse-mode automatic differentiation engine and neural network library built entirely from first principles using strictly NumPy as the sole numerical backend.

---

## 1. System Overview

Module v1.0 establishes the foundational computational substrate for Project Odyssey. It provides a dynamic, tape-based Directed Acyclic Graph (DAG) engine supporting reverse-mode automatic differentiation via Vector-Jacobian Products (VJPs), paired with an object-oriented neural network API and gradient descent optimization routines.

### Core Constraints & Invariants

* **Numerical Dependency**: Strictly `numpy>=1.26.0`[cite: 1, 2]. External autodiff engines or deep learning frameworks (PyTorch, JAX, TensorFlow, or autograd) are strictly forbidden[cite: 1, 2].
* **Execution Paradigm**: Eager dynamic computational graph recorded on the fly during the forward evaluation sweep.
* **Numerical Precision**: Strict 32-bit floating-point precision (`np.float32`) enforced across all tensors to prevent silent memory bloat and type mismatches.
* **Memory Lifecycle**: Lazy gradient allocation (`grad = None` until backpropagation) and non-destructive in-place parameter updates.

---

## 2. Global Odyssey Mandate[cite: 4]

Every module in Project Odyssey must satisfy three non-negotiable pillars before certification[cite: 4]:

* **The Engineering Pillar**: Continuous logging of hardware telemetry, including peak memory footprint, execution latency, and comparative throughput baselines[cite: 4].
* **The Reproduction Pillar**: A clean-room reproduction of **Maclaurin, Duvenaud, & Adams (2015) — *Autograd: Effortless Gradients in Pure Python***, verifying dynamic tape recording and Vector-Jacobian Product (VJP) closures[cite: 1].
* **The Scientific Pillar**: A structured literature review of foundational automatic differentiation papers (Linnainmaa 1970/1976; Rumelhart et al. 1986; Werbos 1974; Paszke et al. 2017).

---

## 3. Repository Architecture

The codebase implements a strict `src/` layout to isolate the package namespace and ensure reproducible packaging:

```text
odyssey/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated formatting, linting, typing, and test gates
├── src/
│   └── odyssey/
│       ├── __init__.py          # Public package exports (Tensor, Module)[cite: 2]
│       ├── py.typed             # PEP 561 inline typing marker[cite: 2]
│       ├── autograd/            # Dynamic computational graph & reverse engine[cite: 2]
│       │   ├── __init__.py
│       │   ├── graph.py         # Topological DFS sorting routines[cite: 2]
│       │   ├── ops.py           # Forward operators & local VJP backward closures[cite: 2]
│       │   └── tensor.py        # Core Tensor DAG node[cite: 2]
│       ├── nn/                  # Neural network abstractions[cite: 2]
│       │   ├── __init__.py
│       │   ├── activations.py   # Subgradient ReLU non-linearity[cite: 2]
│       │   ├── linear.py        # Linear layer with Kaiming/He Normal initialization[cite: 1, 2]
│       │   ├── loss.py          # Numerically stabilized FusedCrossEntropyLoss[cite: 1, 2]
│       │   └── module.py        # Base Module with parameter registration & zeroing[cite: 1, 2]
│       └── optim/               # Parameter optimization routines[cite: 1, 2]
│           ├── __init__.py
│           └── sgd.py           # In-place Stochastic Gradient Descent[cite: 1, 2]
├── tests/
│   ├── conftest.py              # Test fixtures and numerical audit helpers[cite: 2]
│   ├── test_autograd.py         # Branching DAG and diamond problem verification[cite: 1, 2]
│   ├── test_broadcasting.py     # Multi-dimensional broadcasting adjoint un-reduction[cite: 1, 2]
│   ├── test_gradcheck.py        # Centered finite-difference verification (< 1e-5)[cite: 1, 2]
│   ├── test_integration.py      # 2-layer MLP toy dataset convergence (> 90% accuracy)[cite: 1, 2]
│   ├── test_nn.py               # Layer parameter tracking and stability checks[cite: 2]
│   └── test_tensor.py           # Node initialization and operator overloads[cite: 2, 3]
├── .gitignore                   # Excludes .venv, build artifacts, and caches[cite: 2]
├── pyproject.toml               # Unified project metadata, tool configs, and dependencies[cite: 2]
└── README.md
```
