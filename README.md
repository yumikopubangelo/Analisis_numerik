# Numerical Analysis Learning App
A web-based learning application designed to help students understand **Numerical Analysis** concepts through **step-by-step computation, error analysis, and visualization**.

This application focuses on *learning and understanding*, not just producing final answers.

---

## Purpose

This app is built to:
- Visualize numerical methods step-by-step
- Show iteration processes and convergence behavior
- Help beginners understand **why** a method works or fails
- Serve as a learning aid for Numerical Analysis courses

---

## Covered Topics

### Root Finding (Completed)
- Bisection Method
- Regula Falsi
- Newton-Raphson
- Secant Method

### Numerical Integration
- Trapezoidal Rule
- Simpson's Rule

### Numerical Differentiation
- Forward Difference
- Backward Difference
- Central Difference

### Interpolation (Completed)
- Lagrange Polynomial
- Newton Polynomial

### Series Expansion (Completed)
- Taylor Series

### Partial Differential Equations
- Biharmonic Plate Equation Solver
- Advection-Diffusion Equation Solver

### Error & Convergence Analysis
- Absolute error
- Relative error
- Iterative error
- Tolerance-based stopping criteria

---

## Key Features

- Step-by-step iteration tables
- Error calculation at each iteration
- Automatic convergence detection
- Graph visualization:
  - Function plots
  - Error vs iteration
  - 3D surface plots for PDE solutions
  - Contour plots
- Beginner-friendly explanations in simple language
- Unit conversion (kN to N) for engineering applications

---

## Project Structure

```
numerical-analysis-app/
│
├── app.py                    # Main Streamlit app
│
├── core/                     # Numerical computation logic
│   ├── analysis/            # Feature analysis
│   ├── differentiation/     # Numerical differentiation
│   ├── errors/             # Error analysis utilities
│   ├── integration/        # Numerical integration
│   ├── interpolation/      # Interpolation methods
│   ├── pde/               # PDE solvers (biharmonic, advection-diffusion)
│   ├── root_finding/       # Root-finding methods
│   ├── series/             # Taylor series
│   └── utils/              # Helper functions
│
├── ui/                      # UI components and visualization
│   ├── displays/           # Result display modules
│   ├── explanation.py      # Explanations
│   ├── input_form.py       # Input forms
│   ├── output_table.py     # Output tables
│   └── plots.py            # Plotting functions
│
├── docs/                    # Theory and usage documentation
│   ├── theory.md           # Mathematical theory
│   ├── usage.md            # Usage guide
│   └── features_documentation.md
│
├── tests/                   # Validation tests
│
├── quick_test.py           # Quick validation test
├── simple_test.py          # Simple test
├── validate_solver.py       # Solver validation
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Python**
- **Streamlit** (Web UI)
- NumPy
- SymPy
- Matplotlib
- Pandas

---

## How to Run

### 1. Clone the repository
```bash
git clone <repository-url>
cd numerical-analysis-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## Academic Ethics Notice

This application is intended as a **learning tool**, not a shortcut for assignments.

- All methods show calculation steps
- No automatic "answer-only" output
- Users are encouraged to understand and explain results in their own words

---

## Intended Users

- Undergraduate students studying Numerical Analysis
- Beginners struggling with iterative methods
- Engineering students learning plate theory
- Anyone who wants to visualize numerical computation processes

---

## Development Philosophy

- Simple > Complex
- Understandable > Optimized
- Finished > Perfect

This project prioritizes clarity, correctness, and educational value.

---

## License

This project is developed for educational purposes.
Commercial use requires permission from the author.

---

## Author

Developed by a student as a learning and teaching aid for Numerical Analysis.
