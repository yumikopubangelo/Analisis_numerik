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

## Covered Topics (Planned)

Based on a standard Numerical Analysis syllabus:

### Root Finding (Priority)
- Bisection Method  
- Regula Falsi  
- Newton–Raphson  
- Secant Method  

### Error & Convergence Analysis
- Absolute error
- Relative error
- Iterative error
- Tolerance-based stopping criteria

### Interpolation (Planned)
- Lagrange Polynomial
- Newton Polynomial
- Interpolation error

### Series Expansion
- Taylor Series

### Future Extensions
- Numerical Integration (Trapezoidal, Simpson)
- Numerical Differentiation
- Ordinary Differential Equations (Euler, Runge–Kutta)

---

## Key Features

- Step-by-step iteration tables
- Error calculation at each iteration
- Automatic convergence detection
- Graph visualization:
  - Function plots
  - Error vs iteration
- Beginner-friendly explanations in simple language

---

## Project Structure

```

numerical-analysis-app/
│
├── app.py                # Main Streamlit app
│
├── core/                 # Numerical computation logic
│   ├── errors/           # Error analysis utilities
│   ├── root_finding/     # Root-finding methods
│   ├── interpolation/    # Interpolation methods
│   └── utils/            # Helper functions
│
├── ui/                   # UI components and visualization
│
├── docs/                 # Theory and usage documentation
│
├── tests/                # Simple validation tests
│
├── requirements.txt
└── README.md

````

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
````

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

* All methods show calculation steps
* No automatic "answer-only" output
* Users are encouraged to understand and explain results in their own words

---

## Intended Users

* Undergraduate students studying Numerical Analysis
* Beginners struggling with iterative methods
* Anyone who wants to visualize numerical computation processes

---

## Development Philosophy

* Simple > Complex
* Understandable > Optimized
* Finished > Perfect

This project prioritizes clarity, correctness, and educational value.

---

## License

This project is developed for educational purposes.
Commercial use requires permission from the author.

---

## Author

Developed by a student as a learning and teaching aid for Numerical Analysis.


