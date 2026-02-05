#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from core.series.taylor_series import euler_method, taylor_ode_order2, runge_kutta_method


def test_ode_solvers():
    print("=== Testing ODE Solvers ===\n")
    
    # Test case 1: ODE where we know the exact solution
    # y' = x²y - y, y(0) = 1, x_eval = 2
    # Exact solution: y = exp(x³/3 - x) = exp(8/3 - 2) = exp(2/3) ≈ 1.9477
    func_str = "x**2*y - y"
    x0 = 0.0
    y0 = 1.0
    x_eval = 2.0
    n_steps = 100
    
    print(f"Test Case 1: {func_str}, y({x0}) = {y0}, x={x_eval}")
    
    # Euler method
    euler_result = euler_method(func_str, x0, y0, x_eval, n_steps=n_steps)
    print(f"\nEuler Method:")
    print(f"  Approximation: {euler_result['approximation']:.6f}")
    print(f"  Exact: {euler_result['exact']:.6f}")
    print(f"  Absolute Error: {euler_result['absolute_error']:.6e}")
    
    # Taylor ODE Order 2
    taylor_result = taylor_ode_order2(func_str, x0, y0, x_eval, n_steps=n_steps)
    print(f"\nTaylor ODE Order 2:")
    print(f"  Approximation: {taylor_result['approximation']:.6f}")
    print(f"  Exact: {taylor_result['exact']:.6f}")
    print(f"  Absolute Error: {taylor_result['absolute_error']:.6e}")
    
    # Runge-Kutta RK4
    rk4_result = runge_kutta_method(func_str, x0, y0, x_eval, n_steps=n_steps)
    print(f"\nRunge-Kutta RK4:")
    print(f"  Approximation: {rk4_result['approximation']:.6f}")
    print(f"  Exact: {rk4_result['exact']:.6f}")
    print(f"  Absolute Error: {rk4_result['absolute_error']:.6e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test case 2: ODE where we don't know the exact solution
    func_str = "x + y"
    x0 = 0.0
    y0 = 1.0
    x_eval = 1.0
    n_steps = 10
    
    print(f"Test Case 2: {func_str}, y({x0}) = {y0}, x={x_eval}")
    
    result = euler_method(func_str, x0, y0, x_eval, n_steps=n_steps)
    print(f"\nEuler Method:")
    print(f"  Approximation: {result['approximation']:.6f}")
    print(f"  Exact: {result['exact']}")
    print(f"  Absolute Error: {result['absolute_error']}")


if __name__ == "__main__":
    test_ode_solvers()
