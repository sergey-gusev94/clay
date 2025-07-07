#!/usr/bin/env python3
"""
Comprehensive test script for clay.py - tests multiple strategies with one solver.
"""

import clay

if __name__ == "__main__":
    print("Testing clay.py with multiple strategies...")
    
    # Test configuration - multiple strategies with one solver
    reformulation_strategies = [
        "gdp.bigm",
        "gdp.hull", 
        "gdp.hull_exact",
        "gdp.binary_multiplication"
    ]
    
    solver_configs = [
        {"solver": "gams", "subsolver": "gurobi"},  # Just one solver for testing
    ]
    
    # Test with just one problem and one metric first
    problem_name = "CLay0203"  # Smallest problem (3 rectangles, 2 circles)
    metric = "l1"
    
    clay.solve_constrained_layout_problem(
        problem_name=problem_name,
        metric=metric,
        reformulation_strategies=reformulation_strategies,
        solver_configs=solver_configs,
        time_limit=600,  # 10 minutes timeout for testing
    )
    
    print("Comprehensive test completed!") 