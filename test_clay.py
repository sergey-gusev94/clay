#!/usr/bin/env python3
"""
Test script for clay.py - tests one problem with one solver configuration.
"""

import clay

if __name__ == "__main__":
    print("Testing clay.py with a single problem...")
    
    # Test configuration - just one small problem with one solver
    reformulation_strategies = ["gdp.bigm"]  # Start with bigM only
    
    solver_configs = [
        {"solver": "gams", "subsolver": "gurobi"},  # Just one solver
    ]
    
    # Test with just one problem and one metric
    problem_name = "CLay0203"  # Smallest problem (3 rectangles, 2 circles)
    metric = "l1"
    
    clay.solve_constrained_layout_problem(
        problem_name=problem_name,
        metric=metric,
        reformulation_strategies=reformulation_strategies,
        solver_configs=solver_configs,
        time_limit=300,  # 5 minutes timeout for testing
    )
    
    print("Test completed!") 