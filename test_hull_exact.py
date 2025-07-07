#!/usr/bin/env python3
"""
Test script for hull_exact strategy specifically.
"""

import clay

if __name__ == "__main__":
    print("Testing hull_exact strategy...")
    
    # Test configuration - just hull_exact strategy
    reformulation_strategies = ["gdp.hull_exact"]
    
    solver_configs = [
        {"solver": "gams", "subsolver": "gurobi"},
    ]
    
    # Test with CLay0203 problem
    problem_name = "CLay0203"
    metric = "l1"
    
    clay.solve_constrained_layout_problem(
        problem_name=problem_name,
        metric=metric,
        reformulation_strategies=reformulation_strategies,
        solver_configs=solver_configs,
        time_limit=300,
    )
    
    print("Hull exact test completed!") 