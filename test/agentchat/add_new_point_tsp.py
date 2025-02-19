# filename: add_new_point_tsp.py
import random

from extensions.tsp import solve_tsp
from extensions.tsp_api import compare_costs, dists

# Step 1: Solve the TSP problem with the original distances
prev_cost = solve_tsp(dists)

# Step 2: Add a new point (let's say point 5) and assign random distances
new_point = 5
for existing_point in range(1, 5):  # Assuming existing points are 1 to 4
    distance = random.uniform(0, 5)  # Random distance between 0 and 5
    dists[(existing_point, new_point)] = distance
    dists[(new_point, existing_point)] = distance  # Assuming undirected graph

# Step 3: Solve the TSP problem again with the new distances
new_cost = solve_tsp(dists)

# Step 4: Compare the costs and print the result
gap = compare_costs(prev_cost, new_cost)
print("If we add a new point with random distances, the cost will change by", gap * 100, "percent.")
