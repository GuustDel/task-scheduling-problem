import cpmpy as cp
import numpy as np
from fractions import Fraction

# Automated tasks (for cycle time calculation)
automated_tasks = [
    "Lay-up machine", 
    "Bussing machine", 
    "Laminator"
]

# Sequential tasks (with base times)
sequential_tasks = [
    "Wash Glass", 
    "Lay EVA",
    "Lay-up quality Check",
    "Manual Soldering", 
    "Closing", 
    "Poetsen", 
    "Flashen",
]

# Non-sequential tasks (no base time required)
non_sequential_tasks = [
    "Stringing",
    "Operate Lay-up Machine",
    "Operate Bussing Machine",
    "Operate Laminator"
]

# Combined list of all tasks for modeling and skill assignment.
tasks_for_model = sequential_tasks + non_sequential_tasks

# Default base times for sequential tasks (as floats, e.g. 4.5 means 9/2)
seq_times = [5.0, 4.0, 3.5, 3.0, 4.0, 3.0, 2.5]
# (There is no base time for "Stringing" because its only constraint is to assign one worker.)

# Default cycle times for automated tasks
auto_times = [4.0, 3.5, 4.5]  # the slowest is 5.0

# Define a default number of available workers.
# (Assume that each station is manned concurrently so the same worker cannot cover two stations.)
worker_names = ["Arben", "Jamil", "Khairullah", "Fazli", "Mohammedsalih", "Singh", "Chance", "Tashrif", "Shahidullah", "Himmat", "Benda", "Shams", "Beata", "Roger", "Raphael", "Serhii", "Sabba", "Fahim", "Mahmoud", "Fanuel", "Tedros", "Latifi", "Oksana", "Romy", "Zakhel", "Abdul", "Farhadullah"]
default_num_workers = len(worker_names)

skill_matrix = [
    # wash glass, lay EVA, lay-up quality check, manual soldering, closing, poetsen, flashen, stringing, operate lay-up, operate bussing, operate laminator
    [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0], # Arben
    [1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0], # Jamil 
    [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0], # Khairullah
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], # Fazli
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], # Mohammedsalih
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1], # Singh
    [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0], # Chance
    [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0], # Tashrif
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], # Shahidullah
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], # Himmat
    [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], # Benda
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], # Shams
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], # Beata
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1], # Roger
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1], # Raphael
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], # Serhii
    [1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0], # Sabba
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], # Fahim
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], # Mahmoud
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], # Fanuel
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], # Tedros
    [0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0], # Latifi
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], # Oksana
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0], # Romy
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0], # Zakhel
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # Abdul
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0] # Farhadullah
]

# A large constant for lexicographic weighting.
LARGE_WEIGHT = default_num_workers + 1

"""
Build and solve the model.

Parameters:
    seq_times: list of floats for each sequential task.
    auto_times: list of floats for each automated task.
    skill_matrix: 2D list (num_workers x total_tasks) of booleans.
                The columns correspond to tasks_for_model (sequential tasks first, then non-sequential tasks).
    preferred_list: list of booleans (length=num_workers) indicating preferred workers.
    
Returns:
    A dictionary with solution details or None if no solution is found.
"""
n_seq = len(seq_times)                   # number of sequential tasks
total_tasks = len(tasks_for_model)        # total tasks (sequential + non-sequential)
num_workers = len(skill_matrix)

lay_eva_index = sequential_tasks.index("Lay EVA")
operate_layup_index = tasks_for_model.index("Operate Lay-up Machine")
layup_quality_index = sequential_tasks.index("Lay-up quality Check")
operate_bussing_index = tasks_for_model.index("Operate Bussing Machine")

# Compute the bottleneck cycle time T from the automated tasks.
# Use Fraction to represent numbers as rational numbers.
T_frac = max([Fraction(str(t)) for t in auto_times])

# Decision variables:
# For each task m in tasks_for_model, let x[m] be the number of workers assigned.
# For sequential tasks, x[m] must be chosen to satisfy the effective time constraint.
# For non-sequential tasks (e.g. Stringing) we force x[m] == 1.
x = cp.intvar(1, num_workers, shape=total_tasks)

# For each worker i and task m, assign[i, m] is binary: 1 if worker i is assigned to task m.
assign = cp.boolvar(shape=(num_workers, total_tasks))

model = cp.Model()

# Constraint 1: For each task, the number of assigned workers equals x[m].
for m in range(total_tasks):
    model += cp.sum(assign[i, m] for i in range(num_workers)) == x[m]

# Constraint 2: Each worker can be assigned to at most one task.
for i in range(num_workers):
    model += (sum(assign[i, m] for m in range(total_tasks) if m not in [operate_bussing_index, operate_layup_index]) <= 1)

# Constraint 3: A worker may only be assigned to a task if they are skilled.
for i in range(num_workers):
    for m in range(total_tasks):
        if not skill_matrix[i][m]:
            model += (assign[i, m] == 0)

# Constraint 4: Task-specific constraints.
for m in range(total_tasks):
    if m < n_seq:
        # For sequential tasks, enforce: base_time <= T * x[m]
        # For "Wash Glass" (assumed at index 0) we need to wash 2 glasses per panel.
        m_frac = Fraction(str(seq_times[m]))
        multiplier = 2 if m == 0 else 1
        model += (multiplier * m_frac.numerator * T_frac.denominator <= T_frac.numerator * m_frac.denominator * x[m])

# Constraint 5: Additional constraints for non-sequential tasks.
model += cp.sum([assign[i, lay_eva_index] & assign[i, operate_layup_index] for i in range(num_workers)]) >= 1
model += cp.sum([assign[i, layup_quality_index] & assign[i, operate_bussing_index] for i in range(num_workers)]) >= 1

# --- Introduce worker "used" variables for preference tracking ---
used = cp.boolvar(shape=num_workers)
for i in range(num_workers):
    model += (cp.sum([assign[i, m] for m in range(total_tasks)]) >= 1).implies(used[i])

# --- Objective: Minimize total workers used (primary) and then penalize using non-preferred workers.
obj = LARGE_WEIGHT * sum(used[i] for i in range(num_workers)) \
            + sum(used[i] for i in range(num_workers))
model.minimize(obj)

if model.solve():
    print(assign.value())
else:
    print("No solution found.")