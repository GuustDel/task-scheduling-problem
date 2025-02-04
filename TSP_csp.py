#!/usr/bin/env python3
"""
A CPMpy model and Tkinter UI for allocating workers to manual tasks,
using integer (fraction) arithmetic and a soft preference objective.

The production line is for integrated solar panels.
The manual tasks (with base times) and automated tasks (with cycle times)
are entered via the UI.  
Each manual task must run at least as fast as the bottleneck (slowest) automated task.
Manual task effective time = (base time) / (number of workers assigned),
so we enforce
    base_time <= (bottleneck cycle time) * (number of workers assigned)
(with the proper conversion to integers via fractions).

Additionally, users may mark “preferred” workers so that when
solving multiple projects in a day the same workers are favored.
The objective minimizes the number of workers used and, secondarily,
penalizes using non–preferred workers.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from cpmpy import *
import numpy as np
from fractions import Fraction

# === Data definitions ===

# Names for tasks (manual and automated)
manual_tasks = [
    "Wash Glass", 
    "Lay EVA Foil", 
    "Quality Check", 
    "Finish Soldering", 
    "Lay Back Glass", 
    "Polish Glass", 
    "Measure Performance"
]
automated_tasks = [
    "Place Solar Cells", 
    "Solder Solar Cells", 
    "Laminating"
]

# Default base times for manual tasks (as floats, e.g. 4.5 means 9/2)
default_manual_times = [5.0, 4.0, 3.5, 3.0, 4.0, 3.0, 2.5]
# Default cycle times for automated tasks
default_auto_times = [4.0, 3.5, 4.5]  # the slowest is 4.5

# Define a default number of available workers.
# (Assume that each station is manned concurrently so the same worker cannot cover two stations.)
default_num_workers = 10
worker_names = [f"Worker {i+1}" for i in range(default_num_workers)]

# A large constant for lexicographic weighting.
# (It must be larger than the maximum possible difference in the secondary term.)
LARGE_WEIGHT = default_num_workers + 1

# === CPMpy Model building function ===

def solve_model(manual_times, auto_times, skill_matrix, preferred_list):
    """
    Build and solve the model.
    
    Parameters:
      manual_times: list of floats for each manual task.
      auto_times: list of floats for each automated task.
      skill_matrix: 2D list (num_workers x num_manual_tasks) booleans.
      preferred_list: list of booleans (length=num_workers) indicating preferred workers.
      
    Returns:
      A dictionary with solution details or None if no solution is found.
    """
    num_manual = len(manual_times)
    num_workers = len(skill_matrix)
    
    # Compute the bottleneck cycle time T from the automated tasks.
    # Convert using Fraction (using string conversion to preserve decimals)
    T_frac = max([Fraction(str(t)) for t in auto_times])
    # For example, if T = 4.5, then T_frac = Fraction(9,2).
    
    # Decision variables:
    # For each manual task m, x[m] is the number of workers assigned.
    # At least 1 worker must be assigned; at most all available.
    x = [intvar(1, num_workers, name=f"x_{m}") for m in range(num_manual)]
    
    # For each worker i and manual task m, assign[i][m] is 1 if worker i is assigned to task m.
    assign = [[boolvar(name=f"assign_{i}_{m}") for m in range(num_manual)] for i in range(num_workers)]
    
    model = Model()
    
    # Constraint 1: For each manual task, the number of assigned workers equals x[m].
    for m in range(num_manual):
        model += (sum(assign[i][m] for i in range(num_workers)) == x[m])
    
    # Constraint 2: Each worker can be assigned to at most one manual task.
    for i in range(num_workers):
        model += (sum(assign[i][m] for m in range(num_manual)) <= 1)
    
    # Constraint 3: A worker may only be assigned to a task if they are skilled.
    for i in range(num_workers):
        for m in range(num_manual):
            if not skill_matrix[i][m]:
                model += (assign[i][m] == 0)
    
    # Constraint 4: Ensure that the effective time (base_time / x) is at most T.
    # In real numbers: manual_times[m] <= T * x[m].
    # We convert each float to a fraction.
    for m in range(num_manual):
        m_frac = Fraction(str(manual_times[m]))
        # The inequality: m_frac <= T_frac * x[m].
        # Multiply both sides by (T_frac.denominator * m_frac.denominator) to get an integer constraint:
        #   m_frac.numerator * T_frac.denominator <= T_frac.numerator * m_frac.denominator * x[m]
        model += (m_frac.numerator * T_frac.denominator <= T_frac.numerator * m_frac.denominator * x[m])
    
    # --- Introduce worker "used" variables for preference tracking ---
    used = [boolvar(name=f"used_{i}") for i in range(num_workers)]
    # Since each worker can only be assigned at most one manual task,
    # we can force used[i] to equal the (binary) sum over tasks.
    for i in range(num_workers):
        # Because sum(assign[i]) is 0 or 1 by constraint 2.
        model += (used[i] == sum(assign[i][m] for m in range(num_manual)))
    
    # In our model, note that the total number of workers used is
    #    total_used = sum(used[i])
    # which is equal to sum(x[m]) because assignments do not overlap.
    
    # --- Objective: Primary goal is to minimize the total workers used.
    # Secondary (soft) goal is to favor preferred workers.
    # Let p_i = 1 if worker i is preferred; 0 otherwise.
    # We penalize any worker used who is not preferred.
    p = [1 if preferred_list[i] else 0 for i in range(num_workers)]
    # The secondary penalty per worker i is: (1 - p[i]) * used[i]
    # Thus, our weighted objective is:
    #      minimize  LARGE_WEIGHT * sum(used[i]) + sum((1-p[i])*used[i]
    # With LARGE_WEIGHT chosen so that any change in the first term is much more important.
    obj_expr = LARGE_WEIGHT * sum(used[i] for i in range(num_workers)) \
               + sum((1 - p[i]) * used[i] for i in range(num_workers))
    model.minimize(obj_expr)
    
    # Solve the model.
    if model.solve():
        # Gather solution data.
        x_vals = [x[m].value() for m in range(num_manual)]
        assign_matrix = [[assign[i][m].value() for m in range(num_manual)] for i in range(num_workers)]
        used_vals = [used[i].value() for i in range(num_workers)]
        total_workers_used = sum(used_vals)
        
        # Build solution dictionary.
        sol = {
            "x": x_vals,
            "assignment": assign_matrix,
            "used": used_vals,
            "total_workers": total_workers_used,
            "T": T_frac  # the bottleneck cycle time (as Fraction)
        }
        return sol
    else:
        return None

# === Tkinter UI ===

class ProductionLineUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Production Line Worker Allocation")
        
        # Create two main frames: one for the parameters (left) and one for output (right)
        self.left_frame = ttk.Frame(self)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")
        
        self.right_frame = ttk.Frame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # --- Automated task cycle times ---
        auto_frame = ttk.LabelFrame(self.left_frame, text="Automated Task Cycle Times")
        auto_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.auto_entries = []
        for i, task in enumerate(automated_tasks):
            ttk.Label(auto_frame, text=f"{task}:").grid(row=i, column=0, sticky="w")
            entry = ttk.Entry(auto_frame, width=8)
            entry.grid(row=i, column=1, padx=5, pady=2)
            entry.insert(0, str(default_auto_times[i]))
            self.auto_entries.append(entry)
        
        # --- Manual task base times ---
        manual_frame = ttk.LabelFrame(self.left_frame, text="Manual Task Base Times")
        manual_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.manual_entries = []
        for i, task in enumerate(manual_tasks):
            ttk.Label(manual_frame, text=f"{task}:").grid(row=i, column=0, sticky="w")
            entry = ttk.Entry(manual_frame, width=8)
            entry.grid(row=i, column=1, padx=5, pady=2)
            entry.insert(0, str(default_manual_times[i]))
            self.manual_entries.append(entry)
        
        # --- Worker Skill Matrix ---
        skill_frame = ttk.LabelFrame(self.left_frame, text="Worker Skill Matrix (Manual Tasks)")
        skill_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        # Header row: manual task names
        ttk.Label(skill_frame, text="Worker \\ Task").grid(row=0, column=0, padx=3, pady=3)
        for m, task in enumerate(manual_tasks):
            ttk.Label(skill_frame, text=task, wraplength=100).grid(row=0, column=m+1, padx=3, pady=3)
        self.skill_vars = []  # 2D list: [worker][manual_task]
        for i in range(default_num_workers):
            row_vars = []
            ttk.Label(skill_frame, text=worker_names[i]).grid(row=i+1, column=0, padx=3, pady=3, sticky="w")
            for m in range(len(manual_tasks)):
                var = tk.IntVar(value=1)  # default: worker is skilled at every task
                cb = ttk.Checkbutton(skill_frame, variable=var)
                cb.grid(row=i+1, column=m+1, padx=3, pady=3)
                row_vars.append(var)
            self.skill_vars.append(row_vars)
        
        # --- Worker Preferences ---
        pref_frame = ttk.LabelFrame(self.left_frame, text="Worker Preferences")
        pref_frame.grid(row=3, column=0, padx=10, pady=5, sticky="ew")
        ttk.Label(pref_frame, text="Preferred Workers (check to prefer):").grid(row=0, column=0, padx=5, pady=5)
        self.pref_vars = []
        for i in range(default_num_workers):
            var = tk.IntVar(value=0)  # default: not preferred
            cb = ttk.Checkbutton(pref_frame, text=worker_names[i], variable=var)
            cb.grid(row=1, column=i, padx=5, pady=5)
            self.pref_vars.append(var)
        
        # --- Solve button ---
        solve_button = ttk.Button(self.left_frame, text="Solve Allocation", command=self.solve_and_display)
        solve_button.grid(row=4, column=0, padx=10, pady=10)
        
        # --- Results text area in the right frame ---
        self.result_text = tk.Text(self.right_frame, width=90, height=40)
        self.result_text.pack(expand=True, fill="both")
    
    def solve_and_display(self):
        # Read automated task cycle times.
        try:
            auto_times = [float(e.get()) for e in self.auto_entries]
        except ValueError:
            messagebox.showerror("Input error", "Enter valid numbers for automated task cycle times.")
            return
        
        # Read manual task base times.
        try:
            manual_times = [float(e.get()) for e in self.manual_entries]
        except ValueError:
            messagebox.showerror("Input error", "Enter valid numbers for manual task base times.")
            return
        
        # Read the worker skill matrix.
        skill_matrix = []
        for i in range(default_num_workers):
            row = []
            for m in range(len(manual_tasks)):
                row.append(bool(self.skill_vars[i][m].get()))
            skill_matrix.append(row)
        
        # Read worker preferences.
        pref_list = []
        for i in range(default_num_workers):
            pref_list.append(bool(self.pref_vars[i].get()))
        
        # Solve the model.
        sol = solve_model(manual_times, auto_times, skill_matrix, pref_list)
        self.result_text.delete("1.0", tk.END)
        if sol is None:
            self.result_text.insert(tk.END, "No feasible solution found.\n")
        else:
            # Display bottleneck cycle time (as fraction, also show float)
            T_frac = sol["T"]
            T_float = float(T_frac)
            self.result_text.insert(tk.END, f"Bottleneck cycle time: T = {T_frac} (~{T_float:.2f})\n")
            self.result_text.insert(tk.END, f"Total workers used: {sol['total_workers']}\n\n")
            
            # For each manual task, list assigned workers.
            for m, task in enumerate(manual_tasks):
                assigned_workers = []
                for i in range(default_num_workers):
                    if sol["assignment"][i][m] == 1:
                        assigned_workers.append(worker_names[i])
                eff_time = manual_times[m] / sol["x"][m]
                self.result_text.insert(tk.END,
                    f"Task: {task}\n"
                    f"  Base time: {manual_times[m]} -> Workers assigned: {sol['x'][m]}, "
                    f"Effective time: {eff_time:.2f} (<= {T_float})\n"
                    f"  Assigned worker(s): {', '.join(assigned_workers) if assigned_workers else 'None'}\n\n"
                )
            
            # Also list overall which workers were used and note preference.
            self.result_text.insert(tk.END, "Worker Usage:\n")
            for i in range(default_num_workers):
                used_str = "USED" if sol["used"][i] == 1 else "not used"
                pref_str = " (preferred)" if self.pref_vars[i].get() == 1 else ""
                self.result_text.insert(tk.END, f"  {worker_names[i]}: {used_str}{pref_str}\n")
                

if __name__ == "__main__":
    app = ProductionLineUI()
    app.mainloop()