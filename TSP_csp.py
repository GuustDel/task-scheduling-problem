#!/usr/bin/env python3
"""
A CPMpy model and Tkinter UI for allocating workers to manual tasks,
using integer (fraction) arithmetic and a soft preference objective.

There are two groups of manual tasks:
  - Sequential tasks (which must run at a rate matching the bottleneck automated task).
    Their effective time is (base time)/(#workers) and we enforce: 
         base_time <= (bottleneck cycle time) * (#workers)
  - Non-sequential tasks (e.g. "Stringing") which are independent.
    For such tasks, we require that exactly one worker is assigned.

Additionally, users may mark “preferred” workers so that when solving multiple projects
in a day the same workers are favored. The objective minimizes the number of workers used
and, secondarily, penalizes using non–preferred workers.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from cpmpy import *
import numpy as np
from fractions import Fraction

# === Data definitions ===

# Automated tasks (for cycle time calculation)
automated_tasks = [
    "Lay-up machine", 
    "Bussing machine", 
    "Laminator"
]

# Sequential tasks (with base times)
sequential_tasks = [
    "Wash Glass", 
    "Lay EVA Foil / Operate Lay-up Machine", 
    "Quality Check / Operate Bussing Machine", 
    "Finish Soldering", 
    "Lay Back Glass", 
    "Polish Glass", 
    "Measure Performance",
    "Operate Laminator"
]

# Non-sequential tasks (no base time required)
non_sequential_tasks = ["Stringing"]

# Combined list of all tasks for modeling and skill assignment.
tasks_for_model = sequential_tasks + non_sequential_tasks

# Default base times for sequential tasks (as floats, e.g. 4.5 means 9/2)
default_seq_times = [5.0, 4.0, 3.5, 3.0, 4.0, 3.0, 2.5, 1.0]
# (There is no base time for "Stringing" because its only constraint is to assign one worker.)

# Default cycle times for automated tasks
default_auto_times = [4.0, 3.5, 5.0]  # the slowest is 5.0

# Define a default number of available workers.
# (Assume that each station is manned concurrently so the same worker cannot cover two stations.)
default_num_workers = 32
worker_names = [f"Worker {i+1}" for i in range(default_num_workers)]

# A large constant for lexicographic weighting.
LARGE_WEIGHT = default_num_workers + 1

# === CPMpy Model building function ===

def solve_model(seq_times, auto_times, skill_matrix, preferred_list):
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
    
    # Compute the bottleneck cycle time T from the automated tasks.
    # Use Fraction to represent numbers as rational numbers.
    T_frac = max([Fraction(str(t)) for t in auto_times])
    
    # Decision variables:
    # For each task m in tasks_for_model, let x[m] be the number of workers assigned.
    # For sequential tasks, x[m] must be chosen to satisfy the effective time constraint.
    # For non-sequential tasks (e.g. Stringing) we force x[m] == 1.
    x = [intvar(1, num_workers, name=f"x_{m}") for m in range(total_tasks)]
    
    # For each worker i and task m, assign[i][m] is binary: 1 if worker i is assigned to task m.
    assign = [[boolvar(name=f"assign_{i}_{m}") for m in range(total_tasks)] for i in range(num_workers)]
    
    model = Model()
    
    # Constraint 1: For each task, the number of assigned workers equals x[m].
    for m in range(total_tasks):
        model += (sum(assign[i][m] for i in range(num_workers)) == x[m])
    
    # Constraint 2: Each worker can be assigned to at most one task.
    for i in range(num_workers):
        model += (sum(assign[i][m] for m in range(total_tasks)) <= 1)
    
    # Constraint 3: A worker may only be assigned to a task if they are skilled.
    for i in range(num_workers):
        for m in range(total_tasks):
            if not skill_matrix[i][m]:
                model += (assign[i][m] == 0)
    
    # Constraint 4: Task-specific constraints.
    for m in range(total_tasks):
        if m < n_seq:
            # For sequential tasks, enforce: base_time <= T * x[m]
            # For "Wash Glass" (assumed at index 0) we need to wash 2 glasses per panel.
            m_frac = Fraction(str(seq_times[m]))
            multiplier = 2 if m == 0 else 1
            model += (multiplier * m_frac.numerator * T_frac.denominator <= T_frac.numerator * m_frac.denominator * x[m])
        else:
            # For non-sequential tasks (e.g. "Stringing"), exactly one worker must be assigned.
            model += (x[m] == 1)
    
    # --- Introduce worker "used" variables for preference tracking ---
    used = [boolvar(name=f"used_{i}") for i in range(num_workers)]
    for i in range(num_workers):
        model += (used[i] == sum(assign[i][m] for m in range(total_tasks)))
    
    # --- Objective: Minimize total workers used (primary) and then penalize using non-preferred workers.
    p = [1 if preferred_list[i] else 0 for i in range(num_workers)]
    obj_expr = LARGE_WEIGHT * sum(used[i] for i in range(num_workers)) \
               + sum((1 - p[i]) * used[i] for i in range(num_workers))
    model.minimize(obj_expr)
    
    if model.solve():
        x_vals = [x[m].value() for m in range(total_tasks)]
        assign_matrix = [[assign[i][m].value() for m in range(total_tasks)] for i in range(num_workers)]
        used_vals = [used[i].value() for i in range(num_workers)]
        total_workers_used = sum(used_vals)
        
        sol = {
            "x": x_vals,
            "assignment": assign_matrix,
            "used": used_vals,
            "total_workers": total_workers_used,
            "T": T_frac
        }
        return sol
    else:
        return None

# === Tkinter UI ===

class ProductionLineUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Production Line Worker Allocation")

        # Create a canvas and a scrollbar
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel events to the canvas
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)

        # Create two main frames: one for parameters (left) and one for output (right)
        self.left_frame = ttk.Frame(self.scrollable_frame)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="n")

        self.right_frame = ttk.Frame(self.scrollable_frame)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # --- Automated Task Cycle Times ---
        auto_frame = ttk.LabelFrame(self.left_frame, text="Automated Task Cycle Times")
        auto_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.auto_entries = []
        for i, task in enumerate(default_auto_times):
            ttk.Label(auto_frame, text=f"{automated_tasks[i]}:").grid(row=i, column=0, sticky="w")
            entry = ttk.Entry(auto_frame, width=8)
            entry.grid(row=i, column=1, padx=5, pady=2)
            entry.insert(0, str(task))
            self.auto_entries.append(entry)
            ttk.Label(auto_frame, text="min/panel").grid(row=i, column=2, sticky="w", padx=5)

        # --- Sequential Task Base Times ---
        seq_frame = ttk.LabelFrame(self.left_frame, text="Sequential Task Base Times")
        seq_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.seq_entries = []
        for i, task in enumerate(sequential_tasks):
            ttk.Label(seq_frame, text=f"{task}:").grid(row=i, column=0, sticky="w")
            entry = ttk.Entry(seq_frame, width=8)
            entry.grid(row=i, column=1, padx=5, pady=2)
            entry.insert(0, str(default_seq_times[i]))
            self.seq_entries.append(entry)
            ttk.Label(seq_frame, text="min/(panel*worker)").grid(row=i, column=2, sticky="w", padx=5)

        # --- Worker Skill Matrix ---
        skill_frame = ttk.LabelFrame(self.left_frame, text="Worker Skill Matrix (All Tasks)")
        skill_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        ttk.Label(skill_frame, text="Worker \\ Task").grid(row=0, column=0, padx=3, pady=3)
        for m, task in enumerate(tasks_for_model):
            ttk.Label(skill_frame, text=task, wraplength=100).grid(row=0, column=m+1, padx=3, pady=3)
        self.skill_vars = []  # 2D list: [worker][task]
        for i in range(default_num_workers):
            row_vars = []
            ttk.Label(skill_frame, text=worker_names[i]).grid(row=i+1, column=0, padx=3, pady=3, sticky="w")
            for m in range(len(tasks_for_model)):
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
        pref_max_cols = 9  # Maximum number of columns for preference checkbuttons.
        for i in range(default_num_workers):
            col = i % pref_max_cols
            row = 1 + (i // pref_max_cols)
            var = tk.IntVar(value=0)  # default: not preferred
            cb = ttk.Checkbutton(pref_frame, text=worker_names[i], variable=var)
            cb.grid(row=row, column=col, padx=5, pady=5, sticky="w")
            self.pref_vars.append(var)

        # --- Solve Button ---
        solve_button = ttk.Button(self.left_frame, text="Solve Allocation", command=self.solve_and_display)
        solve_button.grid(row=4, column=0, padx=10, pady=10)

        # --- Results Text Area (in the right frame) ---
        self.result_text = tk.Text(self.right_frame, width=90, height=40)
        self.result_text.pack(expand=True, fill="both")

    def _on_mouse_wheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def solve_and_display(self):
        # Read automated task cycle times.
        try:
            auto_times = [float(e.get()) for e in self.auto_entries]
        except ValueError:
            messagebox.showerror("Input error", "Enter valid numbers for automated task cycle times.")
            return
        
        # Read sequential task base times.
        try:
            seq_times = [float(e.get()) for e in self.seq_entries]
        except ValueError:
            messagebox.showerror("Input error", "Enter valid numbers for sequential task base times.")
            return
        
        # Read the worker skill matrix.
        skill_matrix = []
        for i in range(default_num_workers):
            row = []
            for m in range(len(tasks_for_model)):
                row.append(bool(self.skill_vars[i][m].get()))
            skill_matrix.append(row)
        
        # Read worker preferences.
        pref_list = [bool(self.pref_vars[i].get()) for i in range(default_num_workers)]
        
        # Solve the model.
        sol = solve_model(seq_times, auto_times, skill_matrix, pref_list)
        self.result_text.delete("1.0", tk.END)
        if sol is None:
            self.result_text.insert(tk.END, "No feasible solution found.\n")
        else:
            # Display the bottleneck cycle time.
            T_frac = sol["T"]
            T_float = float(T_frac)
            self.result_text.insert(tk.END, f"Bottleneck cycle time: T = {T_frac} (~{T_float:.2f})\n")
            self.result_text.insert(tk.END, f"Total workers used: {sol['total_workers']}\n\n")
            
            # For each task, display assignment details.
            n_seq = len(sequential_tasks)
            for m, task in enumerate(tasks_for_model):
                assigned_workers = []
                for i in range(default_num_workers):
                    if sol["assignment"][i][m] == 1:
                        assigned_workers.append(worker_names[i])
                if m < n_seq:
                    # For sequential tasks, display effective time.
                    eff_time = seq_times[m] / sol["x"][m]
                    self.result_text.insert(tk.END,
                        f"Task: {task}\n"
                        f"  Base time: {seq_times[m]} -> Workers assigned: {sol['x'][m]}, "
                        f"Effective time: {eff_time:.2f} (<= {T_float})\n"
                        f"  Assigned worker(s): {', '.join(assigned_workers) if assigned_workers else 'None'}\n\n"
                    )
                else:
                    # For non-sequential tasks, note that exactly one worker is assigned.
                    self.result_text.insert(tk.END,
                        f"Task: {task} (Non-sequential)\n"
                        f"  Assigned worker(s): {', '.join(assigned_workers) if assigned_workers else 'None'}\n\n"
                    )
            
            # Overall worker usage.
            self.result_text.insert(tk.END, "Worker Usage:\n")
            for i in range(default_num_workers):
                used_str = "USED" if sol["used"][i] == 1 else "not used"
                pref_str = " (preferred)" if self.pref_vars[i].get() == 1 else ""
                self.result_text.insert(tk.END, f"  {worker_names[i]}: {used_str}{pref_str}\n")
                

if __name__ == "__main__":
    app = ProductionLineUI()
    app.mainloop()
