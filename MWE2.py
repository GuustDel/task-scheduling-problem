import cpmpy as cp
import numpy as np

num_workers = 5
n_seq = 7
efficiency = [100, 90, 80, 70, 60, 50, 40]
x = cp.intvar(1, num_workers, shape=n_seq)
occupation = cp.intvar(0, 100, shape=(num_workers, n_seq))

# Define an array of fixed values
efficiency_multiplier = np.zeros((num_workers, n_seq), dtype=int)
for i in range(num_workers):
    for j in range(n_seq):
        efficiency_multiplier[i, j] = int(100 * (efficiency[j]/100)**(i) // 1)

# Add the element constraint to enforce x == values[i]
for m in range(n_seq):
    requred = -((-10000) // (18 * cp.Element(efficiency_multiplier, x[m]-1)))
    model = cp.Model(cp.sum(occupation[i, m] for i in range(num_workers)) == requred)

# Solve the model
if model.solve():
    print("Solution:")
    print("i =", occupation.value())
    print("x =", x.value())
else:	
    print("No solution found. Status:", model.status())