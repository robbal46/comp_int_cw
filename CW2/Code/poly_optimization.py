import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from math import sqrt

from number_optimization import NumberOptimization


target_coefficients = [25, 18, 31, -14, 7, -19]

data = []
for c in target_coefficients:
    n = NumberOptimization(c, fitness_func='value', selection_type='roulette', termination_type='value', retain=0.1, random_select=0.05, mutate=0.01)
    n.create_population(100)
    n.optimize(3)

    data.append(n.pop_history)


max_list = max([len(l) for l in data])

data_array = np.empty((len(target_coefficients), max_list))

for i, d in enumerate(data):
    # Some pop history arrays may be shorter, as calcuation stops when target reached
    # Want to create an 2s array, so pad rows to make them the same length
    data_array[i] = np.pad(d, (0, max_list-len(d)), 'constant', constant_values=(0, d[-1]))

# Transpose so each row is made up of coefficients from each generation
data_array = np.transpose(data_array)

# Plot target polynomial
poly = np.polynomial.polynomial.Polynomial(np.flip(target_coefficients))
x_target, y_target = poly.linspace(100, [-100, 100])
plt.plot(x_target, y_target, label='Target Solution')

# plot each polynomial
for gen, row in enumerate(data_array):
    coeffs = np.flip(row) # Flip so coefficients are lowest -> highest order
    poly = np.polynomial.polynomial.Polynomial(coeffs)

    x,y = poly.linspace(100, [-100, 100])

    plt.plot(x, y, label=f'Gen {gen+1}')



plt.title('Polynomial Plots')
plt.grid(True, linestyle='-')
plt.legend()
plt.show()


