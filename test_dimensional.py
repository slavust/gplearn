import gplearn.genetic
import numpy as np
from sklearn.utils import check_random_state

#random_state = check_random_state(1)
symbolic_regressor = gplearn.genetic.SymbolicRegressor(
    population_size=20,
    generations=100,
    tournament_size=5,
    init_depth=(20, 40),
    p_crossover=0.4,
    p_point_mutation=0.2,
    p_subtree_mutation=0.3,
    p_hoist_mutation=0.0,
    init_method='dimensional',
    function_set=('add', 'sub', 'mul', 'pow'),
    dimensional_max_power=20,
    n_jobs=6,
    verbose=2, low_memory=True)#, random_state=random_state)

S = [float(x*x) for x in range(1, 20)]
H = [float(x) for x in range(10, 40)]
V = [s*h for s, h in zip(S, H)]
SH = [[s, h] for s, h in zip(S, H)]

quantities_units = ((2,), (1,)) # m**2 and m
result_units = (3,)

result = symbolic_regressor.fit(np.array(SH, dtype=float), np.array(V, dtype=float), quantities_units, result_units)
print(symbolic_regressor._program)
