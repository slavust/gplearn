from _program import _Program
import functions
import fitness
import numpy as np
import graphviz as gv


if __name__ == '__main__':
    function_set = set(functions._function_map.values())
    arities = {}
    for function in function_set:
        arity = function.arity
        arities[arity] = arities.get(arity, [])
        arities[arity].append(function)


    quantities = [[1, 0, 2, -3, 0],
                  [-2, 0, 0, 0, 1],
                  [0, 1, 0, 1, 0]]

    required_units = [1, -2, 0]

    program = _Program(
        list(functions._function_map.values()),
        arities,
        init_depth=(4, 7),
        init_method='dimensional',
        n_features=5,
        const_range=(-10000, 10000),
        metric=fitness._fitness_map,
        p_point_replace=0.1,
        parsimony_coefficient=0.9,
        random_state=np.random.mtrand._rand,
        dimensional_max_power=10,
        dimensional_required_units=required_units,
        dimensional_quantities_units=quantities,
        feature_names=['q1', 'q2', 'q3', 'q4', 'q5']
    )

    gv_dot_final = program.export_graphviz()
    source = gv.Source(gv_dot_final)
    source.render(
        filename='_Program.gv',
        view=True)