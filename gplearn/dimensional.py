import sympy as sp
import numpy as np

def random_power(max_power, random_state):
    print('Max power: {}'.format(max_power))
    numerator = random_state.randint(-max_power, max_power)
    print('Numerator: {}'.format(numerator))
    denominator = random_state.randint(1, max_power)
    print('Denominator: {}'.format(denominator))
    return sp.Rational(numerator, denominator)


class HelperTree(object):
    def __init__(self, node, children=[]):
        self.node = node
        self.children = children


def generate_mul_tree(root_node_units, quantities_units, max_depth, max_power, random_state):
    num_rows = len(quantities_units)
    assert num_rows > 0
    num_cols = len(quantities_units[0])
    assert num_cols > 0
    assert num_rows == len(root_units)
    A = sp.Matrix(quantities_units)
    X = sp.Matrix(num_cols, 1, sp.symbols('x0:{}'.format(num_cols)))
    Y = sp.Matrix(num_rows, 1, root_units)
    eq = list(A*X-Y)
    variables = list(X)

    solutions = sp.solve(eq, variables, dict=True)
    solution = solutions[0]
    print(solutions)

    dependent_variables = set(solution.keys())
    all_variables = set(variables)
    free_variables = all_variables - dependent_variables

    free_variable_values = {v: random_power(max_power, random_state) for v in free_variables}
    dependent_variable_values = {v : sp.nsimplify(solution[v].evalf(subs=free_variable_values)) for v in dependent_variables}
    all_variable_values = {**free_variable_values, **dependent_variable_values}

    powers = [sp.Rational(all_variable_values[var]) for var in variables]
    print(powers)

    non_zero_count = sum([1 if p != 0 else 0 for p in powers])
    if non_zero_count == 0:
        return HelperTree(1.0)

    def make_power_tree(base, power):
        base = HelperTree(base)
        power = HelperTree(power)
        return HelperTree('pow', [base, power])

    def make_balanced_mul_tree(multiplier_list):
        multiplier_count = len(multiplier_list)
        assert multiplier_count != 0

        if multiplier_count == 1:
            return multiplier_list[0]
        center_indx = int(multiplier_count / 2)

        left_subtree = make_balanced_mul_tree(multiplier_list[:center_indx])
        right_subtree = make_balanced_mul_tree(multiplier_list[center_indx:])
        root_node = HelperTree('mul', [left_subtree, right_subtree])

        return root_node

    feature_count = len(quantities_units[0])
    quantities_in_powers = [make_power_tree(feature_indx, power) for feature_indx, power in zip(range(feature_count), powers)]
    mul_tree = make_balanced_mul_tree(quantities_in_powers)

    return mul_tree


def debug_render_tree(helper_tree):
    import graphviz as gv
    dot = gv.Digraph(comment='Function tree debug')

    def traverse_tree(dot, parent, parent_id):
        for child, indx in zip(parent.children, range(len(parent.children))):
            child_id = parent_id + '_{}'.format(indx)
            dot.node(child_id, str(child.node))
            dot.edge(parent_id, child_id, label=str(indx))
            traverse_tree(dot, child, child_id)

    root_id = 'root'
    dot.node(root_id)
    traverse_tree(dot, helper_tree, root_id)

    dot.render('test_tree.gv', view=True)


if __name__ == '__main__':
    root_units = [1, -2, 0]
    quantities = [[1, 0, 2, -3, 0],
                  [-2, 0, 0, 0, 1],
                  [0, 1, 0, 1, 0]]

    mul_tree = generate_mul_tree(root_units, quantities, 10, 5, np.random.mtrand._rand)
    debug_render_tree(mul_tree)