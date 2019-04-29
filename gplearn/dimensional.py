import sympy as sp
import numpy as np


def random_power(max_power, random_state):
    numerator = random_state.randint(-max_power, max_power)
    denominator = random_state.randint(1, max_power)
    return sp.Rational(numerator, denominator)


class HelperTree(object):
    def __init__(self, node, units, children=list()):
        self.node = node
        self.units = units
        self.children = children

    def flatten(self):
        flattened_tree = [(self.node, self.units)]
        for child in self.children:
            flattened_tree += child.flatten()
        return flattened_tree


def generate_mul_tree(root_node_units, quantities_units, max_power, function_map, random_state):
    num_rows = len(quantities_units)
    assert num_rows > 0
    num_cols = len(quantities_units[0])
    assert num_rows > 0
    assert num_cols == len(root_node_units)

    # solve linear equation system to find applicable powers of quantities
    unit_matrix = sp.Matrix(quantities_units)
    unit_matrix.transpose() # input form should be [quantity_units_1, ..., quantity_units_n]
    unknowns = sp.Matrix(num_rows, 1, sp.symbols('x0:{}'.format(num_cols)))
    free_coefficients = sp.Matrix(num_cols, 1, root_node_units)
    eq = list(unit_matrix*unknowns-free_coefficients)
    variables = list(unknowns)

    solutions = sp.solve(eq, variables, dict=True)
    if len(solutions) == 0:
        raise ValueError('Required units cannot be derived from given quantities')
    solution = solutions[0]

    dependent_variables = set(solution.keys())
    all_variables = set(variables)
    free_variables = all_variables - dependent_variables

    free_variable_values = {v: random_power(max_power, random_state) for v in free_variables}
    dependent_variable_values = {
        v: sp.nsimplify(
            solution[v].evalf(subs=free_variable_values)
        ) for v in dependent_variables}
    all_variable_values = {**free_variable_values, **dependent_variable_values}
    powers = [sp.Rational(all_variable_values[var]) for var in variables]

    all_powers_zeroes = all([p == 0 for p in powers])
    if all_powers_zeroes:
        return HelperTree(node=1.0, units=powers)

    def make_power_tree(base, base_units, power, pow_func):
        base = HelperTree(node=base, units=base_units)
        power = HelperTree(node=power, units=(0,)*len(base_units))
        powered_units = (unit_pow * power for unit_pow in base_units)
        return HelperTree(
            node=pow_func,
            units=powered_units,
            children=[base, power])

    def make_balanced_mul_tree(multiplier_list, mul_func):
        multiplier_count = len(multiplier_list)
        assert multiplier_count != 0

        if multiplier_count == 1:
            return multiplier_list[0]
        center_indx = int(multiplier_count / 2)

        left_subtree = make_balanced_mul_tree(multiplier_list[:center_indx], mul_func)
        right_subtree = make_balanced_mul_tree(multiplier_list[center_indx:], mul_func)
        left_units = left_subtree.node[1]
        right_units = right_subtree.node[1]
        assert len(left_units) == len(right_units)
        result_units = (left_unit_pow + right_unit_pow
                        for left_unit_pow, right_unit_pow in zip(left_units, right_units))
        root_node = HelperTree(
            node=mul_func,
            units=result_units,
            children=[left_subtree, right_subtree])

        return root_node

    feature_count = len(quantities_units[0])
    quantities_in_powers = [
        make_power_tree(
            feature_indx,
            quantities_units[feature_count],
            powers[feature_indx],
            function_map['pow'])
        for feature_indx in range(feature_count)]
    mul_tree = make_balanced_mul_tree(quantities_in_powers, function_map['mul'])

    return mul_tree


def generate_tree(
        root_node_units,
        quantities_units,
        depth,
        n_features,
        max_power,
        const_range,
        dimensional_functions_map,
        nondimensional_functions_list,
        random_state):
    base_unit_count = len(root_node_units[0])
    dimensional = (root_node_units != [0]*base_unit_count)

    # The only two functions that transform dimensionality are 'pow' and 'mul'
    # so minimum depth tree that produces required units from input quantities
    # is balanced tree with multiplications of quantities powers.
    # TODO: check:
    # Also, it seems to me that minimal multiplier count
    # tends to dependent variable count in equation system.
    # Anyway, I think here might be even not min but average multiplier count
    average_min_depth_multiplier_count = base_unit_count
    average_min_depth = int(np.log2(average_min_depth_multiplier_count)) + 1 # power node adds 1
    required_to_build_minimal_tree = depth <= average_min_depth

    if dimensional:
        if required_to_build_minimal_tree:
            allowed_options = ['mul_power_tree']
        else:
            allowed_options = ['add', 'sub', 'mul_nondimensional', 'mul_power_tree']
        option_indx = random_state.randint(len(allowed_options))
        option_choice = allowed_options[option_indx]
        if option_choice == 'add':
            left_subtree = generate_tree(
                root_node_units,
                quantities_units,
                depth-1,
                n_features,
                max_power,
                const_range,
                dimensional_functions_map,
                nondimensional_functions_list,
                random_state)
            right_subtree = generate_tree(
                root_node_units,
                quantities_units,
                depth-1,
                n_features,
                max_power,
                const_range,
                dimensional_functions_map,
                nondimensional_functions_list,
                random_state)
            assert left_subtree.units == right_subtree.units == root_node_units
            # TODO: isn't it required to add some nondimensional operations here?
            return HelperTree(
                node=dimensional_functions_map['add'],
                units=root_node_units,
                children=[left_subtree, right_subtree])
        elif option_choice == 'sub':
            left_subtree = generate_tree(
                root_node_units,
                quantities_units,
                depth-1,
                n_features,
                max_power,
                const_range,
                dimensional_functions_map,
                nondimensional_functions_list,
                random_state)
            right_subtree = generate_tree(
                root_node_units,
                quantities_units,
                depth-1,
                n_features,
                max_power,
                const_range,
                dimensional_functions_map,
                nondimensional_functions_list,
                random_state)
            assert left_subtree.units == right_subtree.units == root_node_units
            # TODO: isn't it required to add some nondimensional operations here?
            return HelperTree(
                node=dimensional_functions_map['add'],
                units=root_node_units,
                children=[left_subtree, right_subtree])
        elif option_choice == 'mul_nondimensional':
            left_subtree = generate_tree(
                [0]*base_unit_count,
                quantities_units,
                depth - 1,
                n_features,
                max_power,
                const_range,
                dimensional_functions_map,
                nondimensional_functions_list,
                random_state)
            right_subtree = generate_tree(
                root_node_units,
                quantities_units,
                depth-1,
                n_features,
                max_power,
                const_range,
                dimensional_functions_map,
                nondimensional_functions_list,
                random_state)
            assert right_subtree.units == root_node_units
            return HelperTree(
                node=dimensional_functions_map['mul'],
                units=root_node_units,
                children=[left_subtree, right_subtree])
        elif option_choice == 'mul_power_tree':
            # the only functions that can change expression dimensionality are mul and pow on quantities
            return generate_mul_tree(
                root_node_units,
                quantities_units,
                max_power,
                dimensional_functions_map,
                random_state
            )
        assert False
    else:
        if required_to_build_minimal_tree:
            allowed_options = ['constant', 'mul_power_tree']
        else:
            allowed_options = ['constant', 'mul_power_tree', 'nondim_func']
        option_indx = random_state.randint(len(allowed_options))
        option_choice = allowed_options[option_indx]

        unitless = root_node_units

        if option_choice == 'constant':
            # TODO: unitless features?
            number = random_state.uniform(*const_range)
            return HelperTree(node=number, units=unitless)
        elif option_choice == 'mul_power_tree':
            return generate_mul_tree(
                unitless,
                quantities_units,
                max_power,
                dimensional_functions_map,
                random_state)
        elif option_choice == 'nondim_func':
            operator_indx = random_state.randint(len(nondimensional_functions_list))
            operator = nondimensional_functions_list[operator_indx]
            operands = [generate_tree(
                        unitless,
                        quantities_units,
                        depth-1,
                        n_features,
                        max_power,
                        const_range,
                        dimensional_functions_map,
                        nondimensional_functions_list,
                        random_state) for _ in range(operator.arity)]
            return HelperTree(node=operator, units=unitless, children=operands)
        else:
            assert False


def debug_render_tree(helper_tree):
    import graphviz as gv
    dot = gv.Digraph(comment='Function tree debug')

    def traverse_tree(dot, parent, parent_id):
        for child, indx in zip(parent.children, range(len(parent.children))):
            child_id = parent_id + '_{}'.format(indx)
            print(str(child.node))
            if isinstance(child.node, float)\
                    or isinstance(child.node, int)\
                    or isinstance(child.node, sp.Rational):
                dot.node(child_id, str(child.node))
            else:
                dot.node(child_id, str(child.node.name))
            dot.edge(parent_id, child_id, label=str(indx))
            traverse_tree(dot, child, child_id)

    root_id = 'root'
    print(str(helper_tree.node))
    dot.node(root_id, helper_tree.node.name)
    traverse_tree(dot, helper_tree, root_id)

    dot.render('debug.gv', view=True)

def build_program(
        root_node_units,
        quantities_units,
        depth,
        n_features,
        max_power,
        const_range,
        function_list,
        random_state):
    function_names = [function.name for function in function_list]
    dimensional_function_names = ['mul', 'pow', 'add', 'sub']
    has_required_functions = all([name in function_names for name in dimensional_function_names])
    if not has_required_functions:
        raise ValueError("functions {} are required for dimensional analysis.".format(
            ", ".join(dimensional_function_names)))
    dimensional_function_indices = [function_names.index(func) for func in dimensional_function_names]
    dimensional_function_map = {name:function_list[indx]
                                for name, indx in zip(dimensional_function_names, dimensional_function_indices)}
    tree = generate_tree(
        root_node_units,
        quantities_units,
        depth,
        n_features,
        max_power,
        const_range,
        dimensional_function_map,
        function_list,
        random_state)

    debug_render_tree(tree)

    return tree.flatten()


if __name__ == '__main__':
    root_units = [1, -2, 0]
    quantities = [[1, 0, 2, -3, 0],
                  [-2, 0, 0, 0, 1],
                  [0, 1, 0, 1, 0]]

    #mul_tree = generate_mul_tree(root_units, quantities, 10, np.random.mtrand._rand)
    #debug_render_tree(mul_tree)
    whole_tree = generate_tree(
        root_units,
        quantities,
        12,
        10,
        5,
        [('log', 2), ('pow', 2), ('add', 2), ('sub', 2), ('mul', 2), ('sin', 1), ('cos', 1), ('tan', 1)],
        np.random.mtrand._rand)
    debug_render_tree(whole_tree)
