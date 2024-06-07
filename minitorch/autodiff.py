from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1
    vals = list(vals)
    vals[arg] += epsilon
    forward = f(*vals)
    vals[arg] -= 2 * epsilon
    backward = f(*vals)
    return (forward - backward)/(2.0 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # topological sort by DFS traversal
    
    def visit(var: Variable):
        if var.unique_id in visited:
            if visited[var.unique_id] == 1:
                return
            if visited[var.unique_id] == -1:
                raise RuntimeError("The computational graph has a cycle!")
        visited[var.unique_id] = -1
        if not var.is_constant(): 
            for u in var.parents:
                visit(u)
        visited[var.unique_id] = 1
        order.append(var)
        
    visited = {}
    order = []
    visit(variable)
    return order[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # dummy case where the right-most variable is a leaf
    if variable.is_leaf(): 
        variable.accumulate_derivative(deriv)
        return

    intermediate = {} # dictionary to store vars and derivatives, {var.unique_id: List(var, d_out, derivative)}
    intermediate[variable.unique_id] = deriv

    for v in topological_sort(variable):
        if v.is_constant():
            continue
        if v.is_leaf():
            v.accumulate_derivative(intermediate[v.unique_id])
        else:
            # compute derivative based one d_out
            back = v.chain_rule(intermediate[v.unique_id])
            for u in back:
                u_var, u_deriv = u
                if u_var.unique_id in intermediate:
                    intermediate[u_var.unique_id] += u_deriv
                else:
                    intermediate[u_var.unique_id] = u_deriv

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
