from numba import njit
import minitorch
import minitorch.tensor_ops

# MM
print("MATRIX MULTIPLY")
out, a, b = (
    minitorch.zeros((1, 10, 10)),
    minitorch.zeros((1, 10, 20)),
    minitorch.zeros((1, 20, 10)),
)
tmm = minitorch.tensor_ops.tensor_matrix_multiply

tmm(*out.tuple(), *a.tuple(), *b.tuple())