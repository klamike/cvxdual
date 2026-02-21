# cvxdual

`cvxdual` provides automatic dualization for CVXPY models.

- `dualize(problem)` builds the dual problem given the primal.
- `solve_dual(problem, **solve_kwargs)` solves the dual problem given the primal.

## Example

```python
import cvxpy as cp
from cvxdual import dualize, solve_dual

x = cp.Variable(2)
prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])

dualized = dualize(prob)
print(dualized.dual_problem)

solve_dual(prob, solver=cp.SCS)
print(x.value)
```
