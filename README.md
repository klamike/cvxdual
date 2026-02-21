# cvxdual

`cvxdual` provides automatic dualization for CVXPY models.

- `dualize(problem)` builds the dual problem given the primal.
- `solve_dual(problem, **solve_kwargs)` solves the dual problem given the primal.

## Example

```python
import cvxpy as cp
from cvxdual import dualize, solve_dual

x = cp.Variable(name='x')
y = cp.Variable(name='y')
prob = cp.Problem(cp.Minimize(x - y), [x >= 1, y <= 1])

dualized = dualize(prob)

print('\n')
for i in range(len(dualized.meta.dual_variables)):
    var = dualized.meta.dual_variables[i]
    print(f"{var.variable} corresponds to {var.constraint}")
print('\n')
print(dualized.dual_problem)

solve_dual(prob)
print("solve_dual:", x.value, y.value)

x = cp.Variable(name='x')
y = cp.Variable(name='y')
prob = cp.Problem(cp.Minimize(x - y), [x >= 1, y <= 1])
prob.solve()
print("solve:", x.value, y.value)
```
