# Solvers and operators

Linear operators, the Newton and implicit-differentiation solvers and the iterative linear solvers. Guide: [Choosing a solver](../../guide/solvers.md).

## Operators

::: noodl.operators.base
    options:
      members:
        - LinearOperator
        - SparseAssembling
        - SolveResult
        - SolverStatus
        - as_operator

::: noodl.operators.dense
    options:
      members:
        - DenseOperator

::: noodl.operators.graph
    options:
      members:
        - GraphLaplacianOperator

::: noodl.operators.advection
    options:
      members:
        - AdvectionOperator

::: noodl.operators.sparse
    options:
      members:
        - SparseOperator

## Solvers

::: noodl.solvers
    options:
      members:

::: noodl.solvers.select
    options:
      members:
        - solve
        - scipy_importable
        - transport_solver_kwargs

::: noodl.solvers.newton
    options:
      members:
        - newton
        - NewtonResult
        - inner_solve_rtol
        - newton_tolerances

::: noodl.solvers.implicit
    options:
      members:
        - implicit_solve
        - adjoint
        - TransposeOperator
        - refuse_second_order
        - residual_vjp

::: noodl.solvers.iterative
    options:
      members:
        - pcg
        - gmres

::: noodl.solvers.scalar
    options:
      members:
        - solve_monotone

::: noodl.solvers.grounding
    options:
      members:
        - spd_certificate
        - spd_diagnosis
