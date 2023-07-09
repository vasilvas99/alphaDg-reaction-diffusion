#!/usr/bin/env python3

import numpy as np
import ufl
from dolfinx import io, mesh
from dolfinx.fem import Function, FunctionSpace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc
from ufl import dot, dx, grad

r"""
Here we "flip" the problem and write the equations for aDG and diffusion in terms of supersaturation, since it's more natural to think about the supersation of the solution in this case:

$$
    \frac{\partial \theta}{\partial t} = D_c \nabla^2 \theta + 2*D*(1-\theta)^{(D-1)/D}\theta^g
$$

The weak form is then (assuming no-flux (Neumann) b.c.):

$$
    \left(\frac{\partial \theta}{\partial t}, v \right) = - D_c \left( \nabla \theta, \nabla v\right )_{\Omega} +  (2*D*(1-\theta)^{(D-1)/D}\theta^g, v) 
$$

Further, using a BDF for time-discretization, we arrive at the non-linear semi-discrete variational problem:

Given $\theta^{n}$, find $\theta^{n+1}$ for all $v \in V$ such that:

$$
    F_{n+1}(\theta; v) = (\theta^{n+1}, v)_\Omega - (\theta^{n}, v)_\Omega + D_c \Delta t \left( \nabla \theta^{n+1}, \nabla v\right )_\Omega - \Delta t \left( (1-\theta)^{(D-1)/D}\theta^g, v  \right)_\Omega = 0
$$
"""


######### Model Parameters #########
alpha_D = 2  # Dimensionality parameter from aDg (should be <= mesh dimension)
alpha_g = 1  # growth order
alpha_tau = 1  # time scale

D_coef = 0  # Diffusion coefficient

####### Simulation Constants #######
t = 0  # Start time
T = 2  # End time
num_steps = 1000  # num timesteps
dt = (T - t) / num_steps
mesh_size = 10
num_fe = 100


# #### Initial condition function ####
def initial_super_sat_cond(space_point: np.ndarray):
    return 1 + space_point[0] * 0.0


# 2D rectangular mesh with Lagrange elements
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[(0, 0), (mesh_size, mesh_size)],
    n=[num_fe, num_fe],
    cell_type=mesh.CellType.quadrilateral,
)
P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 2)
FE = FunctionSpace(msh, P1)

v = ufl.TestFunction(FE)

theta = Function(FE)
theta0 = Function(FE)

# zero out solution array
theta.x.array[:] = 0.0

# i.c. set
theta.interpolate(initial_super_sat_cond)
theta.x.scatter_forward()

# set up variational problem
react_theta = 2*alpha_D*((1 - theta) ** ((alpha_D - 1) / alpha_D)) * (theta**alpha_g)
F = (
    theta * v * dx
    - theta0 * v * dx
    # + D_coef * dt * dot(grad(theta), grad(v)) * dx
    - dt * react_theta * v * dx
)
problem = NonlinearProblem(F, theta)

# set up non-linear solver
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-12
solver.error_on_nonconvergence = False
solver.max_it = 200
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()

file = io.XDMFFile(MPI.COMM_WORLD, "aDG_diffusion_output.xdmf", "w")
file.write_mesh(msh)
file.write_function(theta, t)
theta0.x.array[:] = theta.x.array

while t < T:
    t += dt
    r = solver.solve(theta)
    print(f"Step {int(t/dt)}: num iterations: {r[0]}")
    if not r[1]:
        print("Newton iteration failed to converge! Exiting")
        break
    theta0.x.array[:] = theta.x.array
    file.write_function(theta, t)

file.close()
