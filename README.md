# Problem set-up

Here we "flip" the problem and write the equations for aDG and diffusion in terms of supersaturation, since it's more natural to think about the supersation of the solution in this case:
$$
    \frac{\partial \theta}{\partial t} = D_c \nabla^2 \theta + (1-\theta)^{(D-1)/D}\theta^g
$$
The weak form is then (assuming no-flux (Neumann) b.c.):

$$
    \left(\frac{\partial \theta}{\partial t}, v \right) = - D_c \left( \nabla \theta, \nabla v\right )_{\Omega} +  (1-\theta)^{(D-1)/D}\theta^g 
$$
Further, using a BDF for time-discretization, we arrive at the non-linear semi-discrete variational problem:

Given $\theta^{n}$, find $\theta^{n+1}$ for all $v \in V$ such that:
$$
    F_{n+1}(\theta; v) = (\theta^{n+1}, v)_{\Omega} - (\theta^{n}, v)_{\Omega} + D_c \Delta t \left( \nabla \theta^{n+1}, \nabla v\right )_{\Omega} - \Delta t \left( (1-\theta)^{(D-1)/D}\theta^g, v  \right)_{\Omega} = 0
$$

# How to run

## Installation

Requires a UNIX system with FEniCSx installed (via docker of otherwise). For installation instructions, go to: [Installation](https://github.com/FEniCS/dolfinx#installation).

## Running

```shell
    mpirun -np $(nproc) python3 ./adg_diff.py
```