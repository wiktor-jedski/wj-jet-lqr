# wj-jet-lqr
A simulation of a jet engine using control library. 
## Features
* `JetEngine` class - represents a jet engine, contains `rhs_J18` method used to calculate derivative of angular velocity of the engine base on engine state equations and `rhs` helper function for solving differential equation
* `matrices_AB` helper function that defines A and B matrices for state equation
* `rkf45` function that implements Runge-Kutta method for solving differential equation
* Sample simulation showcasing the usage of `JetEngine` class for given initial conditions
## Requirements
See [requirements](requirements.txt)
## Documentation
[Final report](sprawozdanie_lqr.pdf) (in Polish)
