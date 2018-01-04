# MBAM
Code for the Manifold Boundary Approximation Method

## List of Files
- geodesic.py
- FiniteDifference.py
- exp_example.py
- exp_example.png
- MMR.py
- MMR_Plots.py
- README.md

## Description

Start by looking at exp_example.py.  This script defines a simple model which is the sum of two exponentials sampled at 3 points.  It defines a function to evaluate the model as well as its first and second derivatives with respect to the parameters.  It then imports functions for solving the geodesic equation.  It solves the geodesic equation and then plots the parameter values along the geodesic.  The output of this script should be similar to exp_example.png

Next, consider the MMR.py which defines a model (a Michaelis-Menten reaction) by solving a nonlinear ordinary differential equation.  This script defines a model by sampling by evaluating this model at three time points.  It also defines functions for calculating first and second derivatives.  Note that evaluating these derivatives involves solving the so-called sensitivity equations.  Alternatively, they can be estimated using finite differences. 

The sciprt MMR_Plots.py solves the geodesic equation for the MMR model and creates several plots to visualize the parameter space, parameter values along the geodesic, and the model manifold.
