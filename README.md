# MBAM
Code for the Manifold Boundary Approximation Method


## Install

### Using pip

``` bash
pip install mbam
```

### From source

``` bash
git clone https://github.com/mktranstrum/MBAM.git
pip install ./MBAM
```


## Example

See [examples](https://github.com/mktranstrum/MBAM/tree/refactoring/examples).

* Start by looking at `exp_example.py`. This script defines a simple model
which is the sum of two exponentials sampled at 3 points. It defines a function
to evaluate the model as well as its first and second derivatives with respect
to the parameters. It then imports functions for solving the geodesic equation.
It solves the geodesic equation and then plots the parameter values along the
geodesic. The output of this script should be similar to `exp_example.png`

* Next, consider the `MMR.py` which defines a model (a Michaelis-Menten
reaction) by solving a nonlinear ordinary differential equation. This script
defines a model by sampling by evaluating this model at three time points. It
also defines functions for calculating first and second derivatives. Note that
evaluating these derivatives involves solving the so-called sensitivity
equations. Alternatively, they can be estimated using finite differences. </br>
The script `MMR_Plots.py` solves the geodesic equation for the MMR model
and creates several plots to visualize the parameter space, parameter values
along the geodesic, and the model manifold.


## Attribution

Please cite [Transtrum, Machta, and Sethna (2011)](https://link.aps.org/doi/10.1103/PhysRevE.83.036701)
and [Transtrum and Qiu (2014)](https://link.aps.org/doi/10.1103/PhysRevLett.113.098701)
if you find this code useful in your research.


## License

mbam is free software distributed under the MIT License; see the
[LICENSE](https://github.com/mktranstrum/MBAM//blob/master/LICENSE) file for
details.
