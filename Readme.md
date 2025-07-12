# Mechanistic PDE Networks for Discovery of Governing Equations

*Mechanistic PDE Networks for Discovery of Governing Equations*, Adeel Pervez, Efstratios Gavves, Francesco Locatello, *International Conference on Machine Learning (ICML) 2025*. \[[Arxiv](http://arxiv.org/abs/2502.18377)\]



## Running Experiments

### Requirements
- Pytorch
- Cupy
- Matplotlib
- Numpy and Scipy


To run the code create a conda environment by using the env.yml file.
Small test examples can be run in the given Jupyter Notebooks.
These include a simple fitting test of a damped sinusoidal wave, a solver comparison with scipy odeint, and 2-body trajectory predictions.

The test examples were run with the dense Cholesky solver. Set the Cholesky solver in config.py

### Examples

The `examples` directory contains Jupyter notebook examples for using the dense direct and sparse multigrid preconditioned solvers for simple PDEs. Also included is an example to fit a PDE to data.  Make sure the top-level directories are in the path before running the notebooks.

### Data

Data for the Burgers and Ginzburg Landau reaction diffusion equations are include in the `data` directory. The reactoin diffusion dataset was generated using the example script at [http://basilisk.fr/src/examples/ginzburg-landau.c](http://basilisk.fr/src/examples/ginzburg-landau.c). The full data is large so a small subset of the data is included in the directory.

### PDE Discovery

**1D Burgers Equation**. For the viscous 1D Burgers equation PDE discovery run

```
PYTHONPATH=. python discovery/burgers_dparam_viscous.py
```

**Ginzburg Landau Reaction Diffusion**. For the reaction-diffusion equation run
```
PYTHONPATH=. python discovery/ginzburg_landau.py
```

Logs are saved in the `logs` directory. 

### ODE Discovery 
The PDE code can also be used for ODE discovery. The repository includes code for learning the Kamani equation from Rheology.
```
PYTHONPATH=. python discovery/kamani.py
```

## Citation
```
@inproceedings{pervezmechnnpde2025,
  title={Mechanistic PDE Networks for Discovery of Governing Equations},
  author={Pervez, Adeel and Gavves, Efstratios and Locatello, Francesco},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```