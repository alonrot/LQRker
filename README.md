# Domain-informed multi-output Gaussian-process state-space model using harmonic features
Auxiliar repository for our [out-of-distribution project](https://github.com/alonrot/ood_project).

Features
+ Implementation of a multi-output Gaussian-process state-space model using the Karhunen–Loève expansion (also known as reduced-rank or weight-space view)
+ Library of domain-informed spectral densities, such as `Dubins Car Spectral Density` and `Van der Pol Spectral Density`
+ Included support for a multi-output Student-t process, which extends the single-dimension Student-t process proposed by [Shah et al., 2014](https://arxiv.org/pdf/1402.4306.pdf)
+ Supports `Random Fourier Features`, `Non-stationary Harmonic Features` and `Discrete Fourier Features`

Install with `pip install -e .`