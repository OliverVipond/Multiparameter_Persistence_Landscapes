# Multiparameter Persistence Landscapes

Python scripts for the computation of [multiparameter persistence landscapes](http://jmlr.org/papers/v21/19-054.html) for 2 parameter persistence modules from the .rivet files computed by [RIVET](https://github.com/rivetTDA/rivet#rivet). The scripts make use of the [pyrivet](https://github.com/rivetTDA/rivet-python) package (a Python API for interacting with [RIVET](https://github.com/rivetTDA/rivet#rivet)).

- ``` multiparameter_landscape.py ``` Defines a multiparameter_landscape class initialised with a .rivet file corresponding to a 2 parameter persistence module.
- ```multiparameter_landscape_plotting.py``` Defines functions to plot landscapes and produce interactive plots of 2D point clouds and their associated multiparameter landscapes. 
- ```one_parameter_classes.py``` Defines functions to plot landscapes and produce interactive plots of 2D point clouds and their associated multiparameter landscapes. 
- ```one_parameter_plotting.py``` Defines functions to plot one parameter landscapes and produce interactive plots of 2D point clouds and their associated single parameter barcodes and landscapes. 
- ```matching_distance.py, hera.py, barcode.py, rank.py, rivet.py``` scripts taken from [pyrivet](https://github.com/rivetTDA/rivet-python).

For those unfamiliar with persistent homology see these [introductory slides](https://olivervipond.github.io/Persistent_Homology_with_Noise/#/) which give an intuitive introduction. The [slides](https://olivervipond.github.io/Persistent_Homology_with_Noise/#/) also provide an introduction as to how one can use multiparameter persistence to filter out noise.

## Example Usage

We include a [jupyter notebook](https://olivervipond.github.io/Multiparameter_Persistence_Landscapes/ExampleNotebook/Examples.html) displaying the interactive plotting capability of the scripts in this repository.

The interactive plots in the notebook are built upon a visualization library [Bokeh][1], and a number of software packages developed by the Topological Data Analysis community. 

- One parameter persistence computations are run using [Ripser][2] founded by [Ulrich Bauer][3]
- Two parameter persistence computations are run using [RIVET][4] founded by [Michael Lesnick][5] and [Matthew Wright][6]
- This notebook was written by [Oliver Vipond][7]

[1]: https://docs.bokeh.org/en/latest/index.html#
[2]: https://ripser.scikit-tda.org/
[3]: https://ulrich-bauer.org/
[4]: https://github.com/rivetTDA/rivet
[5]: https://www.albany.edu/~ML644186/
[6]: https://www.mlwright.org/
[7]: https://www.olivervipond.com/

