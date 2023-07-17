# Deep Learning for Neonatal Brain Injury Grading
***
This repository contains a subset of the source used during *Emmet Horgan's* final year project.  
There are essentially two sections to the repository, the pre-processing functions and the deep learning functions. The purpose of this description is not to explain the theory behind the pre-processing or deep learning, this can be found in the actual [dissertation](./dissertation.pdf), but instead to provide a short guide to the repository and its structure itself. 

### [`dl`](dl.py) 
This file is the highest abstraction for training and testing deep learning networks. The file reads data from a predetermined path not included in this repository. Several arguments can be optionally passed to the main function `run` to alter the training and testing such as the model to use, the optimizer, the number of epochs, batch size etc. The function can also optionally perform k-folds cross validation. 

### [`anser`](anser) 
This is a python package which wraps a lot of generally useable functions into a single place. This contains all of the pre-processing required along with a lot of the functions required for training and testing neural networks. 

#### [`anser.parse`](anser/parse.py)
This module contains a class with several methods for parsing the `.mat` files which contained the raw HRV data (not contained in this repository). 

#### [`anser.interpolate`](anser/interpolate.py)
This module contains a class which inherits from the one defined in [anser.parse](anser/parse.py). It provides additional functionality required for pre-processing like interpolation for resampling and additional methods. 

#### [`anser.driver`](anser/driver.py)
This module provides a function which encapsulates all of the pre-processing functions into one which can be called to employ the full pre-processing pipeline. 

#### [`anser.dl`](anser/dl.py)
This module contains many functions and classes for use with training, testing and also loading data for use with neural networks.

#### [`anser.transforms`](anser/transforms.py)
This module defines many custom transforms which can be used to transform the input images to the network. The interface for interaction with the transforms is through a dictionary which contains a functions as values in a dictionary.

#### [`anser.plotting`](anser/plotting.py)
This module contains many functions for plotting mostly for use in training and testing. Some of the functions here maybe very messy or not actual function at all due to last minute hacking. The [dl](dl.py) script does depend on some of the plotting functions as the training and testing curves are automatically plotted and saved.

#### [`anser.mfuncs`](anser/mfuncs.py) 
This package contains several modules which act as a wrapper for some MATLAB functions used in pre-processing.

#### [`anser/mfuncs`](anser/mfuncs) 
This folder contains some MATLAB `.m` files which are the actual implementations of the functions wrapped in [anser.mfuncs](anser/mfuncs.py).

#### [`anser.filemanage`](anser/filemanage.py)  
This file contains a lot of useful functions for file manipulation and some classes that are very useful for logging purposes during training. A lot of the functions in this module are essential for performing k-folds cross-validation.





