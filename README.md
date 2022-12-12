

# Signal-And-Systems
* [General info](#general-info)
* [Development Environment](#development-environment)


## General info
At the beginning, this is an assignment of ITU Signal And System Course to handle signal manipulations.

*The Discrete Signal Normalization and Convolution
 Discrete signals that have high dimensions and needed the resolution and 
is the best practice to learn how to analyze understanding fundementals about Signal and Systems. 
In this code is about the evaluation of the discrete signals. 
Importing these libraries is necessary are in the following;

`import numpy as np`
`import sys`
`import matplotlib.pyplot as plt`


## Development Environment
This Project is written on PyCharm 2018 version with Windows 10 Operating Sytstem.
Also, This application is tested  on Windows Powershell accurately. 


# Image Filtering using 2D-convolution
&nbsp;&nbsp;&nbsp;&nbsp;Convolution of two functions is one of the most important processes used in signal processing. In areas such as computer graphics and image processing, due to the discrete nature of functions (such as images), we deal with the discrete form of convolution. Among the applications of convolution in these fields, we can mention removing noise and extracting visual features for deep learning (artificial intelligence). Some of these applications have been implemented in this project.

<img src="https://github.com/mrezaamini/Image-Filtering-using-2D-convolution/blob/main/Content/sample.png" alt="example" width="400"/>

&nbsp;&nbsp;&nbsp;&nbsp;Some of the filters that are implemented in this project:
- Sharpness filter (valid/stride = 1)
- Horizontal edge filter (same/stride = 2)
- Embossing filter (valid/stride = 3)
- Gaissian filter (same/stride = 1)
