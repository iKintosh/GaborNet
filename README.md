# GaborNet

[![Coverage][coverage-image]][coverage-url]
[![Build Status][travis-badge]][travis-url]

This research on deep convolutional neural networks proposes a modified architecture that focuses on improving 
convergence and reducing training complexity. The filters in the first layer of network are constrained to fit the 
Gabor function. The parameters of Gabor functions are learnable and updated by standard backpropagation techniques. 
The proposed architecture was tested on several datasets and outperformed the common convolutional networks


[travis-url]: https://travis-ci.com/iKintosh/GaborNet
[travis-badge]: https://travis-ci.com/iKintosh/GaborNet.svg?branch=master
[coverage-image]: https://codecov.io/gh/iKintosh/GaborNet/branch/master/graphs/badge.svg
[coverage-url]: https://codecov.io/gh/iKintosh/GaborNet