# gnudiscriminant

This program is a scientific library using the GSL and is build for calculating two variants of the Fisher’s linear discriminant analysis. The author recommends using the -l flag since this version of the LDA is able to calculate the exact solution to the Fisher’s criterion in the high-dimensional case in a short period of time. This variant of the LDA was first proposed by Hua Yu and Jie Yang in 2001. The algorithm works if the columns of the dataset represent the samples. Make sure to list all samples of one class behind each other. Then the first class will be denoted as class one.

gnudiscriminant further uses source code from the library gnupca written by Prof. Steve Hoffmann. Therefore, you have to install the libraries zlib from madler and the gnu mp bignum library.

The source code was part of a bachelor thesis at the Comp Biol AG of the Fritz Lipmann Institut.
