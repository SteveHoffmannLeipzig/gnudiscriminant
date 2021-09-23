# gnudiscriminant

This program is a scientific library using the GSL and is build for calculating two variants of the Fisher’s linear discriminant analysis. The author recommends using the -l flag since this version of the LDA is able to calculate the real solution for the LDA in the high-dimensional case in a short period of time. This variant of the LDA was first proposed by Hua Yu and Jie Yang in 2001. The algorithm works if the columns of the dataset represent the samples. Make sure to list all samples of one class behind each other. Then the first class will be denoted as class one.
