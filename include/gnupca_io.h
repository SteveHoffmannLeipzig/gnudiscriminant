/*
 *   gnupca 
 *   Copyright (C) Steve Hoffmann
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


/*
 *  gnupca_io.h
 *  
 *
 *  @author Steve Hoffmann
 *  @institute FLI
 *  
 */

#ifndef MAC
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>
#include <assert.h>
#include <tgmath.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_permute_matrix.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_eigen.h>
#include <zlib.h>

#include "stringutils.h"
#include "fileio.h"
#include "filebuffer.h"

/************************************
  *
  * guess file type from gz extensions
  * returns 1 if gzipped file
  *
  ***********************************/

char gnupca_isgz(char *fn);

/************************************
  *
  * open file and assign it co circ buffer
  * exit program on file open error
  *
  ***********************************/


char gnupca_fileopen(circbuffer_t *buf, char *fn) ;

/************************************
  *
  * Close device assigned to circular buffer.
  *
  * \param buf pointer to a circbuffer_t.
  *
  ***********************************/

void gnupca_fileclose(circbuffer_t *buf) ;

/************************************
  *
  * Read a tab-separated file with a matrix to double array.
  * 
  * The function cause the program to exit if matrix is malformed
  * or too large for memory.
  *
  * \param *in file name
  * \param *rows pointer to a size_t object to store row number
  * \param *cols pointer to a size_t object to store col number
  * \return array with matrix data
  *
  ***********************************/

double* csv2matrix (char *in, size_t *rows, size_t *cols);


/************************************
  *
  * Read a tab-separated file into an array of double array.
  * Each row of the file is an individual array.
  *
  * \param *in file name
  * \param *narr pointer to a size_t object storing the number of read arrays (in rows)
  * \param **nelems pointer to array with the lengths of each array (in cols)
  * \return array of arrays with data
  *
  ***********************************/

size_t** csv2arrays (char *in, size_t *narr, size_t **nelems);
void dumparrays(FILE *dev, size_t **arr, size_t narr, size_t *nelems);

/************************************
  *
  * Read gsl_matrix from file via csv2matrix.
  *
  * \param *in pointer to char array with filename
  * \returns gsl_matrix M
  *
  ***********************************/


gsl_matrix* gnupca_readmatrix(char *in);

/************************************
  *
  * Read gsl_matrix from file via csv2matrix 
  * with numeric (!) group information.
  * Strips the group information from input matrix and
  * stores the info in specific group-info arrays
  *
  * \param *in pointer to char array with filename
  * \param ***group pointer to 2-dim array with assigning each row in M to a group.
  * \param *ngroups pointer to size_t to return number of groups
  * \param **pointer to array with the size of each group
  * \param *pointer to gsl matrix with all entries lacking a group assignment (test set)
  * \param gcol column containing the group information
  * \returns M gsl_matrix with assigned groups
  *
  ***********************************/

gsl_matrix* gnupca_readmatrix_groups(char *in, size_t grow, size_t ***groups, size_t *ngroups, 
    size_t **groupsize); 

gsl_matrix* gnupca_get_groupcols(gsl_matrix *M, size_t **groups, size_t ngroups, size_t *groupsize);

void gnupca_matrix_subtract_vector(gsl_matrix *A, gsl_vector* x); 

