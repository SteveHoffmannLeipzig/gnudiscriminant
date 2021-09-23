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
 *  gnupca_io.c
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
#include "gnupca_io.h"

/************************************
  *
  * guess file type from gz extensions
  * returns 1 if gzipped file
  *
  ***********************************/

char gnupca_isgz(char *fn) {
  uint32_t prefixlen = bl_fileprefixlen(fn);

    if(strncmp(&fn[prefixlen], ".gz", 3) == 0 || 
       strncmp(&fn[prefixlen], ".gzip", 5) == 0 ||
       strncmp(&fn[prefixlen], ".bgz", 4) == 0 || 
       strncmp(&fn[prefixlen], ".bgzip", 6) == 0 ) {
      printf("guessing gzip from extension of input file '%s'.\n",fn);
      return 1;
    }

    return 0;
}

/************************************
  *
  * open file and assign it co circ buffer
  * exit program on file open error
  *
  ***********************************/


char gnupca_fileopen(circbuffer_t *buf, char *fn) {
  FILE *fp;
  gzFile gzfp;

  if(gnupca_isgz(fn)) {
    gzfp = gzopen(fn, "r"); 
    
    if(gzfp == NULL) {
      printf("gzopen of file '%s' failed: %s. Aborting.\n", fn, strerror(errno));
      exit(EXIT_FAILURE);
    }
    //open buffers
    bl_circBufferInitGz(buf, 100000, gzfp, NULL);

  } else {
    fp = fopen(fn, "r"); 
  
    if(fp == NULL) {
      printf("fopen of file '%s' failed: %s. Aborting.\n", fn, strerror(errno));
      exit(EXIT_FAILURE);
    }
    //open buffers
    bl_circBufferInit(buf, 100000, fp, NULL);
  }

  return 1;
}
/************************************
  *
  * Close device assigned to circular buffer.
  *
  * \param buf pointer to a circbuffer_t.
  *
  ***********************************/

void gnupca_fileclose(circbuffer_t *buf) {

  if(buf->gzip) {
    gzclose(buf->gzdev);
  } else {
    fclose(buf->dev);
  }

  return;
}

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

double* csv2matrix (char *in, size_t *rows, size_t *cols) { 
  char *line; 
  uint32_t len=0, n=0, m=0, i; 
  circbuffer_t buf;

  stringset_t *set; 
  double *arr = NULL;
  uint32_t increment = 2;
  uint32_t alen = increment;
  uint32_t acur = 0;

  arr = ALLOCMEMORY(NULL, NULL, double, alen);
  gnupca_fileopen(&buf, in);

  while((line = bl_circBufferReadLine(&buf, &len))) { 
    //leverage tokenizer
    set = tokensToStringset(NULL, "\t", line, len);  
    //check dimensions, i.e. number of columns
    if(m != 0 && m != set->noofstrings) {
      fprintf(stderr, "Unequal col numbers (%d != %d). Malformed matrix?\n", m, set->noofstrings);
      EXIT_FAILURE;
    }
    //reading fields
    for(i=0; i < set->noofstrings; i++){
      //convert string to double
      double val = atof(set->strings[i].str);
      if(acur == alen) {
        //reallocate and check success
        arr = ALLOCMEMORY(NULL, arr, double, alen+increment);
        if(arr == NULL) {
          fprintf(stderr, "Alloc for %d elems failed. Machine out of RAM?\n", alen+increment);
          EXIT_FAILURE;
        }
        alen += increment; 
      }
      arr[acur] = val; 
      acur++; 
    }
    //store column number for cross check
    m = set->noofstrings;
    //increment row number
    n++;
    //cleanup
    destructStringset(NULL, set);
    FREEMEMORY(NULL, line);
  }
  
  gnupca_fileclose(&buf);
  bl_circBufferDestruct(&buf);  

  *rows = n;
  *cols = m;
  return arr;
}
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

size_t** csv2arrays (char *in, size_t *narr, size_t **nelems) { 
  char *line; 
  uint32_t len=0, i; 
  circbuffer_t buf;

  stringset_t *set; 
  size_t **arr = NULL;
  size_t *arr_elems = NULL;
  uint32_t increment = 2;
  size_t alen = increment;
  size_t acur = 0;

  arr = ALLOCMEMORY(NULL, NULL, size_t*, alen);
  arr_elems = ALLOCMEMORY(NULL, NULL, size_t, alen);
  gnupca_fileopen(&buf, in);

  while((line = bl_circBufferReadLine(&buf, &len))) { 
    //leverage tokenizer
    set = tokensToStringset(NULL, "\t", line, len);  
    //prepare array
    if(acur == alen) {
      //reallocate and check success
      arr = ALLOCMEMORY(NULL, arr, size_t*, alen+increment);
      arr_elems = ALLOCMEMORY(NULL, arr_elems, size_t, alen+increment);

      if(arr == NULL) {
        fprintf(stderr, "Alloc for %lu elems failed. Machine out of RAM?\n", alen+increment);
        EXIT_FAILURE;
      }
      alen += increment; 
    }
    arr[acur] = ALLOCMEMORY(NULL, NULL, size_t, set->noofstrings);
    arr_elems[acur] = set->noofstrings;
    //reading fields
    for(i=0; i < set->noofstrings; i++){
      //convert string to double
      arr[acur][i] = atol(set->strings[i].str);  
    }
    acur++; 
    //cleanup
    destructStringset(NULL, set);
    FREEMEMORY(NULL, line);
  }
  
  gnupca_fileclose(&buf);
  bl_circBufferDestruct(&buf);  

  *nelems = arr_elems;
  *narr = acur;
  return arr;
}

void dumparrays(FILE *dev, size_t **arr, size_t narr, size_t *nelems) {
  size_t i, j;

  fprintf(dev, "narr %lu\n", narr);
  for(i=0; i < narr; i++) {
    fprintf(dev, "nelems[%lu]=%lu\n", i, nelems[i]);
  }
  
  for(i=0; i < narr; i++) {
    for(j=0; j < nelems[i]; j++) {
      if(j > 0) fprintf(dev, "\t");
      fprintf(dev, "%lu", arr[i][j]);
    }
    fprintf(dev, "\n");
  }

  return;
}



/************************************
  *
  * Read gsl_matrix from file via csv2matrix.
  *
  * \param *in pointer to char array with filename
  * \returns gsl_matrix M
  *
  ***********************************/


gsl_matrix* gnupca_readmatrix(char *in) {
  double *arr;
  size_t m=0, n=0;
  gsl_matrix *M;
  gsl_block *block;

  arr = csv2matrix(in, &m, &n);
  
  block = ALLOCMEMORY(NULL, NULL, gsl_block, 1);
  block->data = arr;
  block->size = m*n;

  M = ALLOCMEMORY(NULL, NULL, gsl_matrix, 1);
  M->block = block;
  M->data = arr;
  M->size1 = m;
  M->size2 = n;
  M->tda = n;
  M->owner = 1; 

  return M;
}

/*
 *  Return a sub matrix with all columns
 *  indicated by the group info arrays. Useful for sampling.
 *  Attention: the groups are updated
 *  
 *  \param *M pointer to gsl_matrix with all columns
 *  \param **groups array of arrays with columnn numbers
 *  \param ngroups number of groups, i.e. first dim of **groups
 *  \param *groupsize size of groups, i.e. second dims(!) of **groups
 *  \return A gsl_matrix pointer with desired groups
 *
 */


gsl_matrix* gnupca_get_groupcols(gsl_matrix *M, size_t **groups, size_t ngroups, size_t *groupsize) {
  gsl_matrix* T;
  size_t i, j, k, cur=0, n=0;

  //determine col number of matrix
  for(k=0; k < ngroups; k++) {
    n+= groupsize[k];
  }

  T = gsl_matrix_calloc(M->size1, n);

  for(i=0; i < M->size1; i++) {

    for(cur=0, k=0; k < ngroups; k++) {
      for(j=0; j < groupsize[k]; j++) {
        gsl_matrix_set(T, i, cur, gsl_matrix_get(M, i, groups[k][j]));
        cur++;
      }
    }
  }

   return T;
}


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
    size_t **groupsize) {
  
  size_t i,j,k, ng=0, nTrain=0;
  size_t nooftest=0;
  size_t *Gnum=NULL, **Gmem=NULL ,*Gsze=NULL, *Tmem=NULL;
  
  gsl_matrix *A=NULL;  
  gsl_matrix* M = gnupca_readmatrix(in);

  for(j=0; j < M->size2; j++) {
    size_t class = (int) gsl_matrix_get(M, grow, j);
    //if there is class information, i.e. class > 0
    if(class != 0) {
      //check if class has been seen already
      for(k=0;  k < ng; k++) {
        if(Gnum[k] == class) break;
      }
      //class has not yet been seen
      if(k == ng) {
        Gnum = ALLOCMEMORY(NULL, Gnum, size_t, ng+1);
        Gmem = ALLOCMEMORY(NULL, Gmem, size_t*, ng+1); 
        Gsze = ALLOCMEMORY(NULL, Gsze, size_t, ng+1);
        Gnum[ng] = class;
        Gmem[ng] = NULL;
        Gsze[ng] = 0;
        ng += 1;
      }
      //update info
      Gmem[k] = ALLOCMEMORY(NULL, Gmem[k], size_t, Gsze[k]+1);
      Gmem[k][Gsze[k]] = j;
      Gsze[k] += 1;
      nTrain +=1;
    } else {
      Tmem = ALLOCMEMORY(NULL, Tmem, size_t, nooftest+1);
      Tmem[nooftest] = j; 
      nooftest += 1;
    }
  }

  if(0) {
  //dump group info
  for(k=0; k < ng; k++) {
    fprintf(stderr, "group %lu w/ %lu data points\n", Gnum[k], Gsze[k]);
    for(j=0; j < Gsze[k]; j++) {
      fprintf(stderr, "member of group %lu : %lu\n", Gnum[k], Gmem[k][j]);
    }
  }
  }

  //gnupca_writematrix(stdout, NULL, "Train", ',', Train, 0);


  //removing group information
    A = gsl_matrix_calloc(M->size1-1, M->size2);
    for(i=1; i < M->size1; i++) {
      for(j=0; j < M->size2; j++) {
        gsl_matrix_set(A, i-1, j, gsl_matrix_get(M, i, j));
      }
    }

  //gnupca_writematrix(stdout, NULL, "A", ',', A, 0);
 
 
  *groups=Gmem;
  *groupsize = Gsze;
  *ngroups = ng;

  gsl_matrix_free(M);
  FREEMEMORY(NULL, Gnum);
 
  return A;
}

/************************************
  *
  * This function is normaly in the gnupca_math.c file
  * It substracts the rows of the gsl_matrix A with the given gsl_vector x
  * 
  * Comment and inserted by A.F.
  *
  ***********************************/
void gnupca_matrix_subtract_vector(gsl_matrix *A, gsl_vector* x) {
  size_t i,j;
  assert(A->size1 == x->size);
  //columns first
  for(j=0; j < A->size2; j++) {
    for(i=0; i < A->size1; i++) {
      double tmp = gsl_matrix_get(A, i, j);
      tmp -= gsl_vector_get(x, i);
      gsl_matrix_set(A, i, j, tmp);
    }
  }
  return;
}

