/*
 *   gnudiscriminant - a linear discriminant analysis tool
 *	 Copyright (C) Alexander Frotscher
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
 *  discriminant.h
 *  
 *
 *  @author Alexander Frotscher
 *  @institute FLI
 *  
 */

#include <stdio.h>
#include <string.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <assert.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics_double.h>
#include "gnupca_io.h"
#include <getopt.h>

typedef enum
{
	false,
	true
} bool;

int ripleyLDA( char *, char *,double);
int YuLDA( char *,  char *,double);

gsl_matrix *loadTrainingData(char *, size_t **, size_t *, double **);
gsl_matrix_view *makeClasses(gsl_matrix *, size_t ***, size_t **, size_t *);
gsl_matrix *calcGroupMeans(gsl_matrix *, gsl_matrix_view *, size_t *);
gsl_matrix* getPrincipalComponents(gsl_matrix*, gsl_vector*, bool*);
gsl_vector *getRank(gsl_vector *, size_t *,double);
gsl_matrix *truncatePC(gsl_matrix *, size_t *, bool *);
gsl_vector *scaleSingular(gsl_vector *, size_t, size_t *);
gsl_vector *scaleSingular2(gsl_vector *, size_t *);
gsl_matrix *getData(const char *);
gsl_matrix *projectGroupMeans(gsl_matrix *, gsl_matrix *);
gsl_matrix *projectTrainingSet( char *, gsl_matrix *);
void classification(gsl_matrix *, gsl_vector_view *, size_t *);
double sumofsq(gsl_vector *);

//methods for r-version
gsl_vector *varianceInv(gsl_matrix *);
gsl_matrix *scaleData(gsl_matrix *, gsl_vector *, size_t *, bool *);
gsl_matrix *scaling(gsl_matrix *, gsl_vector *, gsl_vector *);
gsl_matrix *calcRipleyLDA(gsl_matrix *, gsl_matrix *);

//methods for yu-version
gsl_matrix *calcGammaMatrix(gsl_matrix *, gsl_vector *);
gsl_vector *calcUpsilon(gsl_matrix *, gsl_vector *);
gsl_matrix *calcYuLDA(gsl_matrix *, gsl_matrix *, gsl_vector *);
