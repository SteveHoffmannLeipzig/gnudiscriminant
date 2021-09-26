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
 *  discriminant.c
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
#include "discriminant.h"
#include <getopt.h>

/*********
 * 
 * This method classifys a given trainingSet according to the given trainingdata like the R programming language version of the LDA. 
 * Saves the classification in class.txt and the probabilitys in posterior.txt
 * 
 * \param *trainingdata a pointer to the file containing the trainingdata
 * \param *trainingset a pointer to the file containing the trainingset 
 * 
**********/
int ripleyLDA(  char *trainingdata,  char *trainingSet, double threshold)
{

	size_t grow=0;
	size_t classnumber;
	size_t* groupsize=NULL;
	size_t **groups = NULL;
	
	gsl_matrix* m =gnupca_readmatrix_groups(trainingdata,grow,&groups,&classnumber,&groupsize);
	double prior = (double)(*groupsize)/(double)m->size2;
	gsl_matrix_view *classes = makeClasses(m, &groupsize, &classnumber);
	gsl_matrix *groupmeans = calcGroupMeans(m, classes, &classnumber);

	gsl_vector_view *vectorviews = (gsl_vector_view *)malloc(classnumber * sizeof(gsl_matrix_column(groupmeans, 0)));
	if (vectorviews == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		EXIT_FAILURE;
	}
	for (size_t i = 0; i < classnumber; i++)
	{
		vectorviews[i] = gsl_matrix_column(groupmeans, i);
	}

	//subtract data with groupmeans
	for (size_t i = 0; i < classnumber; i++)
	{
		gnupca_matrix_subtract_vector((gsl_matrix *)&classes[i], (gsl_vector *)&vectorviews[i]);
	}
	free(vectorviews);
	vectorviews = NULL;
	free(classes);
	classes = NULL;

	//Check for high-dimensional data
	bool b = false;
	bool *ptr = &b;
	if (m->size1 > m->size2)
	{
		b = true;
	}
	size_t dataPoints = m->size2;

	//Scale the data according to the variance and the prior
	gsl_vector *diaM = varianceInv(m);
	gsl_matrix* mt= gsl_matrix_alloc(m->size2,m->size1);
	gsl_matrix_transpose_memcpy(mt,m);
	gsl_matrix_free(m);
	gsl_matrix *mR = scaleData(mt, diaM, &classnumber, ptr);

	//get eigenvectors of within-class covariance matrix
	gsl_vector *singularvalues = gsl_vector_alloc(mR->size2);
	gsl_matrix *pc = getPrincipalComponents(mR,singularvalues,ptr);
	size_t rank = singularvalues->size;
	gsl_vector *svalues = getRank(singularvalues, &rank, threshold);
	bool b2=false;
	bool *ptr2=&b2;
	gsl_matrix *eigenvec = truncatePC(pc, &rank, ptr2);

	//If pc is truncated delte the old version
	if (b2 == true)
	{
		gsl_matrix_free(pc);
	}

	//scale the eigenvectors
	gsl_matrix *scale = scaling(eigenvec, diaM, svalues);

	//get eigenvectors of between-class covariance matrix
	gsl_matrix *gm = gsl_matrix_alloc(scale->size2, groupmeans->size2);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans,sqrt((dataPoints * prior)/( 1.0 * (classnumber - 1))), scale, groupmeans, 0.0, gm);
	gsl_vector *singularvalues2 = gsl_vector_alloc(gm->size2);
	b=true;
	gsl_matrix *pc2 = getPrincipalComponents(gm,singularvalues2,ptr);
	rank = singularvalues2->size;
	gsl_vector *svalues2 = getRank(singularvalues2, &rank, threshold);
	gsl_matrix *eigenvec2 = truncatePC(pc2, &rank, ptr);

	//If pc is truncated delte the old version
	if (b == true)
	{
		gsl_matrix_free(pc2);
	}

	gsl_matrix *lda = calcRipleyLDA(eigenvec2,scale);

	//Project to new space
	gsl_matrix *t = projectTrainingSet(trainingSet, lda);
	gsl_matrix *pGroupMeans = projectGroupMeans(groupmeans, lda);
	gsl_matrix_free(lda);

	gsl_vector_view *pVectorviews = (gsl_vector_view *)malloc(classnumber * sizeof(gsl_matrix_column(pGroupMeans, 0)));
	if (pVectorviews == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		EXIT_FAILURE;
	}
	for (size_t i = 0; i < classnumber; i++)
	{
		pVectorviews[i] = gsl_matrix_column(pGroupMeans, i);
	}

	classification(t, pVectorviews, &classnumber);

	free(pVectorviews);
	pVectorviews = NULL;
	gsl_matrix_free(pGroupMeans);
	gsl_matrix_free(t);
	gsl_matrix_free(groupmeans);
	return EXIT_SUCCESS;

}

/*********
 * 
 * This method classifys a given trainingSet according to the given trainingdata like the LDA version proposed by Yu and Yang. 
 * Saves the classification in class.txt and the probabilitys in posterior.txt
 * 
 * \param *trainingdata a pointer to the file containing the trainingdata
 * \param *trainingset a pointer to the file containing the trainingset 
 * 
**********/
int YuLDA( char *trainingdata, char *trainingSet, double threshold)
{
	size_t grow=0;
	size_t classnumber;
	size_t* groupsize=NULL;
	size_t **groups = NULL;
	
	gsl_matrix* m =gnupca_readmatrix_groups(trainingdata,grow,&groups,&classnumber,&groupsize);
	gsl_matrix_view *classes = makeClasses(m, &groupsize, &classnumber);
	gsl_matrix *groupmeans = calcGroupMeans(m, classes, &classnumber);

	gsl_vector_view *vectorviews = (gsl_vector_view *)malloc(classnumber * sizeof(gsl_matrix_column(groupmeans, 0)));
	if (vectorviews == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		EXIT_FAILURE;
	}
	for (size_t i = 0; i < classnumber; i++)
	{
		vectorviews[i] = gsl_matrix_column(groupmeans, i);
	}

	//subtract data with groupmeans
	for (size_t i = 0; i < classnumber; i++)
	{
		gnupca_matrix_subtract_vector((gsl_matrix *)&classes[i], (gsl_vector *)&vectorviews[i]);
	}
	free(vectorviews);
	vectorviews = NULL;
	free(classes);
	classes = NULL;
	
	bool b = true;
	bool *ptr = &b;
	size_t samples=m->size2;

	//get eigenvalues and eigenvectors from the between-class covariance matrix
	gsl_matrix *gm = gsl_matrix_alloc(groupmeans->size1, groupmeans->size2);
	gsl_matrix_memcpy(gm, groupmeans);
	gsl_vector *singularvalues = gsl_vector_alloc(gm->size2);
	gsl_matrix *pc = getPrincipalComponents(gm,singularvalues,ptr);

	size_t rank = singularvalues->size;
	gsl_vector *svalues = getRank(singularvalues, &rank, threshold);
	gsl_matrix *eigenvec = truncatePC(pc, &rank, ptr);

	//If pc is truncated delte the old version
	if (b == true)
	{
		gsl_matrix_free(pc);
	}

	gsl_vector *eigenvalues = scaleSingular2(svalues, &classnumber);
	gsl_vector_free(svalues);
	gsl_matrix *gamma = calcGammaMatrix(eigenvec, eigenvalues);
	gsl_vector_free(eigenvalues);

	//get eigenvalues and eigenvectors from the within-class covariance matrix in the new space
	gsl_matrix *c = gsl_matrix_alloc(m->size2, gamma->size2);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, m, gamma, 0.0, c);
	gsl_matrix_free(m);
	gsl_vector *singularvalues2 = gsl_vector_alloc(c->size2);
	b=false;
	gsl_matrix* pc2 = getPrincipalComponents(c,singularvalues2,ptr);

	//transform the eigenvalues
	gsl_vector *eigenvalues2 = scaleSingular(singularvalues2, samples, &classnumber);
	gsl_vector_free(singularvalues2);

	//calculate the transformation matrix
	gsl_vector *upsilon = calcUpsilon(pc2, eigenvalues2);
	gsl_matrix *lda = calcYuLDA(pc2, gamma, upsilon);
	gsl_matrix_free(gamma);
	gsl_matrix_free(pc2);

	//project the data to the new space
	gsl_matrix *x_star = projectTrainingSet(trainingSet, lda);
	gsl_matrix *pGroupMeans = projectGroupMeans(groupmeans, lda);
	gsl_matrix_free(lda);

	gsl_vector_view *pVectorviews = (gsl_vector_view *)malloc(classnumber * sizeof(gsl_matrix_column(pGroupMeans, 0)));
	if (pVectorviews == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		EXIT_FAILURE;
	}
	for (size_t i = 0; i < classnumber; i++)
	{
		pVectorviews[i] = gsl_matrix_column(pGroupMeans, i);
	}

	classification(x_star, pVectorviews, &classnumber);

	free(pVectorviews);
	pVectorviews = NULL;
	gsl_matrix_free(pGroupMeans);
	gsl_matrix_free(x_star);
	gsl_matrix_free(groupmeans);
	return EXIT_SUCCESS;
}

/*********
 * 
 * This method loads a trainingset into a gsl_matrix. It is capable of assigning diffrent prior's to the classes. 
 * Should only be used for small input data. A better workaround can be found in gnupca_io.c
 * 
 * \param *trainingdata a pointer to the file containing the trainingdata
 * \param **classsize a pointer to a size_t array containing the number of samples per class
 * \param *classnumber a pointer to a size_t variable containing the number of classes
 * \param **prior a pointer to a double array containing the prior probabilitys for every class
 * 
 * \return a gsl_matrix containing the data
 * 
**********/
gsl_matrix *loadTrainingData(char *trainingdata, size_t **classsize, size_t *classnumber, double **prior)
{
	size_t columns = 0;
	size_t rows = 0;
	FILE *pData;

	//open file
	if ((pData = fopen(trainingdata, "r")) == NULL)
	{
		fprintf(stderr, "cannot open trainingsdata \n");
		exit(EXIT_FAILURE);
	}
	//get number of rows
	else
	{
		int ch = fgetc(pData);
		while (ch != '\r')
		{
			ch = fgetc(pData);
		}
		while (!feof(pData))
		{
			ch = fgetc(pData);
			if (ch == '\r')
			{
				rows++;
			}
		}
	}
	rewind(pData);

	//get columns and classsize
	size_t count = 0;
	size_t n = 1;
	int ch = fgetc(pData);
	while (ch != '\r')
	{
		if (ch != '\t')
		{
			columns++;
		}
		ch = fgetc(pData);
	}
	rewind(pData);

	//get classsize for each class
	*classsize = (size_t *)malloc(columns * sizeof(size_t));
	if (*classsize == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		exit(EXIT_FAILURE);
	}
	ch = fgetc(pData);
	ch = ch - 48;
	while (ch != '\r')
	{
		while (ch == n)
		{
			count++;
			ch = fgetc(pData);
			if (ch == '\t')
			{
				ch = fgetc(pData);
				ch = ch - 48;
			}
		}
		(*classsize)[n - 1] = count;
		count = 0;
		n++;
	}
	*classnumber = n - 1;
	*classsize = (size_t *)realloc(*classsize, *classnumber * sizeof(size_t));
	if (*classsize == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		exit(EXIT_FAILURE);
	}
	//calc priors for classes
	*prior = (double *)malloc(*classnumber * sizeof(double));
	if (*prior == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		exit(EXIT_FAILURE);
	}
	double col = (double)columns;
	for (size_t i = 0; i < *classnumber; i++)
	{
		double classS = (double)(*classsize)[i];
		(*prior)[i] = classS / col;
	}

	//get data
	gsl_matrix *m = gsl_matrix_alloc(rows, columns);
	rows = 0;
	columns = 0;
	n = 0;
	char lineIn[25];
	ch = fgetc(pData);
	while (!feof(pData))
	{
		if (ch == '\t')
		{
			lineIn[n] = '\0';
			gsl_matrix_set(m, rows, columns, atof(lineIn));
			columns++;
			n = 0;
		}
		else if (ch == '\r')
		{
			lineIn[n] = '\0';
			gsl_matrix_set(m, rows, columns, atof(lineIn));
			columns = 0;
			rows++;
			n = 0;
		}
		else
		{
			lineIn[n++] = ch;
		}
		ch = fgetc(pData);
	}
	fclose(pData);
	return m;
}

/*********
 * 
 * This method creates matrix views for each class in order to work with all samples of one class
 * 
 * \param *m a pointer to the gsl_matrix containing the data
 * \param **classsize a pointer to a size_t array containing the number of samples per class
 * \param *classnumber a pointer to a size_t variable containing the number of classes
 * 
 * \return a matrix view of the classes
 * 
**********/
gsl_matrix_view *makeClasses(gsl_matrix *m, size_t **classsize, size_t *classnumber)
{
	gsl_matrix_view *classes = (gsl_matrix_view *)malloc(*classnumber * sizeof(gsl_matrix_submatrix(m, 0, 0, m->size1, (*classsize)[0])));
	if (classes == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		exit(EXIT_FAILURE);
	}
	for (size_t i = 0; i < *classnumber; i++)
	{
		classes[i] = gsl_matrix_submatrix(m, 0, (*classsize)[i] * i, m->size1, (*classsize)[i]);
	}
	return classes;
}

/*********
 * 
 * This method calculates the mean vector for each class and stores them in a gsl_matrix
 * 
 * \param *m a pointer to the gsl_matrix containing the data
 * \param *classes a pointer to the gsl_matrix_view for each class
 * \param *classnumber a pointer to a size_t variable containing the number of classes
 * 
 * \return a gsl_matrix, where each column is a mean vector of one class
 * 
**********/
gsl_matrix *calcGroupMeans(gsl_matrix *m, gsl_matrix_view *classes, size_t *classnumber)
{
	gsl_matrix *groupmeans = gsl_matrix_alloc(m->size1, *classnumber);
	for (size_t k = 0; k < *classnumber; k++)
	{
		for (size_t i = 0; i < m->size1; i++)
		{
			double tmpmean = 0;
			for (size_t j = 0; j < *(&classes[k].matrix.size2); j++)
			{
				tmpmean += (gsl_matrix_get((gsl_matrix *)&classes[k], i, j) - tmpmean) / (j + 1);
			}
			gsl_matrix_set(groupmeans, i, k, tmpmean);
		}
	}
	return groupmeans;
}


/*********
 * 
 * This method calculates the principal components of the matrix m and stores them in a gsl_matrix
 * 
 * \param *m a pointer to the gsl_matrix containing the data
 * \param *singularvalues a pointer to the gsl_vector containing the singularvalues
 * \param *v a pointer to a gsl_matrix which will be filled with the right singular vectors
 * \param *b a pointer to a bool variable that realizes if the pc is in v or u
 * 
 * \return a gsl_matrix containing the pc
 * 
**********/
gsl_matrix* getPrincipalComponents(gsl_matrix* m, gsl_vector* singularvalues, bool* b) {
	gsl_matrix* x = gsl_matrix_alloc(m->size2, m->size2);
	gsl_vector* work = gsl_vector_alloc(m->size2);
	gsl_matrix *v = gsl_matrix_alloc(m->size2, m->size2);
	gsl_linalg_SV_decomp_mod(m, x, v, singularvalues, work);
	if (*b == true) {
		gsl_matrix_free(v);
		return m;
	}
	else {
		gsl_matrix_free(m);
		return v;
	}

}

/*********
 * 
 * This method decides the ammount of truncation the algorithms use
 * 
 * \param *singularvalues a pointer to the gsl_vector containing the singularvalues
 * \param *rank a pointer to the size_t variable containing the rank of the data
 * 
 * \return a gsl_vector of the singularvalues up to the desired rank
 * 
**********/
gsl_vector *getRank(gsl_vector *singularvalues, size_t *rank, double threshold)
{
	while (gsl_vector_get(singularvalues, *rank - 1) < threshold)
	{
		(*rank)--;
	}
	if (*rank == singularvalues->size)
	{
		return singularvalues;
	}
	else
	{
		gsl_vector *sv = gsl_vector_alloc(*rank);
		for (size_t i = 0; i < *rank; i++)
		{
			gsl_vector_set(sv, i, gsl_vector_get(singularvalues, i));
		}
		gsl_vector_free(singularvalues);
		return sv;
	}
}

/*********
 * 
 * This method truncates the singularvectors needed for the diagonalization according to the desired rank
 * 
 * \param *pc a pointer to a gsl_matrix, where each column is a singularvector
 * \param *rank a pointer to the size_t variable containing the rank of the data
 * \param *b a pointer to a bool variable determining if the rank was truncated
 * 
 * \return a gsl_matrix of the singularvectors up to the desired rank
 * 
**********/
gsl_matrix *truncatePC(gsl_matrix *pc, size_t *rank, bool *b)
{
	if (pc->size2 == *rank)
	{
		*b = false;
		return pc;
	}
	else
	{
		*b = true;
		gsl_matrix *pc2 = gsl_matrix_alloc(pc->size1, *rank);
		for (size_t i = 0; i < pc2->size1; i++)
		{
			for (size_t j = 0; j < pc2->size2; j++)
			{
				gsl_matrix_set(pc2, i, j, gsl_matrix_get(pc, i, j));
			}
		}
		return pc2;
	}
}

/*********
 * 
 * This method transforms the singularvalues to the eigenvalues of the within-class covariance matrix
 * 
 * \param *singularvalues a pointer to the gsl_vector containing the singularvalues
 * \param *samples a poinet to a size_t variable containing the number of samples
 * \param *classnumber a pointer to a size_t variable containing the number of classes
 * 
 * \return a gsl_vector with the eigenvalues from the within-class covariance matrix
 * 
**********/
gsl_vector *scaleSingular(gsl_vector *singularvalues, size_t samples, size_t *classnumber)
{
	gsl_vector *eigenv = gsl_vector_alloc(singularvalues->size);
	for (size_t i = 0; i < eigenv->size; i++)
	{
		gsl_vector_set(eigenv, i, pow(gsl_vector_get(singularvalues, i), 2) / (samples - *classnumber));
	}
	return eigenv;
}

/*********
 * 
 * This method transforms the singularvalues to the eigenvalues of the between-class covariance matrix
 * 
 * \param *singularvalues a pointer to the gsl_vector containing the singularvalues
 * \param *samples a poinet to a size_t variable containing the number of samples
 * \param *classnumber a pointer to a size_t variable containing the number of classes
 * 
 * \return a gsl_vector with the eigenvalues from the between-class covariance matrix
 * 
**********/
gsl_vector *scaleSingular2(gsl_vector *singularvalues, size_t *classnumber)
{
	gsl_vector *eigenv = gsl_vector_alloc(singularvalues->size);
	for (size_t i = 0; i < eigenv->size; i++)
	{
		gsl_vector_set(eigenv, i, pow(gsl_vector_get(singularvalues, i), 2) / ((*classnumber) - 1));
	}
	return eigenv;
}

/*********
 * 
 * This method loads the validation set into a matrix. Only use for small data
 * 
 * \param *devset a pointer to the file containing the validationset
 * 
 * \return a gsl_matrix of the validationset
 * 
**********/
gsl_matrix *getData(const char *devset)
{
	size_t columns = 0;
	size_t rows = 0;
	FILE *pData;

	//get number of columns and rows from data
	if ((pData = fopen(devset, "r")) == NULL)
	{
		fprintf(stderr, "cannot open file \n");
		exit(EXIT_FAILURE);
	}
	else
	{
		int ch = getc(pData);
		while (ch != '\n')
		{
			if (ch == '\t')
			{
				columns++;
			}
			ch = fgetc(pData);
		}
		columns++;
		while (!feof(pData))
		{
			if (ch == '\n')
			{
				rows++;
			}
			ch = fgetc(pData);
		}
		//get data
		rewind(pData);
		gsl_matrix *m = gsl_matrix_alloc(rows, columns);
		rows = 0;
		columns = 0;
		size_t n = 0;
		char lineIn[25];
		ch = fgetc(pData);
		while (!feof(pData))
		{
			if (ch == '\t')
			{
				lineIn[n] = '\0';
				gsl_matrix_set(m, rows, columns, atof(lineIn));
				columns++;
				n = 0;
			}
			else if (ch == '\n')
			{
				lineIn[n] = '\0';
				gsl_matrix_set(m, rows, columns, atof(lineIn));
				columns = 0;
				rows++;
				n = 0;
			}
			else
			{
				lineIn[n++] = ch;
			}
			ch = fgetc(pData);
		}
		fclose(pData);
		return m;
	}
}

/*********
 * 
 * This method projects the mean of the data to the new space found by the LDA
 * 
 * \param *groupmeans a pointer to the gsl_matrix containing the mean vectors
 * \param *lda a pointer to the matrix containing the linear discriminants
 * 
 * \return a gsl_matrix containing the projected mean vectors
 * 
**********/
gsl_matrix *projectGroupMeans(gsl_matrix *groupmeans, gsl_matrix *lda)
{
	gsl_matrix *tmp = gsl_matrix_alloc(groupmeans->size2, lda->size2);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, groupmeans, lda, 0.0, tmp);
	gsl_matrix *pGroupMeans = gsl_matrix_alloc(tmp->size2, tmp->size1);
	gsl_matrix_transpose_memcpy(pGroupMeans, tmp);
	gsl_matrix_free(tmp);
	return pGroupMeans;
}

/*********
 * 
 * This method projects the validationset to the new space found by the LDA
 * 
 * \param *trainingSet a pointer to the file containing the validationset
 * \param *lda a pointer to the matrix containing the linear discriminants
 * 
 * \return a gsl_matrix containing the projected data
 * 
**********/
gsl_matrix *projectTrainingSet( char *trainingSet, gsl_matrix *lda)
{
	gsl_matrix *m1 = gnupca_readmatrix(trainingSet);
	gsl_matrix *tmp = gsl_matrix_alloc(m1->size2, lda->size2);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, m1, lda, 0.0, tmp);
	gsl_matrix_free(m1);
	gsl_matrix *mp = gsl_matrix_alloc(tmp->size2, tmp->size1);
	gsl_matrix_transpose_memcpy(mp, tmp);
	gsl_matrix_free(tmp);
	return mp;
}

/*********
 * 
 * This method classifys the projected trainingset according to the euclidean distance with the
 * projected class means only if the prior probabilitys for each class is equal
 * 
 * \param *t a pointer to the gsl_matrix containing the projected data
 * \param *vectorviews a pointer to the gsl_vector_view containing the projected mean vectors
 * \param *classnumber a pointer to a size_t variable containing the number of classes
 * \param *prior a pointer to a double containing the prior probability
 * 
 * 
**********/
void classification(gsl_matrix *t, gsl_vector_view *vectorviews, size_t *classnumber)
{
	FILE* pData;
	FILE* pData2;

	//open file
	if ((pData = fopen("class.txt", "w+")) == NULL) {
		fprintf(stderr, "cannot open result file \n");
		exit(EXIT_FAILURE);
	}
	if ((pData2 = fopen("posterior.txt", "w+")) == NULL) {
		fprintf(stderr, "cannot open result file \n");
		exit(EXIT_FAILURE);
	}

	double *result = (double *)malloc(*classnumber * t->size2 * sizeof(double));
	if (result == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		exit(EXIT_FAILURE);
	}
	size_t index = 0;
	for (size_t j = 0; j < t->size2; j++)
	{
		for (size_t i = 0; i < *classnumber; i++)
		{
			gsl_vector_view v = gsl_matrix_column(t, j);
			gsl_vector *tmp = gsl_vector_alloc(t->size1);
			gsl_vector_memcpy(tmp, (gsl_vector *)&v);
			gsl_vector_sub(tmp, (gsl_vector *)&vectorviews[i]);
			result[index] = 0.5 * sumofsq(tmp);
			index++;
			if (index % *classnumber == 0)
			{
				index--;
				double scale = 0;
				for (int k = 0; k < *classnumber; k++)
				{
					result[index - k] = exp(-result[index - k]);
					scale += result[index - k];
				}
				scale = 1 / scale;
				double cl = 0;
				int ck = 0;
				for (int k = *classnumber - 1; k >= 0; k--)
				{
					result[index - k] = result[index - k] * scale;
					if (cl < result[index - k]) {
						cl = result[index - k];
						ck = (i-k)+1;
					}
					fprintf(pData2,"Probability of sample %zu to be in class %zu = %.17g\n", j, (i - k) + 1, result[index - k]);
				}
				fprintf(pData,"%d",ck);
				fputc('\t', pData);
				fputc('\n',pData2);
				index++;
			}
		}
	}
	printf("Success.\n");
}

double sumofsq(gsl_vector *v)
{
	double sum = 0;
	for (size_t i = 0; i < v->size; i++)
	{
		sum += pow(gsl_vector_get(v, i), 2);
	}
	return sum;
}

/*********
 * 
 * This method calculates the inverse of the main diagonal from the within-class covariance matrix
 * 
 * \param *m a pointer to the gsl_matrix with the data
 * 
 * \return a pointer to the gsl_vector containing the inverse of the diagonal from the within-class covariance matrix
 * 
**********/
gsl_vector *varianceInv(gsl_matrix *m)
{
	gsl_vector *diaM = gsl_vector_calloc(m->size1);
	for (size_t i = 0; i < m->size2; i++)
	{
		for (size_t j = 0; j < m->size1; j++)
		{
			gsl_vector_set(diaM, j, gsl_vector_get(diaM,j) + gsl_matrix_get(m, j, i) * gsl_matrix_get(m, j, i));
		}
	}
	double size = m->size2;
	gsl_vector_scale(diaM, 1 / (size - 1));
	for (size_t i = 0; i < diaM->size; i++)
	{
		gsl_vector_set(diaM, i, 1.0 / gsl_vector_get(diaM, i));
	}
	return diaM;
}

/*********
 * 
 * This method scales the data
 * 
 * \param *m a pointer to the gsl_matrix with the data
 * \param *diaM a pointer to the gsl_vector containing the inverse of the diagonal from the within-class covariance matrix
 * \param *classnumber a pointer to a size_t variable containing the number of classes
 * \param *b a pointer to a bool variable that realizes if the data is high-dimensional or not
 * 
 * \return a pointer to the gsl_matrix containing the scaled data
 * 
**********/
gsl_matrix *scaleData(gsl_matrix *m, gsl_vector *diaM, size_t *classnumber, bool *b)

{
	gsl_matrix_scale(m,sqrt(1.0 / (m->size1 - *classnumber)));

	// diagonal matrix multiplication from the right with a vector
	gsl_vector_view *columnviews = (gsl_vector_view *)malloc(m->size2 * sizeof(gsl_matrix_column(m, 0)));
	if (columnviews == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		EXIT_FAILURE;
	}
	for (size_t i = 0; i < m->size2; i++)
	{
		columnviews[i] = gsl_matrix_column(m, i);
		gsl_vector_scale((gsl_vector*)&columnviews[i], gsl_vector_get(diaM,i));
	}
	free(columnviews);
	columnviews=NULL;

	if (*b == true)
	{
		gsl_matrix *m2 = gsl_matrix_alloc(m->size2, m->size1);
		gsl_matrix_transpose_memcpy(m2, m);
		gsl_matrix_free(m);
		return m2;
	}
	else
	{
		return m;
	}
}

/*********
 * 
 * This method scales the principal components
 * 
 * \param *eigenvec a pointer to the matrix containing the eigenvectors of the within-class covariance matrix
 * \param *diaM a pointer to the gsl_vector containing the inverse of the diagonal from the within-class covariance matrix
 * \param *svalues a pointer to a gsl_vector containing the truncated singular values
 * 
 * \return a pointer to the gsl_matrix containing the scaled data
 * 
**********/
gsl_matrix *scaling(gsl_matrix *eigenvec, gsl_vector *diaM, gsl_vector *svalues)
{
	// diagonal matrix multiplication from the left with a vector
	gsl_vector_view *rowviews = (gsl_vector_view *)malloc(eigenvec->size1 * sizeof(gsl_matrix_row(eigenvec, 0)));
	if (rowviews == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		EXIT_FAILURE;
	}
	for (size_t i = 0; i < eigenvec->size1; i++)
	{
		rowviews[i] = gsl_matrix_row(eigenvec, i);
		gsl_vector_scale((gsl_vector*)&rowviews[i], gsl_vector_get(diaM,i));
	}
	free(rowviews);
	rowviews=NULL;
	gsl_vector_free(diaM);

	// diagonal matrix multiplication from the right with a vector
	gsl_vector_view *columnviews = (gsl_vector_view *)malloc(eigenvec->size2 * sizeof(gsl_matrix_column(eigenvec, 0)));
	if (columnviews == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		EXIT_FAILURE;
	}
	for (size_t i = 0; i < eigenvec->size2; i++)
	{
		columnviews[i] = gsl_matrix_column(eigenvec, i);
		gsl_vector_scale((gsl_vector*)&columnviews[i],1.0/gsl_vector_get(svalues,i));
	}
	free(columnviews);
	columnviews=NULL;

	return eigenvec;
}

/*********
 * 
 * This method calculates the transformation matrix for the r-version of the LDA
 * 
 * \param *pc a pointer to a gsl_matrix containing the pc from the between-class covariance matrix
 * \param *scale a pointer to the gsl_matrix containing the scaled principal components from the within-class covariance matrix
 * 
 * \return a pointer to a gsl_matrix containing the transformation matrix
 * 
**********/
gsl_matrix *calcRipleyLDA(gsl_matrix *pc, gsl_matrix *scale)
{
	gsl_matrix *lda = gsl_matrix_alloc(scale->size1, pc->size2);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, scale, pc, 0.0, lda);
	gsl_matrix_free(scale);
	gsl_matrix_free(pc);
	return lda;
}

/*********
 * 
 * This method calculates the "Gamma" matrix for the YU-LDA
 * 
 * \param *eigenvec a pointer to the matrix containing the eigenvectors of the between-class covariance matrix
 * \param *eigenvalues a pointer to a gsl_vector containing the eigenvalues of the between-class covariance matrix
 * 
 * \return a pointer to the gsl_matrix "Gamma"
 * 
**********/
gsl_matrix *calcGammaMatrix(gsl_matrix *eigenvec, gsl_vector *eigenvalues)
{
	// diagonal matrix multiplication from the right with a vector
	gsl_vector_view *columnviews = (gsl_vector_view *)malloc(eigenvec->size2 * sizeof(gsl_matrix_column(eigenvec, 0)));
	if (columnviews == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		EXIT_FAILURE;
	}
	for (size_t i = 0; i < eigenvec->size2; i++)
	{
		columnviews[i] = gsl_matrix_column(eigenvec, i);
		gsl_vector_scale((gsl_vector*)&columnviews[i],1.0/(sqrt(gsl_vector_get(eigenvalues,i))));
	}
	free(columnviews);
	columnviews=NULL;

	return eigenvec;
}

/*********
 * 
 * This method calculates the Upsilon matrix for the YU-LDA stored in a vector
 * 
 * \param *v a pointer to the matrix containing the eigenvectors of the transformed within-class covariance matrix
 * \param *eigenvalues a pointer to a gsl_vector containing the eigenvalues of the transformed within-class covariance matrix
 * 
 * \return a pointer to the gsl_vector Upsilon
 * 
**********/
gsl_vector *calcUpsilon(gsl_matrix *v, gsl_vector *eigenvalues)
{
	gsl_matrix* tmp=gsl_matrix_alloc(v->size2,v->size1);
	gsl_matrix_transpose_memcpy(tmp,v);

	// diagonal matrix multiplication from the left with a vector
	gsl_vector_view *rowviews = (gsl_vector_view *)malloc(tmp->size1 * sizeof(gsl_matrix_row(tmp, 0)));
	if (rowviews == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		EXIT_FAILURE;
	}
	for (size_t i = 0; i < tmp->size1; i++)
	{
		rowviews[i] = gsl_matrix_row(tmp, i);
		gsl_vector_scale((gsl_vector*)&rowviews[i], gsl_vector_get(eigenvalues,i));
	}
	free(rowviews);
	rowviews=NULL;
	gsl_vector_free(eigenvalues);

	
	gsl_matrix *tmp2 = gsl_matrix_alloc(v->size1, tmp->size2);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, v, tmp, 0.0, tmp2);
	gsl_matrix_free(tmp);
	gsl_matrix *tmp3 = gsl_matrix_alloc(tmp2->size1, v->size2);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, tmp2, v, 0.0, tmp3);
	gsl_matrix_free(tmp2);
	gsl_matrix *dw = gsl_matrix_alloc(v->size2, tmp3->size2);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, v, tmp3, 0.0, dw);
	gsl_matrix_free(tmp3);
	gsl_vector *upsilon = gsl_vector_alloc(dw->size1);
	for (size_t i = 0; i < upsilon->size; i++)
	{
		gsl_vector_set(upsilon, i, 1.0 / sqrt(gsl_matrix_get(dw, i, i)));
	}
	gsl_matrix_free(dw);
	return upsilon;
}

/*********
 * 
 * This method calculates the transformation matrix for the YU-version of the LDA
 * 
 * \param *v a pointer to the matrix containing the eigenvectors of the transformed within-class covariance matrix
 * \param *a a pointer to the gsl_matrix "Gamma"
 * \param *upsilon a pointer to the gsl_vector Upsilon
 * 
 * \return a pointer to a gsl_matrix containing the transformation matrix
 * 
**********/
gsl_matrix *calcYuLDA(gsl_matrix *v, gsl_matrix *gamma, gsl_vector *upsilon)
{
	gsl_matrix *tmp = gsl_matrix_alloc(v->size2, gamma->size1);
	gsl_blas_dgemm(CblasTrans, CblasTrans, 1.0, v, gamma, 0.0, tmp);

	// diagonal matrix multiplication from the left with a vector
	gsl_vector_view *rowviews = (gsl_vector_view *)malloc(tmp->size1 * sizeof(gsl_matrix_row(tmp, 0)));
	if (rowviews == NULL)
	{
		fprintf(stderr, "memory could not be given free\n");
		EXIT_FAILURE;
	}
	for (size_t i = 0; i < tmp->size1; i++)
	{
		rowviews[i] = gsl_matrix_row(tmp, i);
		gsl_vector_scale((gsl_vector*)&rowviews[i], gsl_vector_get(upsilon,i));
	}
	free(rowviews);
	rowviews=NULL;
	gsl_vector_free(upsilon);

	gsl_matrix *lda = gsl_matrix_alloc(tmp->size2, tmp->size1);
	gsl_matrix_transpose_memcpy(lda, tmp);
	gsl_matrix_free(tmp);
	return lda;
}