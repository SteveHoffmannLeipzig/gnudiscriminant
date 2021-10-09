/*
 *   gnudiscriminant - a linear discriminant analysis tool
 * 	 Copyright (C) Alexander Frotscher
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
 *  main.c
 *  
 *
 *  @author Alexander Frotscher
 *  @institute FLI
 *  
 */

#include <stdio.h>
#include <string.h>
#include "discriminant.h"

int main(int argc, char *argv[])
{
	double threshold=pow(10,-4);
	if (argc<4){
		printf("Usage: -r  <Path to training data set> <Path to validation data set> | -l <Path to training data set> <Path to validation data set>\nUse either the -r flag or the -l flag. The files for the datasets have to be tab-separated files and the columns have to represent the samples.\nThe training data set needs to contain the labels as numeric values in the first row. The number of features for both datasets has to be equal.\n");
	}
	int option;
	int rflag = 0;
	int lflag = 0;
	while ((option = getopt(argc, argv, "rl")) != -1)
	{
		switch (option)
		{
		case 'r':
			if (rflag)
			{
				printf("Use only one algorithm.\n");
				exit(EXIT_SUCCESS);
				break;
			}
			else
			{
				rflag++;
				lflag++;
			}
			ripleyLDA(argv[2], argv[3],threshold);
			break;
		case 'l':
			if (lflag)
			{
				printf("Use only one algorithm.\n");
				exit(EXIT_SUCCESS);
				break;
			}
			else
			{
				rflag++;
				lflag++;
			}
			YuLDA(argv[2], argv[3],threshold);
			break;
		default:
			printf("Usage: -r  <Path to training data set> <Path to validation data set> | -l <Path to training data set> <Path to validation data set>\nUse either the -r flag or the -l flag. The files for the datasets have to be tab-separated files and the columns have to represent the samples.\nThe training data set needs to contain the labels as numeric values in the first row. The number of features for both datasets has to be equal.\n");
		}
	}
}
