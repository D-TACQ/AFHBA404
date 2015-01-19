/*
 * check-tlatch.c

 *
 *  Created on: 10 Jan 2015
 *      Author: pgm
 */


#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
	unsigned tl0;
	unsigned tl1;
	unsigned cursor = 1;
	unsigned prev_error = 0;
	int errors = 0;

	if (fread(&tl0, sizeof(unsigned), 1, stdin) != 1){
		perror("unable to fread stdin");
		exit(1);
	}

	printf("%10s/%-10s %10s %10s %10s %s\n",
		"errors", "total", "since", "tl0", "tl1", "missed");
	for(cursor = 1; fread(&tl1, sizeof(unsigned), 1, stdin) == 1;
			++cursor, tl0 = tl1){
		if (tl1 != tl0+1){
			++errors;
			printf("%10d/%-10d %10d 0x%08x 0x%08x %d\n",
				errors, cursor, cursor-prev_error, tl0, tl1, tl1-tl0-1);
			prev_error = cursor;
		}
	}
}


