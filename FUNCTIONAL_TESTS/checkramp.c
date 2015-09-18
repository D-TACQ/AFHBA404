/* ------------------------------------------------------------------------- */
/* file checkramp.c							     */
/* ------------------------------------------------------------------------- */
/*   Copyright (C) 2011 Peter Milne, D-TACQ Solutions Ltd
 *                      <Peter dot Milne at D hyphen TACQ dot com>

    http://www.d-tacq.com

    This program is free software; you can redistribute it and/or modify
    it under the terms of Version 2 of the GNU General Public License
    as published by the Free Software Foundation;

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                */
/* ------------------------------------------------------------------------- */

/** @file checkramp.c : validates ACQ196 simulate data ramp
 * Refs: 
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>


int main(int argc, char* argv[])
{
	unsigned xx;
	unsigned xx0;
	unsigned xx1;
	unsigned long long ii = 0;
	unsigned errors = 0;
	unsigned error_report = 0;
	int verbose = 0;
	int lwstride = 1;

	if (getenv("LWSTRIDE") != 0){
		lwstride = atoi(getenv("LWSTRIDE"));
	}
	if (argc > 1 && strcmp(argv[1], "-v") == 0){
		verbose = 1;
	}
	for(ii = 0; fread(&xx, sizeof(xx), 1, stdin); ii += lwstride){
		if (ii == 0){
			xx0 = xx1 = xx;
		}else{
			if (xx != xx1 + 1){
				if (++error_report < 10){
					printf("%012llx 0x%08x 0x%08x ** ERROR **\n",
					       ii, xx1, xx);
				}
				++errors;
			}else{
				error_report = 0;
			}
			xx1 = xx;
		}

		if (verbose && ii % 0x40000 == 0){
			printf("%012llx  %08x %08x  %d errors\n",
			       ii, xx0, xx, errors);
		}
		if (lwstride>1){
			fseek(stdin, (lwstride-1)*sizeof(xx), SEEK_CUR);
		}
	}	

	printf("%012llx bytes %llu Mbytes %d errors\n",
		ii*4, ii*4/0x100000, errors);		
}
