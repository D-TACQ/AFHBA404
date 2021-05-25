/* ------------------------------------------------------------------------- */
/* file checkramp480.c							     */
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

#include <assert.h>

int main(int argc, char* argv[])
{
	unsigned* xx;
	unsigned xx0;
	unsigned xx1;
	unsigned long long ii = 0;
	unsigned long long ii1 = 0;
	unsigned long long bytes = 0;
	unsigned errors = 0;
	unsigned error_report = 0;
	int verbose = 0;
	int nchan = 8;
	int ichan = 0;
	int wrapcount = (1<<14);

	fprintf(stderr, "wrapcount:%d\n", wrapcount);

	if (argc > 1) nchan = atoi(argv[1]);
	if (argc > 2) ichan = atoi(argv[2]);
	assert(ichan < nchan);
	if (argc > 3) wrapcount = atoi(argv[3]);

	if (getenv("CHECKRAMP480_VERBOSE")) verbose = atoi(getenv("CHECKRAMP480_VERBOSE"));

	xx = malloc(nchan*sizeof(short));
	
	for(ii = 0; fread(xx, sizeof(short), nchan, stdin) == nchan; ++ii){
		if (ii == 0){
			xx0 = xx1 = xx[ichan];
		}else{
			if (xx[ichan] != xx1 + 1){
				if (errors == 0 || ii1 && ii - ii1 != wrapcount){
					if (++error_report < 10 || verbose > 1){
	printf("%012llx [%lld] +[%lld] 0x%08x 0x%08x ** ERROR **\n",
		bytes, ii, ii1? ii-ii1: 0, xx1, xx[ichan]);
					}
					++errors;
				}
				ii1 = ii;
			}else{
				error_report = 0;
			}
			xx1 = xx[ichan];
		}

		if (verbose && ii % 0x40000 == 0){
			printf("%012llx  %08x %08x  %d errors\n", ii, xx0, *xx, errors);
		}
		bytes += nchan*sizeof(short);
	}	

	printf("%012llx bytes %llu Mbytes %d errors\n",
		ii*sizeof(short)*nchan, ii*sizeof(short)*nchan/0x100000, errors);		
	return errors;
}


