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

#define MAGIC	0xaa55f150
#define MAGIC_M 0xfffffff0

int VERBOSE = 0;
int NCHAN = 16;

int ESCHAN = 16;
bool isES(short* xx)
{
	unsigned *yy = (unsigned *)xx;
	return 
            (yy[0]&MAGIC_M) == MAGIC &&
            (yy[1]&MAGIC_M) == MAGIC &&
            (yy[2]&MAGIC_M) == MAGIC &&
            (yy[3]&MAGIC_M) == MAGIC;
}

static char dumpstr[128];

char* dumpES(short *xx)
{
	unsigned *yy = (unsigned*)xx;
	snprintf(dumpstr, 128, " %08x, %08x, %08x, %08x, %8d, %8d, %8d, %8d",
 	    yy[0], yy[1], yy[2], yy[3], yy[4], yy[5], yy[6], yy[7]);
	return dumpstr;
}

char* dump(short* xx)
{
	int ii;
	unsigned short* yy = (unsigned short*)xx;
	dumpstr[0] = '\0';
	char* cursor = dumpstr;
	for (ii = 0; ii < NCHAN; ++ii){
		cursor += sprintf(cursor, "%04x%s", yy[ii], ii+1==NCHAN? "": ",");
	}
	return dumpstr;
}

void _find_es()
{
	short *yy = new short[ESCHAN];	
	unsigned long sample = 0;

	while (fread(yy, sizeof(short), NCHAN, stdin) == NCHAN){
		if (isES(yy)){
                	if (ESCHAN > NCHAN){
				fread(yy+NCHAN, sizeof(short), NCHAN, stdin);
			}
                    	printf("%10ld,%s\n", sample, dumpES(yy));
		}else{
			if (VERBOSE > 1){
		    		printf("%10ld,%s\n", sample, dump(yy));
			}
		    	sample += 1;
		}
	}
        printf("processed %ld samples\n", sample);
}
void find_es()
{
	if (ESCHAN < NCHAN){
		ESCHAN = NCHAN;
	}
	_find_es();
}

int main(int argc, char* argv[])
{
	if (getenv("NCHAN") != 0){
		NCHAN = atoi(getenv("NCHAN"));
                printf("NCHAN set %d\n", NCHAN);
	}
	if (getenv("VERBOSE") != 0){
		VERBOSE = atoi(getenv("VERBOSE"));
	}

        find_es();
}
