/* ------------------------------------------------------------------------- */
/* multisitecheckramp.cpp  Validate multiple file, multiple site data for ramp
 * Project: AFHBA404
 * Created: 14 Jul 2017  			/ User: pgm
 * ------------------------------------------------------------------------- *
 *   Copyright (C) 2017 Peter Milne, D-TACQ Solutions Ltd         *
 *                      <peter dot milne at D hyphen TACQ dot com>           *
 *                                                                           *
 *  This program is free software; you can redistribute it and/or modify     *
 *  it under the terms of Version 2 of the GNU General Public License        *
 *  as published by the Free Software Foundation;                            *
 *                                                                           *
 *  This program is distributed in the hope that it will be useful,          *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 *  GNU General Public License for more details.                             *
 *                                                                           *
 *  You should have received a copy of the GNU General Public License        *
 *  along with this program; if not, write to the Free Software              *
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                *
 *
 * TODO 
 * TODO
/* ------------------------------------------------------------------------- */

#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <assert.h>
#include "../local.h"

#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace AcqData {
	vector<string> fn;	/* file name (triplet) */
	vector<int> sites = { 1, 2, 3, 4, 5, 6 };
	vector<int> rsites = { 3, 4, 5 , 6 };
	int cps = 16;
	int bl = 0x400000;
	unsigned ramp_start[7];		// index from 1 */
};

#define NGRP	3

int check_ramp_site(unsigned *b0, unsigned offset, int nrows, unsigned& start)
{
	int fail = 0;
	unsigned *br = b0;					/* row ptr */
	unsigned stride = AcqData::sites.size()*AcqData::cps;

#if 0
	FILE *pp = popen("hexdump -e '4/4 \"%08x,\" \"\\n\"'", "w");
	if (!pp){
		perror("hexdump failed");
		exit(1);
	}
	fwrite(b0, sizeof(unsigned), AcqData::cps, pp);
	pclose(pp);
#endif
	for (int row = 0; row < nrows; ++row, br += stride){
		unsigned *bs = br + offset;		// slice ptr
		for (int col = 0; col < AcqData::cps; ++col, ++start){
			if (bs[col] != start+1){
				printf("ramp fail [%d][%d] %x %x\n",
					row, col, bs[col], start+1);
				fail += 1;
				start = bs[col];
			}else{
				;
				/* printf("ramp pass [%d][%d] %x %x\n",
					row, col, bs[col], start+1); */
			}
		}
	}
	return fail;
}
void process_mapped_data(unsigned * ba, int len)
{
	int nrows = len/AcqData::sites.size()/AcqData::cps/sizeof(unsigned);
	for (int site : AcqData::rsites){
		check_ramp_site(ba, (site-1)*AcqData::cps, nrows, AcqData::ramp_start[site]);
	}
}
void process_group()
{
	unsigned *ba = (unsigned*)0x40000000;
	char *bax = (char*)ba;
	FILE *fp[NGRP];
	void *pdata[NGRP];
	int ii = 0;

	for (string& s : AcqData::fn){
		cout << s << " ";
		fp[ii] = fopen(s.c_str(), "r");
		if (fp[ii] == 0){
			perror("failed to open file");
			exit(1);
		}
		pdata[ii] = mmap(bax, AcqData::bl,
				PROT_READ, MAP_SHARED|MAP_FIXED,
				fileno(fp[ii]), 0);
		if (pdata[ii] != bax){
			perror("mmap() failed to get the hint");
			exit(1);
		}
		bax += AcqData::bl;
		ii += 1;
	}
	process_mapped_data(ba, NGRP*AcqData::bl);
	cout << "\n";
	for (ii = 0; ii < NGRP; ++ii){
		munmap(pdata[ii], AcqData::bl);
		fclose(fp[ii]);
	}
}
void process_files()
{
	char buf[80];

	while(fgets(buf, 80, stdin)){
		chomp(buf);
		AcqData::fn.push_back(buf);
		if (AcqData::fn.size() == NGRP){
			process_group();
			AcqData::fn.clear();
		}
	}
}

void ui(int argc, char* argv[])
{

}

int main(int argc, char* argv[])
{
	ui(argc, argv);
	process_files();
}



