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

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <iterator>

#include <popt.h>

#define MAXSITES	6
using namespace std;

namespace AcqData {
	vector<string> fn;	/* file name (triplet) */
	vector<int> sites = { 1, 2, 3, 4, 5, 6 };
	vector<int> rsites = { 1, 2, 3, 4, 5 , 6 };
	int cps = 16;							/* columns per site */
	int bl = 0x400000;
	unsigned ramp_start[7];		// index from 1 */
};

int fail_stop = 0;
int accept_first_ramp_value;

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
		for (int col = 0; col < AcqData::cps; ++col){
			if (bs[col] != start+1){
				printf("ramp fail [%d][%d] %x %x\n",
					row, col, bs[col], start+1);
				fail += 1;
				start = bs[col];
				if (fail_stop != 0 && fail > fail_stop){
					printf("quitting on fail_stop\n");
					exit(1);
				}
			}else{
				++start;;
				/* printf("ramp pass [%d][%d] %x %x\n",
					row, col, bs[col], start+1); */
			}
		}
	}
	return fail;
}
int pass_count;
int fail_count[MAXSITES+1] = {};

int show_fail_summary()
{
	bool ok = true;
	printf("CUMULATIVE ");
        for (int site = 1; site <= MAXSITES; ++site){
                if (fail_count[site]){
                        printf("%d:%d ", site, fail_count[site]);
                        ok = 0;
                }
        }
        printf(" OK:%d", pass_count);
        if (!ok){
                printf(" FAIL\n");
        }else{
                printf(" PASS\n");
        }
	return ok;
}

void process_mapped_data(unsigned * ba, int len)
{
	int nrows = len/AcqData::sites.size()/AcqData::cps/sizeof(unsigned);
	bool ok = true;
#pragma omp parallel for
//	for (int site : AcqData::rsites){
	for (int ii = 0; ii < AcqData::rsites.size(); ++ii){
		int site = AcqData::rsites[ii];
		int fc = check_ramp_site(ba, (site-1)*AcqData::cps, nrows, AcqData::ramp_start[site]);
		if (fc){
			fail_count[site] += fc;
			ok = false;
		}
	}

	if (!ok){
		show_fail_summary();
	}else{
		++pass_count;
		printf(" PASS\n");
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
		if (ii ==0) cout << s << " ";
		fp[ii] = fopen(s.c_str(), "r");
//		printf("fopen %s\n", s.c_str());
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
//		printf("mmap:%x\n", bax);
		bax += AcqData::bl;
		ii += 1;
	}
	int ngrp = ii;
	process_mapped_data(ba, ngrp*AcqData::bl);

	for (ii = 0; ii < ngrp; ++ii){
		munmap(pdata[ii], AcqData::bl);
		fclose(fp[ii]);
	}
}

int calc_NGRP() {
	int nsites = AcqData::rsites.size();
	int ngrp = 1;
	switch(nsites){
	case 3:
	case 6:
		ngrp = 3;
	}

	printf("calc_NGRP() return ngrp=%d\n", ngrp);
	return ngrp;	
}

int process_files()
{
	char buf[80];
	int ngrp = calc_NGRP();

	while(fgets(buf, 80, stdin)){
		chomp(buf);
		AcqData::fn.push_back(buf);
		if (AcqData::fn.size() == ngrp){
			process_group();
			AcqData::fn.clear();
		}
	}

	printf("checkramp run complete: ");
	return !show_fail_summary();	// 0 is success for progs
}

void vector_replace(vector<int>& v, string str, string id)
{
	replace( str.begin(), str.end(), ',', ' ');
	istringstream buf(str);
	istream_iterator<std::string> beg(buf), end;

	vector<std::string> tokens(beg, end);
	v.clear();
	for (auto& s: tokens){
	     v.push_back(std::stoi(s));
	}
	cerr << id << " sites set:\"";
	for (int& s: v){
	       cerr << s;
	}
	cerr << "\" size:" << v.size() << "\n";
}
const char* sim_str;
const char* sites_str;

struct poptOption opt_table[] = {
	{ "sites", 'S', POPT_ARG_STRING, &::sites_str, 'S', "csv list of sites" },
	{ "sim", 's', POPT_ARG_STRING, &::sim_str, 's', "csv list of sites with simulation ramp" },
	{ "cps", 'c', POPT_ARG_INT, &AcqData::cps, 0, "columns (of u32) per site" },
       POPT_AUTOHELP
       POPT_TABLEEND
};


void ui(int argc, const char* argv[])
{
	
	if (getenv("FAIL_STOP")){
		fail_stop = atoi(getenv("FAIL_STOP"));
	}

	poptContext opt_context =
                        poptGetContext(argv[0], argc, argv, opt_table, 0);

	int rc;

        while ( (rc = poptGetNextOpt( opt_context )) >= 0 ){
                switch(rc){
		case 's':
			vector_replace(AcqData::rsites, sim_str, "rsites");
			break;
		case 'S':
			vector_replace(AcqData::sites, sites_str, "sites");
			break;
                }
        }
        for (int&s: AcqData::rsites){
        	vector<int>& sl = AcqData::sites;
        	if (find(sl.begin(), sl.end(), s) == sl.end()){
        		cerr << "ERROR: ramp site" << s << "not in sites\n";
        		exit(1);
        	}
        }
}

int main(int argc, const char* argv[])
{
	ui(argc, argv);
	return process_files();
}



