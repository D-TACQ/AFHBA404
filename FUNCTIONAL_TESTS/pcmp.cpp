/* ------------------------------------------------------------------------- */
/* pcmp.cpp : compare program prints variations with word size awareness
 * Project: AFHBA404
 * Created: 4 April 2024
 * ------------------------------------------------------------------------- *
 *   Copyright (C) 2024 Peter Milne, D-TACQ Solutions Ltd         *
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
#include <string.h>
#include <stdlib.h>

#include <popt.h>

#include <algorithm>    // std::min

#define EPB_HISTO_LEN	8U
#define EPB_HISTO_MSK   (EPB_HISTO_LEN-1)
class WordCompare {
public:
	virtual int operator() (char* buffers[2], unsigned long long& cursor, size_t lenb) = 0;

	static WordCompare& factory(unsigned wordsize);
	static bool heal;
	static unsigned fixups;
	static unsigned left_fix;
	static unsigned errors_per_buffer[EPB_HISTO_LEN];
};

bool WordCompare::heal;
unsigned WordCompare::fixups;
unsigned WordCompare::left_fix;
unsigned WordCompare::errors_per_buffer[];

template<class C> 
class ConcreteWordCompare: public WordCompare {
	char format[128];
	unsigned nrow(){
		switch(sizeof(C)){
			default:
			case 8: return 1;
			case 4: return 2;
			case 2: return 4;
			case 1: return 8;
		}
	}
	
	void print(unsigned long long cursor, C* left, C* right) {
		printf("%016llx:", cursor);
		printf(format, *left);
		printf("|");
		printf(format, *right);
		printf("\n");
	}

	void attempt_heal(C* buffers[2], unsigned ii)
	{
		C left = buffers[0][ii];
		C right = buffers[1][ii];

		if ((left&~0xff) == (right&~0xff)){
			if ((left&0xff) != 0 && (right&0xff) == 0){
				printf("healing ..");
				printf(format, left);
				printf(" => ");
				printf(format, right);
				printf("\n");
				buffers[1][ii] = left;
				++fixups;
			}else if ((left&0xff) == 0 && (right&0xff) != 0){
				++left_fix;
			}
		}
	}
	int _cmp(C* buffers[2], unsigned long long cursor, size_t lenb){
		unsigned lenC = lenb/sizeof(C);
		unsigned step;
		unsigned errors = 0;
		unsigned old_fixups = fixups;
		unsigned old_left_fix = left_fix;
		for (unsigned ii = 0; ii < lenC; ii += step){
			if (buffers[0][ii] != buffers[1][ii]){
				printf("%08x:", ii);
				print(cursor+ii, buffers[0]+ii, buffers[1]+ii);
				if (heal){
					attempt_heal(buffers, ii);
				}
				step = nrow();
				++errors;
			}else{
				step = 1;
			}
		}
		errors_per_buffer[std::min(errors, EPB_HISTO_MSK)]++;
		if (errors == 2){
			fprintf(stderr, "SPECIAL CASE errors 2. L=%u R=%u\n",
					fixups-old_fixups, left_fix-old_left_fix);
		}
		return errors;
	}
public:
	ConcreteWordCompare() {
		snprintf(format, 128, "%%0%lullx", sizeof(C)*2);
		puts(format);
	}
	virtual int operator() (char* buffers[2], unsigned long long& cursor, size_t lenb){
		return _cmp((C**) buffers, cursor, lenb);
	}
};
WordCompare& WordCompare::factory(unsigned ws) {
	switch(ws){
	case 8:
		return * new ConcreteWordCompare<unsigned long long>();
	default:
		return * new ConcreteWordCompare<unsigned char>();
	}
}
namespace G {
	unsigned wordsize = 8;
	unsigned buflen = 0x400000;
	int heal = 0;
	const char* fn[2];
	FILE *fp[2];
	int max_err = 0;
	FILE *fp_heal;
}


struct poptOption opt_table[] = {
        { "ws", 'w', POPT_ARG_INT, &G::wordsize, 'S', "word size 1,2,4,8 bytes" },
	{ "heal", 'h', POPT_ARG_INT, &G::heal, 'H', "attempt to heal errors" },
	{ "maxerr", 'm', POPT_ARG_INT, &G::max_err, 'M', "maximum errors to process" },
       POPT_AUTOHELP
       POPT_TABLEEND
};

void ui(int argc, const char* argv[])
{
	poptContext opt_context =
                        poptGetContext(argv[0], argc, argv, opt_table, 0);
        int rc;

        while ( (rc = poptGetNextOpt( opt_context )) >= 0 ){
		;
	}
	for (int ii = 0; ii <= 1; ++ii){
		G::fn[ii] = poptGetArg(opt_context);
		if (G::fn[ii] == 0){
			fprintf(stderr, "ERROR: %s file name not present\n", ii? "second": "first");
			exit(-1);
		}else if ((G::fp[ii] = fopen(G::fn[ii], "r")) == 0){
			fprintf(stderr, "ERROR file open \"%s\" failed\n", G::fn[ii]);
			exit(-1);
		}
	}

	if (G::heal){
		char fn_heal[128];
		snprintf(fn_heal,  128, "%s.heal", G::fn[1]);
		G::fp_heal = fopen(fn_heal, "w");
		if (G::fp_heal == 0){
			perror(fn_heal);
			exit(1);
		}
		WordCompare::heal = true;
	}
}

enum C_STATE { C_FINISH_OK, C_FINISH_ERR, C_BUSY };

int good_buffers;
int bad_buffers;

C_STATE compare(char* buffers[2], unsigned long long& cursor, WordCompare& wc)
{
	size_t nelems[2];
	size_t nel;

	if (feof(G::fp[0]) || feof(G::fp[1])){
		fprintf(stderr, "reached the end of the road: %llx %d/%d\n", 
				cursor, bad_buffers, good_buffers+bad_buffers);
		return C_FINISH_OK;
	}
	for (int ii = 0; ii <= 1; ++ii){
		nelems[ii] = fread(buffers[ii], 1, G::buflen, G::fp[ii]);
	}
	if (nelems[0] == nelems[1] && nelems[0] == 0){
		return C_FINISH_OK;
	}
	if (nelems[0] != nelems[1]){
		int iearly = nelems[0] < nelems[1]? 0: 1;
		fprintf(stderr, "early termination %s at %llu\n", G::fn[iearly], cursor+nelems[iearly]);
		nel = nelems[iearly];
	}else{
		nel = nelems[0];
	}
	if (memcmp(buffers[0], buffers[1], nel) == 0){
		C_STATE rc = nel == G::buflen? C_BUSY: C_FINISH_OK;
		++good_buffers;
	}else{
		++bad_buffers;
		wc(buffers, cursor, nel);
		fprintf(stdout, "ERROR: at %llx %d/%d\n", cursor+nel, bad_buffers, good_buffers+bad_buffers);
		if (G::max_err && bad_buffers >= G::max_err){
			return C_FINISH_ERR;
		}
	}
	if (G::heal){
		fwrite(buffers[1], 1, G::buflen, G::fp_heal);
	}
	cursor += nel;

	return C_BUSY;
}

int main(int argc, const char* argv[])
{
	ui(argc, argv);	
	char* buffers[2];
	unsigned long long cursor = 0;
	for (int ii = 0; ii <= 1; ++ii){
		buffers[ii] = new char[G::buflen];
	}
	WordCompare& wc = WordCompare::factory(G::wordsize);

	while(1){
		switch(compare(buffers, cursor, wc)){
			case C_FINISH_OK:
				if (bad_buffers){
					fprintf(stderr, "ERROR: at %llx %d/%d", 
							cursor, bad_buffers, good_buffers+bad_buffers);
					if (G::heal){
						fprintf(stderr, " HEAL count:%u,%u %s", 
								WordCompare::fixups, WordCompare::left_fix,
								WordCompare::fixups==bad_buffers? "ALL GOOD": 
								(WordCompare::fixups+WordCompare::left_fix) == bad_buffers? "All errors fixable": "");
					}
					fprintf(stderr, "\n");
					fprintf(stderr, "ERRORS per buffer:");
					for (int ii = 0; ii < EPB_HISTO_LEN; ++ii){
					       fprintf(stderr, "%4u,", WordCompare::errors_per_buffer[ii]);
					}
					fprintf(stderr, "\n");
				}
				return 0;
			case C_FINISH_ERR:
				return 1;
			default:
				;
		}
	}
}


