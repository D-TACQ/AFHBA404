/* hbfarm425 - split off 2,3,4 x 16 channel blocks */

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

FILE *fp_out[5];	/* index from 1 */

const char* process;
const char* outroot = ".";
int names_on_stdin;

int next(int ii){
	if (++ii > 4) ii = 1;
	if (fp_out[ii] == 0){
		return next(ii);		/* recursion : woohoo! */
	}
	return ii;
}

int hbfarm(FILE *fin)
{
	unsigned lw[8];
	int ii = next(0);
	while(fread(lw, sizeof(unsigned), 8, fin) == 8){
		fwrite(lw, sizeof(unsigned), 8, fp_out[ii]);
		ii = next(ii);	
	}
}

char* chomp(char* str) {
	char *nl;
	for (; nl = rindex(str, '\n'); ){
		*nl = '\0';
	}
	return str;
}
int main(int argc, char* argv[])
{
	int sinks = 0;
	int ii;
	if (argc < 2){
		fprintf(stderr, "hbfarm425 site1 site2 site3 site4\n");	
		return 1;
	}	
	if (getenv("HBFARM_PROCESS")){
		process = getenv("HBFARM_PROCESS");		
	}
	if (getenv("OUTROOT")){
		outroot = getenv("OUTROOT");
	}
	if (getenv("NAMES_ON_STDIN")){
		names_on_stdin = 1;
	}

	for (ii = 1; ii < argc; ++ii){
		int site = atoi(argv[ii]);
		if (site < 1 || site > 4){
			fprintf(stderr, "site %d out of range\n", site);
			return -1;
		}else if (fp_out[site]){
			fprintf(stderr, "site %d already in use\n", site);
		}else{
			if (process){
				fp_out[site] = popen(process, "w");
				if (fp_out[site] == 0){
					perror(process);
					return 1;
				}
			}else{
				char fname[128];
				sprintf(fname, "%s/hbfarm425.%d", outroot, site);
				fp_out[site] = fopen(fname, "w");
				if (fp_out[site] == 0){
					perror(fname);
					return 1;
				}
			}
			sinks++;
		}
	}
	if (sinks == 0){
		fprintf(stderr, "must specify 1 or more sinks\n");
		return 1;
	}
	if (names_on_stdin){
		char fname[80];
		while (fgets(fname, 80, stdin) && chomp(fname)){
			FILE *fp = fopen(fname, "r");
			if (fp == 0){
				perror(fname);
				return -1;
			}
			hbfarm(fp);
			fclose(fp);
		}
	}else{
		return hbfarm(stdin);
	}
}


