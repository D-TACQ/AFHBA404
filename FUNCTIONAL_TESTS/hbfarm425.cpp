/* hbfarm425 - split off 2,3,4 x 16 channel blocks */

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>

int verbose;

#define MAXSITES	6
 
class HBFarm {
	const char* process;
	const char* outroot;
	FILE *fp_out[MAXSITES+1];	/* index from 1 */
	
	int ii;
	int sitelw;
	unsigned *lw;
	int next(int _ii){
		if (++_ii > MAXSITES) _ii = 1;
		if (fp_out[_ii] == 0){
			return next(_ii);/* recursion : woohoo! */
		}
		return _ii;
	}
public:
	int operator() (FILE *fin)
	{
		while(fread(lw, sizeof(unsigned), sitelw, fin) == sitelw){
			fwrite(lw, sizeof(unsigned), sitelw, fp_out[ii]);
			ii = next(ii);	
		}
	}
	int addSite(int site) {
		if (site < 1 || site > MAXSITES){
			fprintf(stderr, "site %d out of range\n", site);
			return -1;
		}else if (fp_out[site]){
			fprintf(stderr, "site %d already in use\n", site);
			return -1;
		}
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
		return 0;
	}
	HBFarm(int _sitelw = 8) : sitelw(_sitelw) {
		lw = new unsigned [_sitelw];
		ii = next(0);
		memset(fp_out, 0, sizeof(fp_out));
		if (getenv("HBFARM_PROCESS")){
			process = getenv("HBFARM_PROCESS");
			if (verbose) fprintf(stderr, "HBFarm process:%s lw:%d\n", process, sitelw);		
		}else{
			process = 0;
		}
		if (getenv("OUTROOT")){
			outroot = getenv("OUTROOT");
		}else{
			outroot = ".";
		}
	}
};


char* chomp(char* str) {
	char *nl;
	for (; nl = rindex(str, '\n'); ){
		*nl = '\0';
	}
	return str;
}

int main(int argc, char* argv[])
{
	if (argc < 2){
		fprintf(stderr, "hbfarm425 site1 site2 site3 site4\n");	
		return 1;
	}	

	int sinks = 0;
	int sitelw = 8;
	if (getenv("HBFARM_SITELW")) sitelw = atoi(getenv("HBFARM_SITELW"));
	if (getenv("HBFARM_VERBOSE")) verbose = atoi(getenv("HBFARM_VERBOSE"));
	HBFarm hbfarm(sitelw);

	for (int ii = 1; ii < argc; ++ii, ++sinks){
		if (hbfarm.addSite(atoi(argv[ii]))){
			return -1;	
		}
	}
	if (sinks == 0){
		fprintf(stderr, "must specify 1 or more sinks\n");
		return 1;
	}
	if (verbose) fprintf(stderr, "hbfarm %d sinks\n", sinks);

	if (getenv("NAMES_ON_STDIN")){
		char fname[80];
		while (fgets(fname, 80, stdin) && chomp(fname)){
			if (verbose) fprintf(stderr, "hbfarm %s\n", fname);
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


