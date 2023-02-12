#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>		/* @@todo use site standard popt() */
#include <arpa/inet.h>

#include <errno.h>

/*
 * This file takes input from nc as below:
 * nc acq2106_112 4210 | pv | ./isramp
 */

namespace G {
	int maxcols = 104; // Number of columns of data
	int countcol = 96; // Column where the ramp is
	int step = 1; // Default step. For sample counter in spad step = 1
	int bigendian = 0;
	int maxerrs = 0;
	int ignore_first_entry = 0;
	const char* fname = "-";
	bool stdin_is_list_of_fnames;
	int verbose = 0;
	char *logname = NULL;
};

void get_args(int argc, char* const argv[]){
    int opt;
    while((opt = getopt(argc, argv, "b:m:c:s:i:E:N:v:L:")) != -1) {
	switch(opt) {
	case 'm':
		G::maxcols = atoi(optarg);
		break;
	case 'c':
		G::countcol = atoi(optarg);
		break;
	case 'b':
		G::bigendian = atoi(optarg);
		if (G::bigendian) printf("Hello Moto\n");
	case 's':
		G::step = atoi(optarg);
		printf("%i\n", atoi(optarg));
		break;
	case 'i':
		G::ignore_first_entry = atoi(optarg);
		break;
	case 'E':
		G::maxerrs = atoi(optarg);
		break;
	case 'N':
	    	G::stdin_is_list_of_fnames = atoi(optarg);
		break;
	case 'v':
		G::verbose = atoi(optarg);
		break;
	case 'L':
		G::logname = optarg;
		break;
	default:
	    fprintf(stderr, "USAGE -b BIGENDIAN -m MAXCOLS -c COUNTCOL -s STEP -E MAXERRORS -N STDIN_IS_LIST_OF_FNAMES -v VERBOSE -L LOGNAME\n");
	    exit(1);
	}
    }
    if (optind < argc){
	G::fname = argv[optind];
    }
}


struct Calcs {
	unsigned long long ii = 0;
	unsigned long long previous_error = 0;
	unsigned errors = 0;
	unsigned error_report = 0;
	unsigned xx1;
};

int isramp(FILE* fp, Calcs& calcs){
	int file_ec = 0;
	for (unsigned xx; ; ++calcs.ii, calcs.xx1 = xx){
		unsigned buffer[G::maxcols];
		int nread = fread(buffer, sizeof(unsigned), G::maxcols, fp); // read  G::maxcols channels of data.

		if (nread != G::maxcols){
			break;
		}
		xx = buffer[G::countcol];

		if (G::bigendian){
			xx = ntohl(xx);
		}
		if (xx == calcs.xx1 + G::step) {
			calcs.error_report = 0;
		} else if (G::ignore_first_entry && calcs.ii==0){
			;
		} else {
			file_ec++;
			++calcs.errors;
			if (G::maxerrs && calcs.errors >= G::maxerrs){				// mv file out the way, FAST
				char fname_err[80];
				snprintf(fname_err, 80, "%s.err", G::fname);
				rename(G::fname, fname_err);
			}
			if (++calcs.error_report < 5){

				printf("%s: %lld: %012llx 0x%08x 0x%08x **ERROR** Sample jump: %8d, %10d bytes. Interval: %8lu, %10lu bytes\n",
						G::fname,
						calcs.error_report,
						calcs.ii, calcs.xx1, xx, xx - calcs.xx1, (xx-calcs.xx1)*G::maxcols*sizeof(unsigned),
						calcs.ii-calcs.previous_error, (calcs.ii-calcs.previous_error)*G::maxcols*sizeof(unsigned));
				calcs.previous_error = calcs.ii;
			}
			if (G::maxerrs && calcs.errors >= G::maxerrs){
				return -file_ec;
			}
		}
	}
	return file_ec;
}

FILE* fopen_read_or_die(const char* fname){
	FILE *fp = fopen(fname, "r");
	if (fp == 0){
		perror(fname);
		exit(errno);
	}
	return fp;
}

void print_report(const char* fname, int file_ec, Calcs& calcs)
{
	unsigned long long samples = calcs.ii/G::step;
	printf("%s status:%s cumulative: samples:%lld errors:%u error_rate:%.2f\n",
			fname, file_ec? "ERR": "OK", samples, calcs.errors, 100.0*calcs.errors/samples);
}

void write_log(Calcs& calcs){
	unsigned long long samples = calcs.ii/G::step;
        static FILE *fp;
        if (fp == 0){
                char filename[300];
		printf("log file: %s \n", G::logname);
		fp = fopen(G::logname, "w");
                if (fp == 0){
                        fprintf(stderr, "ERROR failed to open log file\n");
                        exit(1);
                }
        }
        fseek(fp, 0, SEEK_SET);
        fprintf(fp, "Total: %lld Errors %u \n", samples, calcs.errors);
}

int main(int argc, char* const argv[]) {
	get_args(argc, argv);

	FILE* fp = stdin;

	if (strcmp(G::fname, "-") != 0){
		fp = fopen_read_or_die(G::fname);
	}
	Calcs calcs = {};
	int file_ec;

	if (!G::stdin_is_list_of_fnames){
		file_ec = isramp(fp, calcs);
		if (G::verbose){
			print_report(G::fname, file_ec, calcs);
		}
		if (file_ec < 0){
			exit(1);
		}
	}else{
		char fname[132];
		G::fname = fname;                   // for isramp error report
		while(fgets(fname, 132, stdin)){
			int len = strlen(fname);
			if (len > 1){
				fname[len-1] = '\0';
				fp = fopen_read_or_die(fname);
				file_ec = isramp(fp, calcs);
				fclose(fp);
				if (G::verbose){
					print_report(fname, file_ec, calcs);
				}
				if (G::logname){
					write_log(calcs);
				}
				if (file_ec < 0){
					exit(1);
				}
			}
		}
	}
	return 0;
}
