#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>		/* @@todo use site standard popt() */
#include <arpa/inet.h>

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
};
int main(int argc, char *argv[]) {


    FILE* fp = stdin;
    char* fname = 0;

    int opt;
    while((opt = getopt(argc, argv, "b:m:c:s:i:E:")) != -1) {
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
        default:
            fprintf(stderr, "USAGE -b BIGENDIAN -m MAXCOLS -c COUNTCOL -s STEP -E MAXERRORS\n");
            return 1; 
        }
    }
    if (optind < argc){
        fp = fopen(fname = argv[optind], "r");
        if (fp == 0){
            fprintf(stderr, "ERROR: failed to open \"%s\"\n", fname);
            return 1;
        }
    }
    unsigned long long ii = 0;
    unsigned long long previous_error = 0;
    unsigned errors = 0;
    unsigned error_report = 0;

    for (unsigned xx, xx1 = 0; ; ++ii, xx1 = xx){    
        unsigned buffer[G::maxcols];
        int nread = fread(buffer, sizeof(unsigned), G::maxcols, fp); // read  G::maxcols channels of data.

        if (nread != G::maxcols){
	    break;
        }
        xx = buffer[G::countcol];

        if (G::bigendian){
	    xx = ntohl(xx);
        } 
        if (xx == xx1 + G::step) {
            error_report = 0;
        } else if (G::ignore_first_entry && ii==0){
           ;
        } else {
            ++errors;
            if (G::maxerrs && errors >= G::maxerrs){				// mv file out the way, FAST
             	char fname_err[80];
             	snprintf(fname_err, 80, "%s.err", fname);
             	rename(fname, fname_err);
            }
            if (++error_report < 5){

                printf("%s: %lld: %012llx 0x%08x 0x%08x **ERROR** Sample jump: %8d, %10d bytes. Interval: %8lu, %10lu bytes\n",
                fname,
                error_report,
            	ii, xx1, xx, xx - xx1, (xx-xx1)*G::maxcols*sizeof(unsigned),
		ii-previous_error, (ii-previous_error)*G::maxcols*sizeof(unsigned));
		previous_error = ii;
            }
            if (G::maxerrs && errors >= G::maxerrs){
        	    return 1;
            }
        }
    }
    if (errors){
        //printf("%lld: total errors %d\n", ii, errors);
        return 1;
    }
    return 0;
}
