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
int main(int argc, char *argv[]) {

    int maxcols = 104; // Number of columns of data
    int countcol = 96; // Column where the ramp is
    int step = 1; // Default step. For sample counter in spad step = 1
    int bigendian = 0;
    int ignore_first_entry = 0;
    FILE* fp = stdin;
    char* fname = 0;

    int opt;
    while((opt = getopt(argc, argv, "b:m:c:s:i:")) != -1) {
        switch(opt) {
        case 'm':
            maxcols = atoi(optarg);
            break;
        case 'c':
            countcol = atoi(optarg);
            break;
        case 'b':
            bigendian = atoi(optarg);
            if (bigendian) printf("Hello Moto\n");
        case 's':
            step = atoi(optarg);
            printf("%i\n", atoi(optarg));
            break;
        case 'i':
            ignore_first_entry = atoi(optarg);
	    break;
        default:
            fprintf(stderr, "USAGE -b BIGENDIAN -m MAXCOLS -c COUNTCOL -s STEP\n");
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
        unsigned buffer[maxcols];
        int nread = fread(buffer, sizeof(unsigned), maxcols, fp); // read 104 channels of data.

        if (nread != maxcols){
	    break;
        }
        xx = buffer[countcol];

        if (bigendian){
	    xx = ntohl(xx);
        } 
        if (xx == xx1 + step) {
            error_report = 0;
        } else if (ignore_first_entry && ii==0){
           ;
        } else {
            if (++error_report < 5){

                printf("%s: %lld: %012llx 0x%08x 0x%08x **ERROR** Sample jump: %8d, %10d bytes. Interval: %8lu, %10lu bytes\n",
                fname,
                error_report,
            	ii, xx1, xx, xx - xx1, (xx-xx1)*maxcols*sizeof(unsigned), 
		ii-previous_error, (ii-previous_error)*maxcols*sizeof(unsigned));
		previous_error = ii;
            }
            ++errors;
        }
    }
    if (errors){
//        printf("%lld: total errors %d\n", ii, errors);
        return 1;
    }
    return 0;
}
