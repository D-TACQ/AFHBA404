#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>		/* @@todo use site standard popt() */
#include <arpa/inet.h>

#include <errno.h>
#include <signal.h>
#include <time.h>

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
	int period_report_sec = 0;     // report every N seconds
	int period_message_req;        // set by signal, cleared by report
};

void period_message_req_enable(int s)
{
	G::period_message_req = 1;
//	printf("%s G::period_message_req: %d\n", __FUNCTION__, G::period_message_req);
	alarm(G::period_report_sec);
}


#include <string>

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

void get_args(int argc, char* const argv[]){
    int opt;
    while((opt = getopt(argc, argv, "b:m:c:s:i:E:N:v:L:p:")) != -1) {
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
	case 'p':
		G::period_report_sec = atoi(optarg);
		if (G::period_report_sec){
			signal(SIGALRM, period_message_req_enable);
			alarm(G::period_report_sec);
		}
		break;
	case 'L':
		G::logname = optarg;
		break;
	default:
	    fprintf(stderr, "USAGE -b BIGENDIAN -m MAXCOLS -c COUNTCOL -s STEP -E MAXERRORS -N STDIN_IS_LIST_OF_FNAMES -v VERBOSE -L LOGNAME -p PERIOD_REPORT\n");
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
} G_calcs;

int isramp(FILE* fp, Calcs& calcs){
	int file_ec = 0;
	Calcs previous_calcs = {};
	unsigned errored_intervals = 0;
	unsigned clean_intervals = 0;

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
				if (!G::logname){
					printf("%s: %d: %012llx 0x%08x 0x%08x **ERROR** Sample jump: %8d, %10ld bytes. Interval: %8llu, %10llu bytes\n",
							G::fname,
							calcs.error_report,
							calcs.ii, calcs.xx1, xx, xx - calcs.xx1, (xx-calcs.xx1)*G::maxcols*sizeof(unsigned),
							calcs.ii-calcs.previous_error, (calcs.ii-calcs.previous_error)*G::maxcols*sizeof(unsigned));
					calcs.previous_error = calcs.ii;
				}
			}
			if (G::maxerrs && calcs.errors >= G::maxerrs){
				return -file_ec;
			}
		}
		if (G::period_message_req){
			float mbps = 0;
			const char* status = "CLEAN";
			if (previous_calcs.ii){
				mbps = (calcs.ii - previous_calcs.ii)*G::maxcols*sizeof(unsigned);
				mbps /= (G::period_report_sec*0x100000);
			}
			if (calcs.errors > previous_calcs.errors){
				errored_intervals++;
				status = "DIRTY";	
			}else{
				clean_intervals++;
			}	
			printf("%s bytes: 0x%012llx %12llu %6.2f MB/s errors: %u %s intervals: clean:%u dirty:%u\n", 
					currentDateTime().c_str(), calcs.ii, calcs.ii, mbps, calcs.errors, 
					status, clean_intervals, errored_intervals);
			G::period_message_req = 0;
			previous_calcs = calcs;
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

void catch_int(int sig_num)
{
	static time_t last_time;
	time_t now = time(0);

	if (now-last_time < 1){
		exit(1);
	}else{
		last_time = now;
	}
	print_report("isramp", G_calcs.errors, G_calcs);
}
int main(int argc, char* const argv[]) {
	get_args(argc, argv);

	FILE* fp = stdin;

	if (strcmp(G::fname, "-") != 0){
		fp = fopen_read_or_die(G::fname);
	}

	int file_ec;

	signal(SIGINT, catch_int);

	if (!G::stdin_is_list_of_fnames){
		file_ec = isramp(fp, G_calcs);
		if (G::verbose){
			print_report(G::fname, file_ec, G_calcs);
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
				file_ec = isramp(fp, G_calcs);
				fclose(fp);
				if (G::verbose){
					print_report(fname, file_ec, G_calcs);
				}
				if (G::logname){
					write_log(G_calcs);
				}
				if (file_ec < 0){
					exit(1);
				}
			}
		}
	}
	return 0;
}
