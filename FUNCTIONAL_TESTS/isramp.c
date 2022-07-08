#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
/*
 * This file takes input from nc as below:
 * nc acq2106_112 4210 | pv | ./isramp
 */
int main(int argc, char *argv[]) {

    int maxcols = 104; // Number of columns of data
    int countcol = 96; // Column where the ramp is
    int step = 1; // Default step. For sample counter in spad step = 1.
    unsigned xx;
    int xx1 = 0;
    unsigned long long ii = 1;
    unsigned long long previous_error = 0;
    unsigned errors = 0;
    unsigned error_report = 0;
    unsigned int aa = 0;

    int opt;
    while((opt = getopt(argc, argv, "m:c:s:")) != -1)
    {
      switch(opt) {
        case 'm':
          maxcols = atoi(optarg);
          printf("%i\n", atoi(optarg));
          break;
        case 'c':
          countcol = atoi(optarg);
          printf("%i\n", atoi(optarg));
          break;
        case 's':
          step = atoi(optarg);
          printf("%i\n", atoi(optarg));
          break;
        default:
          printf("No args given %d \n", opt);
          break;
      }
    }
    unsigned buffer[maxcols];
    
    while(1) {

      int nread = fread(buffer, sizeof(unsigned), maxcols, stdin); // read 104 channels of data.

      if (nread != maxcols){
          printf("nread != maxcols %d %d\n", nread, maxcols);	
      }
      aa = buffer[countcol];

      if (aa == xx1 + step) {
          error_report = 0;
      } else {
          if (++error_report < 10){

            printf("%d: %012llx 0x%08x 0x%08x **ERROR** Sample jump: %8d, %10d bytes. Interval: %8lu, %10lu bytes\n",
                error_report,
            	ii, xx1, aa, aa - xx1, (aa-xx1)*maxcols*sizeof(unsigned), 
		ii-previous_error, (ii-previous_error)*maxcols*sizeof(unsigned));
		previous_error = ii;
          }
          ++errors;
      }

      ii++;
      xx1 = aa;
      }
      return 0;
}
