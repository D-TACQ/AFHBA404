/* ------------------------------------------------------------------------- */
/*   Copyright (C) 2013 Peter Milne, D-TACQ Solutions Ltd
 *                      <Peter dot Milne at D hyphen TACQ dot com>
    http://www.d-tacq.com

    This program is free software; you can redistribute it and/or modify
    it under the terms of Version 2 of the GNU General Public License
    as published by the Free Software Foundation;

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                */
/* ------------------------------------------------------------------------- */

/** @file xiloader decode/load xilinx .bit file
 * Refs:
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "popt.h"

#define VERID	"xiloader r1.01 (c) D-TACQ Solutions"
/* valid on Zynq */
#define FPGA_PORT "/dev/xdevcfg"

#define STDOUT_STR	"-"
#define STDIN_STR	"-"

typedef unsigned u32;

static inline u32 bs(u32 num) {
     u32 swapped = num>>24 & 0x000000ff | // move byte 3 to byte 0
	  num<<8  & 0x00ff0000 | // move byte 1 to byte 2
	  num>>8  & 0x0000ff00 | // move byte 2 to byte 1
	  num<<24 & 0xff000000;  // byte 0 to byte 3
     return swapped;
}

#define BYTESWAP(val)	bs(val)

#define MAXHEADER	1024

static const uint8_t ident_string[] = {
	0x00, 0x09, 0x0f, 0xf0, 0x0f, 0xf0, 0x0f, 0xf0,
	0x0f, 0xf0, 0x00, 0x00, 0x01, 0x61, 0x00
};

static const uint8_t bits_start_marker[] = {
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};





/* Command Line Argument Parsing*/

struct Options {
	const char *in;
	const char *out;
	int info;
	int load;
	int raw;
	int quiet;
	int input_len; 			/* if known */
} OPTS = {
		STDIN_STR, FPGA_PORT, 1, 0, 0, 0
};



static struct poptOption opt_table[] = {
	{ "outfile", 'o', POPT_ARG_STRING, &OPTS.out, 'o',
		"output destination [default:"FPGA_PORT "] .. -:stdout"},
	{ "infile",  'i', POPT_ARG_STRING, &OPTS.in, 'i',
		"input destination, - : stdin"			},
	{ "info",   'I',  POPT_ARG_NONE, 0,  'I', },
	{ "load",   'L',  POPT_ARG_NONE, 0,  'L', },
	{ "raw",    'R',  POPT_ARG_NONE, 0,  'R', },
	{ "quiet",  'q',  POPT_ARG_NONE, 0,  'q'  },
	POPT_AUTOHELP
	POPT_TABLEEND
};

void get_options(int argc, const char* argv[])
{
	poptContext opt_context =
                poptGetContext(argv[0], argc, argv, opt_table, 0);
	int rc;
	const char* key;

	while ((rc = poptGetNextOpt(opt_context)) > 0){
		switch(rc){
		case 'I':
			OPTS.info = 1; break;
		case 'L':
			OPTS.load = 1; break;
		case 'R':
			OPTS.raw = 1; break;
		case 'Q':
			OPTS.quiet = 1; break;
		}
	}
	/* alternatively, file name may be simple the first arg ..
	 */
	key = poptGetArg(opt_context);
	if (key != 0){
		OPTS.in = key;
	}
	if (!OPTS.quiet){
		fprintf(stderr, "%s\n", VERID);
	}
}

int input_size_unknown()
{
	return strcmp(OPTS.in, STDIN_STR) == 0;
}


FILE *ifp;
int icursor;

int getdata(void** buffer)
{
     void* buf;
     int maxlen;
     int len = 0;
     int nread;

     if (OPTS.in == 0){
	  printf("ERROR:infile not specified\n");
	  exit(1);
     }
     if (input_size_unknown()){
	     ifp = stdin;
	     maxlen = MAXHEADER;
     }else{
	     struct stat sb;
	     if (stat(OPTS.in, &sb) == -1){
		     perror(OPTS.in);
		     exit(1);
	     } else {
		     maxlen = len = sb.st_size;

		     if (len < MAXHEADER){
			  printf("ERROR: length too short\n");
			  exit(1);
		     }

		     ifp = fopen(OPTS.in, "r");
		     if (!ifp){
			  perror(OPTS.in);
			  exit(1);
		     }
	     }
     }


     buf = malloc(maxlen);
     icursor = nread = fread(buf, 1, maxlen, ifp);
     if (nread < -0){
	     perror(OPTS.in);
	     exit(1);
     }
     if (nread != maxlen){
	     if (nread < MAXHEADER){
		     fprintf(stderr, "failed to read enough data for header\n");
		     exit(1);
	     }
	     if (len){
		     fprintf(stderr, "length is known but failed to read\n");
		     exit(1);
	     }
     }

     *buffer = buf;
     return input_size_unknown() ? 0: nread;
}

void* get_remaining_data(void* buffer, int buffer_size)
{
	int nread;
	buffer = realloc(buffer, buffer_size);
	if (buffer == NULL){
		perror("realloc failed");
		exit(1);
	}
	nread = fread((char*)buffer+icursor, 1, buffer_size-icursor, ifp);
	if (nread < 0){
		perror("fread failed");
		exit(1);
	}
	if (nread+icursor < buffer_size){
		fprintf(stderr, "ERROR: not enough data:%d wanted %d\n",
				nread+icursor, buffer_size);
		exit(1);
	}
	return buffer;
}

int triplet(uint8_t * cursor, uint8_t t1, uint8_t t2, uint8_t t3)
{
     return cursor[0]==t1 && cursor[1]==t2 && cursor[2]==t3;
}


	
void print_header(uint8_t *header, int eoh_location, int stream_size)
{
     int ii = sizeof(ident_string);

     fprintf(stderr, "Xilinx Bitstream header.\n");
     fprintf(stderr, "%-25s : %x\n", "built with tool version", header[ii]);
     fprintf(stderr, "%-25s : ", "generated from filename");
/*                      123456789012345678901234 */
     while(++ii < eoh_location){
	  if (header[ii] == 0x3B){
	       break;
	  }else{
	       fprintf(stderr, "%c", header[ii]);
	  }
     }
     fprintf(stderr, "\n");

     for (; ii < eoh_location; ii++){
	  if (triplet(header+ii, 0x00, 0x62, 0x00)){
	       ii += 4;
	       break;
	  }
     }

     fprintf(stderr, "%-25s : ", "part");
     for (; ii < eoh_location; ii++){
	  if (triplet(header+ii, 0x00, 0x63, 0x00)){
	       ii += 3;
	       fprintf(stderr, "\n");
	       fprintf(stderr, "%-25s : ", "date");
	  }else if (triplet(header+ii, 0x00, 0x64, 0x00)){
	       ii += 3;
	       fprintf(stderr, "\n");
	       fprintf(stderr, "%-25s : ", "time");
	  }else if (triplet(header+ii, 0x00, 0x65, 0x00)){
	       break;
	  }else{
	       int cx = header[ii];
	       if (isprint(cx)){
		    fprintf(stderr, "%c", cx);
	       }
	  }
     }
     fprintf(stderr, "\n");
     fprintf(stderr, "%-25s : %d\n", "bitstream data starts at", eoh_location);
     fprintf(stderr, "%-25s : %d\n", "bitstream data size", stream_size);
}


/* compensate for the poor design of Xilinx driver that attempts to kmalloc ALL */

#define MAX1WRITE	0x100000
void load_bitstream(uint32_t* bitstream, int nbytes)
/* byte swap it and dump it into the FPGA */
{
     FILE *ofp = stdout;
     int totwrite = 0;
     char* bcursor = (char*)bitstream;

     if (strcmp(OPTS.out, STDOUT_STR) != 0){
	     ofp = fopen (OPTS.out, "w");
	     if (!ofp){
		     perror(OPTS.out);
		     exit(1);
	     }
     }

     if (!OPTS.raw){
	  int bit_words = nbytes/sizeof(uint32_t);
	  int ii;
	  for (ii = 0; ii <= bit_words; ii++) {
	       bitstream[ii] = BYTESWAP(bitstream[ii]);
	  }
     }


     while (totwrite < nbytes){
     	     int req_bytes = nbytes - totwrite;
     	     int act_bytes;
     	     if (req_bytes > MAX1WRITE) req_bytes = MAX1WRITE;
     	     act_bytes = fwrite(bcursor, 1, req_bytes, ofp);
     	     if (act_bytes != req_bytes){
     		     fprintf(stderr, "ERROR request: %d got %d bytes\n", req_bytes, act_bytes);
     	     }
     	     bcursor += act_bytes;
     	     totwrite += act_bytes;
     }

     fclose(ofp);
}

int is_bitstream_marker(uint8_t * cursor)
{
	return memcmp(cursor, bits_start_marker, sizeof bits_start_marker) == 0;
}

int main(int argc, const char* argv[])
{
     int ii;
     int eoh_location = 0;
     void* read_buffer;
     int in_size;
     uint32_t stream_size;
     uint8_t *header;
     int start_bitstream_found = 0;

     get_options(argc, argv);

     in_size = getdata(&read_buffer);
     header = (uint8_t *)read_buffer;


     for (ii = 0; ii < sizeof(ident_string)-1; ii++) {
	  if  (header[ii] == ident_string[ii]){
	       continue;
	  }else{
	       printf("Could not find Xilinx bitstream header.\n");
	       exit(1);
	  }
     }

     for (ii = 0; ii < MAXHEADER; ii++) {
	  if ( triplet(header+ii, 0x00, 0x65, 0x00)){
	       stream_size = (header[ii+3] << 16)|
		    (header[ii+4] <<  8)|
		    (header[ii+5]);
	       if (in_size){
		       eoh_location = in_size-stream_size;
	       }
	       break;
	  }
     }

     for (; ii < MAXHEADER; ++ii){
	     if (is_bitstream_marker(header+ii)){
		     start_bitstream_found = 1;
		     if (eoh_location == ii){
			     fprintf(stderr, "eoh_location matched\n");
			     break;
		     }else if (eoh_location == 0){
			     fprintf(stderr, "eoh_location set %d\n", eoh_location);
			     eoh_location = ii;
			     break;
		     }else{
			     fprintf(stderr, "ERROR eoh_location mismatch %d %d\n",
					eoh_location, ii);
			     exit(1);
		     }
	     }
     }
     if (start_bitstream_found == 0){
	     fprintf(stderr, "ERROR: bitstream data NOT found\n");
	     exit(1);
     }
     if (eoh_location == 0){	     
	     fprintf(stderr, "ERROR:eoh_location NOT FOUND\n");
	     exit(1);
     }

     if (OPTS.info && !OPTS.quiet){
	  print_header(header, eoh_location, stream_size);
     }
     if (OPTS.load){
	 if (OPTS.load && stream_size > in_size){
		 read_buffer = get_remaining_data(
				 read_buffer, stream_size+eoh_location);
		 header = (uint8_t *)read_buffer;
	 }

	 load_bitstream((uint32_t*)(header+eoh_location), stream_size);
     }
     exit(0);
}
