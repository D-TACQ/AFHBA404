/*
 * shm.c
 *
 *  Created on: 12 Dec 2018
 *      Author: pgm
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#define SHM_INTS	128

#define SHM_LEN 	(SHM_INTS*sizeof(int))

int *shm;

void shm_connect()
{
	const int SIZE = SHM_LEN;

	int shm_fd;
	void *ptr;

	/* create the shared memory segment */
	shm_fd = shm_open("afhba-llcontrol", O_CREAT | O_RDWR, 0666);

	/* configure the size of the shared memory segment */
	ftruncate(shm_fd,SIZE);

	/* now map the shared memory segment in the address space of the process */
	ptr = mmap(0,SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
	if (ptr == MAP_FAILED) {
		printf("Map failed\n");
		exit(1);
	}


	shm = ptr;
	memset(ptr, 0, SIZE);
}
