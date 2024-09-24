/*
 * RTM_T_Device.cpp
 *
 *  Created on: Apr 28, 2011
 *      Author: pgm
 */

/** @file RTM_T_Device.cpp
 *  @brief interface to RTM_T Device (historic name for AFHBA404)
 */

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "RTM_T_Device.h"
#include "local.h"

static unsigned calc_nbuffers(unsigned devnum);
static unsigned calc_maxlen(unsigned devnum);
static unsigned calc_transfer_buffers(unsigned devnum);

void throw_os_error(const std::string &hint, int err) {
	errno = err;
	throw std::runtime_error(hint);
}

int RTM_T_Device::_open(const char *name, int mode) {
	const int fp = open(name, mode);
	if (fp == -1) {
		const int err = errno;
		_close();
		throw_os_error(std::string("open ") + name, err);
	}
	return fp;
}

inline void RTM_T_Device::_close() {
	for (const auto buffer : host_buffers) {
		munmap(buffer, maxlen);
	}
	if (dmaread_fd != -1)
		close(dmaread_fd);
	if (regread_fd != -1)
		close(regread_fd);
}

RTM_T_Device::~RTM_T_Device() { _close(); }
RTM_T_Device::RTM_T_Device(int _devnum)
    : devnum(_devnum % 100), nbuffers(calc_nbuffers(devnum)),
      maxlen(calc_maxlen(devnum)),
      transfer_buffers(calc_transfer_buffers(devnum)) {
	char name[128];

	sprintf(name, "/dev/rtm-t.%u.ctrl", devnum);
	name_ctlroot = name;

	sprintf(name, "/dev/rtm-t.%u", devnum);
	name_dmaread = name;
	dmaread_fd = _open(name);

	sprintf(name, "/dev/rtm-t.%u.regs", devnum);
	name_regread = name;
	regread_fd = _open(name);
    host_buffers.reserve(nbuffers);
	for (int ib = 0; ib < nbuffers; ++ib) {
		sprintf(name, "/dev/rtm-t.%u.data/hb%04d", devnum, ib);
		const int fd = _open(name);
		void *const va =
		    mmap(0, maxlen, PROT_READ, MAP_SHARED, fd, 0);
		close(fd);
		if (va == (void *)-1) {
			const int err = errno;
			_close();
			throw_os_error(std::string("mmap ") + name, err);
		} else {
			host_buffers.push_back(va);
		}
	}
}

static int getKnob(const char *knob, unsigned *value) {
	FILE *const fp = fopen(knob, "r");
	if (!fp) {
		return -1;
	}
	const int rc = fscanf(fp, "%u", value);
	fclose(fp);
	return rc;
}

#define PARAMETERS "/sys/module/afhba/parameters/"
#define BUFFER_LEN PARAMETERS "buffer_len"
#define NBUFFERS PARAMETERS "nbuffers"
#define TRANSFER_BUFFERS PARAMETERS "transfer_buffers"

static unsigned calc_transfer_buffers(unsigned devnum) {
	unsigned transfer_buffers = 0;
	if (getKnob(TRANSFER_BUFFERS, &transfer_buffers) != 1) {
		throw std::runtime_error("getKnob " TRANSFER_BUFFERS " failed");
	}
	return transfer_buffers;
}

static unsigned calc_nbuffers(unsigned devnum) {
	unsigned nbuffers = 0;
	if (getKnob(NBUFFERS, &nbuffers) != 1) {
		throw std::runtime_error("getKnob " NBUFFERS " failed");
	}
	return nbuffers;
}
static unsigned calc_maxlen(unsigned devnum) {
	unsigned buffer_len = 0;
	char knob[80];
	snprintf(knob, 80, "/dev/rtm-t.%u.ctrl/buffer_len", devnum);
	if (getKnob(knob, &buffer_len) == 1) {
		return buffer_len;
	} else if (getKnob(BUFFER_LEN, &buffer_len) == 1) {
		return buffer_len;
	} else {
		throw std::runtime_error("buffer_len not set");
	}
}
