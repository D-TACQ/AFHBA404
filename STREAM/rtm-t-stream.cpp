/* ------------------------------------------------------------------------- */
/* rtm-t-stream.so RTM-T PCIe Host Side stream access	             	     */
/* ------------------------------------------------------------------------- */
/*   Copyright (C) 2010 Peter Milne, D-TACQ Solutions Ltd
 *                      <Peter dot Milne at D hyphen TACQ dot com>

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

/** @file rtm-t-stream.cpp
 *  @brief D-TACQ PCIe RTM_T stream access solib
 * Continuous streaming using PCIe and buffers
 * */

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include "RTM_T_Device.h"
#include "local.h"
#include "rtm-t-stream.h"
#include "rtm-t_ioctl.h"

#define DBG(args...) // fprintf(stderr, args)

#ifdef __GNUC__
#define LIKELY(x) (__builtin_expect(!!(x), 1))
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

#define SBDN RTM_T_Device::MAXBUF
typedef struct {
	RTM_T_Device *dev;
	int stop;
	int nbuf;
	int ibuf;
	struct StreamBufferDef sbd[SBDN];
} STREAM, *STREAM_P;

#define TEST_NULLPTR(ptr)                                                      \
	do {                                                                       \
		if UNLIKELY (!ptr) {                                                   \
			fprintf(stderr, "NULL pointer error\n");                           \
			return -1;                                                         \
		}                                                                      \
	} while (0)
#define ERROR_START(errorcode, message)                                        \
	do {                                                                       \
		perror(message);                                                       \
		if (stream && stream->dev)                                             \
			delete stream->dev;                                                \
		if (stream)                                                            \
			free(stream);                                                      \
		return errorcode;                                                      \
	} while (0)

#define CHECK(i) (pfds[i].revents & POLLIN)
#define CHECK_STOP CHECK(0)
#define CHECK_FD CHECK(1)
static int fetch_buffers(STREAM_P const stream) {
	stream->ibuf = 0;
	const int fd = stream->dev->getDeviceHandle();
	struct pollfd pfds[2] = {
	    {.fd = stream->stop, .events = POLLIN},
	    {.fd = fd, .events = POLLIN},
	};
	const int ready = poll(pfds, 2, -1);
	if LIKELY (ready > 0) {
		if LIKELY (!CHECK_STOP && CHECK_FD) {
			const int nread = read(fd, stream->sbd, SBDN * SBDSZ);
			DBG("nread=%d\n", nread);
			if UNLIKELY (nread < 0) {
				perror("read error");
				return -1;
			} else {
				stream->nbuf = nread / SBDSZ;
				return 0;
			}
		} else {
			stream->nbuf = 0;
			return 0; // end if stream
		}
	} else if (ready < -1) {
		perror("poll error");
		return -1;
	} else {
		stream->nbuf = 0;
		return 0; // timeout == -1 so end if stream ?
	}
}

static inline int get_bufno(StreamBufferDef *sbd) {
	if UNLIKELY ((sbd->ibuf & IBUF_MAGIC_MASK) != IBUF_MAGIC) {
		fprintf(stderr, "ERROR NOT IBUF_MAGIC %08x %08x\n", sbd->ibuf,
		        sbd->esta);
		return -1;
	}
	return sbd->ibuf & IBUF_IBUF;
}

static inline int release_buffer(void *const handle, const int bufno) {
	return write(((STREAM_P)handle)->dev->getDeviceHandle(), &bufno,
	             sizeof(int)) != sizeof(int);
}

EXPORT int RtmStreamStart(void **handle, const int devnum, const int NBUFS,
                          int *const maxlen) {
	TEST_NULLPTR(handle);
	STREAM_P const stream = (STREAM_P)calloc(1, sizeof(STREAM));
	if UNLIKELY (!stream)
		ERROR_START(1, "malloc failed");
	if UNLIKELY ((stream->stop = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) < 0)
		ERROR_START(2, "eventfd failed");
	try {
		stream->dev = new RTM_T_Device(devnum);
	} catch (const std::exception &e) {
		ERROR_START(3, e.what());
	}
	const int fp = stream->dev->getDeviceHandle();
	const int rc = NBUFS ? ioctl(fp, RTM_T_START_STREAM_MAX, &NBUFS)
	                     : ioctl(fp, RTM_T_START_STREAM);
	if UNLIKELY (rc)
		ERROR_START(4, "ioctl failed");
	stream->nbuf = 0;
	stream->ibuf = 0;
	if (maxlen)
		*maxlen = stream->dev->maxlen;
	*handle = (void *)stream;
	return 0;
}

EXPORT int RtmStreamStop(void *handle) {
	TEST_NULLPTR(handle);
	STREAM_P const stream = (STREAM_P)handle;
	const uint64_t value = 1;
	return write(stream->stop, &value, sizeof(value)) != sizeof(value);
}

EXPORT int RtmStreamClose(void *handle) {
	TEST_NULLPTR(handle);
	STREAM_P const stream = (STREAM_P)handle;
	delete stream->dev;
	close(stream->stop);
	free(handle);
	return 0;
}

EXPORT int RtmStreamGetBuffer(void *handle, void *const buf, const int buflen) {
	TEST_NULLPTR(handle);
	TEST_NULLPTR(buf);
	STREAM_P const stream = (STREAM_P)handle;
	if (stream->ibuf == stream->nbuf) {
		DBG("update ibuf=%d nbuf=%d\n", stream->ibuf, stream->nbuf);
		if (fetch_buffers(stream)) {
			perror("fetch error");
			return -1;
		}
		if UNLIKELY (!stream->nbuf) {
			return 1;
		}
	}
	const int bufno = get_bufno(&stream->sbd[stream->ibuf++]);
	if (bufno == -1)
		return -1;
	DBG("ibuf=%d nbuf=%d bufno=%d\n", stream->ibuf, stream->nbuf, bufno);
	memcpy(buf, stream->dev->getHostBufferMapping(bufno), buflen);
	return release_buffer(stream, bufno);
}
