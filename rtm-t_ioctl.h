/** @file rtm-t_ioctl.h
 *  @brief RTM-T [AFHBA404] Driver external API.						     */
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

/** @file rtm-t_ioctl.h D-TACQ RTM-T Device Driver API. */

#ifndef __RTM_T_IOCTL_H__
#define __RTM_T_IOCTL_H__
#include <linux/ioctl.h>
struct LLC_DEF			/**< arg for ioctl RTM_T_START_LLC */
{
	u8 clk_div;		/**< 1..255: ECM value 1..255 usec with 1MHz EXTCLK */
	u8 fill;
	u8 clk_pos;		/**< 1: clk rising */
	u8 trg_pos;		/**< 1: trg rising */
	u32 target_pa;		/**< target bus address round to 1K boundary.
	                         *   RTM_T_USE_HOSTBUF=> use Driver buffer 0 */
};

struct AO_LLC_DEF	/** arg for ioctl RTM_T_START_AOLLC         */
{
	int length;		/**< length in bytes 8, 64 or 72             */
	u32 src_pa;		/**< source bus address round to 1k boundary.
			         *   RTM_T_USE_HOSTBUF=> use Driver buffer 0 */
};

struct XLLC_DEF {
	unsigned len;     	/** length in bytes - will round up to next %64 */
	u32 pa;		  		/**< SRC or DST buffer PA - round to 1K
	 	 	   	   	   	 *   RTM_T_USE_HOSTBUF=> use Driver buffer 0 */
};

struct AB 			/** Define two buffers A, B for ping/pong        */
{
	struct XLLC_DEF buffers[2];
};

#define MAXABN	256
//----GPU-related-definitions--------------------------------------------------
// for boundary alignment requirement
#define GPU_BOUND_SHIFT 16
#define GPU_BOUND_SIZE ((u64)1 << GPU_BOUND_SHIFT)
#define GPU_BOUND_OFFSET (GPU_BOUND_SIZE-1)
#define GPU_BOUND_MASK (~GPU_BOUND_OFFSET)

//-----------------------------------------------------------------------------

struct gpudma_lock_t {
    void*    handle;

    uint64_t addr_ai;
    uint64_t size_ai;
    size_t   page_count_ai;
    int      ind_ai;

    uint64_t addr_ao;
    uint64_t size_ao;
    size_t   page_count_ao;
    int      ind_ao;
};

struct ABN 			/** Define N buffers.   */
{
	int ndesc;
	struct XLLC_DEF buffers[MAXABN];
	/* others tag on behind */
};

#define MAX_AO_BUF		4

#define AO_BURST_ID		0xA0B55555
struct gpudma_unlock_t {
    void*    handle;
};

//-----------------------------------------------------------------------------

struct gpudma_state_t {
    void*       handle;
    size_t      page_count;
    size_t      page_size;
    uint64_t    pages[1];
};

struct AO_BURST {
	unsigned id;
	unsigned nbuf;
	unsigned tickms;       // msec per tick : 0 means use interrupt
};

#define VALID_AO_BURST(p) (((struct AO_BURST*)p)->id == AO_BURST_ID)

#define RTM_T_USE_HOSTBUF	0

#define DMAGIC 0xDB

#define RTM_T_START_STREAM	_IO(DMAGIC,   1)
/**< **ioctl** Start High Throughput Streaming */
#define RTM_T_START_LLC	 	_IOW(DMAGIC,   2, struct LLC_DEF)
/**< **ioctl** Start Low Latency Control */


#define RTM_T_START_STREAM_MAX	_IOW(DMAGIC,   3, u32)
/**< **ioctl** Start High Throughput Streaming specify max buffers. */

#define RTM_T_START_AOLLC	_IOW(DMAGIC,   4, struct AO_LLC_DEF)

#define AFHBA_START_AI_LLC	_IOWR(DMAGIC,   5, struct XLLC_DEF)
/**< **ioctl** ACQ2106 Start Low Latency Control Inbound
 * outputs actual pa used
 */

#define AFHBA_START_AO_LLC	_IOWR(DMAGIC,   6, struct XLLC_DEF)
/**< **ioctl** ACQ2106 Start Low Latency Control Outbound */

#define AFHBA_START_AI_AB	_IOWR(DMAGIC,   7, struct AB)
/**< **ioctl** ACQ2106 Start AI, Buffer A/B struct XLLC_DEF [2].
 * streaming rules: 4K boundary, 1K size modulus
 */

#define AFHBA_GPUMEM_LOCK	_IOWR(DMAGIC,	8, struct gpudma_lock_t)
/* Pins address of GPU memory to use */

#define AFHBA_START_AI_ABN	_IOWR(DMAGIC, 8, struct ABN)
/**< **ioctl** AFHBA_START_AI_ABN LLC, multiple buffers, INPUT */
#define AFHBA_START_AO_ABN	_IOWR(DMAGIC, 9, struct ABN)
/**< **ioctl** AFHBA_START_AO_ABN LLC, multiple buffers, OUTPUT */

#define AFHBA_AO_BURST_INIT		 _IOWR(DMAGIC, 10, struct AO_BURST)
/**< **ioctl** define an AO_BURST setup */

#define AFHBA_AO_BURST_SETBUF  	_IOWR(DMAGIC, 12, u32)
/**< **ioctl** define current buffer id */


#define RTM_T_START_STREAM_AO _IO(DMAGIC,   11)
/**< **ioctl** RTM_T_START_STREAM_AO appears in stub app code, but not in driver .. */

struct StreamBufferDef {
	u32 ibuf;
	u32 esta;
};
#define IBUF_MAGIC	0xb1f00000
#define IBUF_MAGIC_MASK	0xfff00000
#define IBUF_IDX	0x000f0000
#define IBUF_IDX_SHL	16
#define IBUF_IBUF	0x0000ffff
#define ESTA_CRC	0x0000ffff
#define SBDSZ		sizeof(struct StreamBufferDef)

#endif /* __RTM_T_IOCTL_H__ */
