/* rtm-t_ioctl.h RTM-T Driver external API				     */
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

struct AO_LLC_DEF
{				/**< arg for ioctl RTM_T_START_AOLLC         */
	int length;		/**< length in bytes 8, 64 or 72             */
	u32 src_pa;		/**< source bus address round to 1k boundary.
			         *   RTM_T_USE_HOSTBUF=> use Driver buffer 0 */
};

struct XLLC_DEF {
	unsigned len;     /**< length in bytes - will round up to next %64 */
	u32 pa;		  /**< SRC or DST buffer PA - round to 1K
	 	 	   *   RTM_T_USE_HOSTBUF=> use Driver buffer 0 */
};

struct AB {
	struct XLLC_DEF buffers[2];
};

#define MAXABN	16

struct ABN {
	int ndesc;
	struct XLLC_DEF buffers[MAXABN];
	/* others tag on behind */
};

#define RTM_T_USE_HOSTBUF	0

#define DMAGIC 0xDB

#define RTM_T_START_STREAM	_IO(DMAGIC,   1)
/**< ioctl Start High Throughput Streaming */
#define RTM_T_START_LLC	 	_IOW(DMAGIC,   2, struct LLC_DEF)
/**< ioctl Start Low Latency Control */


#define RTM_T_START_STREAM_MAX	_IOW(DMAGIC,   3, u32)
/**< ioctl Start High Throughput Streaming specify max buffers. */

#define RTM_T_START_AOLLC	_IOW(DMAGIC,   4, struct AO_LLC_DEF)

#define AFHBA_START_AI_LLC	_IOWR(DMAGIC,   5, struct XLLC_DEF)
/**< ioctl ACQ2106 Start Low Latency Control Inbound
 * outputs actual pa used
 */

#define AFHBA_START_AO_LLC	_IOWR(DMAGIC,   6, struct XLLC_DEF)
/**< ioctl ACQ2106 Start Low Latency Control Outbound */

#define AFHBA_START_AI_AB	_IOWR(DMAGIC,   7, struct AB)
/**< ioctl ACQ2106 Start AI, Buffer A/B struct XLLC_DEF [2]. 
 * streaming rules: 4K boundary, 1K size modulus
 */

#define AFHBA_START_AI_ABN	_IOWR(DMAGIC, 8, struct ABN)

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
