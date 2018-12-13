/*
 * afhba-get_shared_mapping.h
 *
 *  Created on: 13 Dec 2018
 *      Author: pgm
 */

#ifndef LLCONTROL_AFHBA_GET_SHARED_MAPPING_H_
#define LLCONTROL_AFHBA_GET_SHARED_MAPPING_H_

/* Kludge alert */
typedef unsigned       u32;
typedef unsigned short u16;
typedef unsigned char  u8;


#include "../rtm-t_ioctl.h"

#ifndef HB_LEN
#define HB_LEN  0x100000		/* 1MB HOST BUFFERSW */
#endif

void get_shared_mapping(int devnum, int ibuf, struct XLLC_DEF* xllc_def, void** pbuf);


#endif /* LLCONTROL_AFHBA_GET_SHARED_MAPPING_H_ */
