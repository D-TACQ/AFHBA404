/*
 * RTM_T_Device.h
 *
 *  Created on: Apr 28, 2011
 *      Author: pgm
 */

/** @file RTM_T_Device.h
 *  @brief interface to RTM_T Device (historic name for AFHBA404)
 */

#ifndef RTM_T_DEVICE_H_
#define RTM_T_DEVICE_H_

#include <map>
#include <string>

#include <fcntl.h>
#include <unistd.h>

class RTM_T_Device {
	enum { CTRL_ROOT=-1, MINOR_REGREAD=253, MINOR_DMAREAD=254 };

	std::map<int, std::string> names;
	std::map<int, void*> host_buffers;
	std::map<int, int> handles;

	void _open(int id, int mode = O_RDWR){
		int fp = open(names[id].c_str(), mode);

		if (fp == -1){
			perror(names[id].c_str());
			_exit(errno);
		}else{
			handles[id] = fp;
		}
	}

	void _close(void){
		std::map<int, int>::const_iterator iter;

		for (iter = handles.begin(); iter != handles.end(); ++iter){
			close(iter->second);
		}
	}

public:
	const unsigned devnum;
	const unsigned nbuffers;
 	const unsigned maxlen;
	const unsigned transfer_buffers;

	RTM_T_Device(int _devnum);
	virtual ~RTM_T_Device() {
		_close();
	}
	const char *getDevice(void) {
		return names[MINOR_DMAREAD].c_str();
	}
	const int getDeviceHandle(void) {
		return handles[MINOR_DMAREAD];
	}
	const char *getRegsDevice(void) {
		return names[MINOR_REGREAD].c_str();
	}
	const void *getHostBufferMapping(int ibuf = 0) {
		return host_buffers[ibuf];
	}
	void *getHostBufferMappingW(int ibuf = 0) {
		return host_buffers[ibuf];
	}

	const char *getControlRoot(void) {
		return names[CTRL_ROOT].c_str();
	}
	int getDevnum(void) const {
		return devnum;
	}
	int next(int ibuf){
		return ++ibuf == nbuffers? 0: ibuf;
	}

	enum {
		MAXBUF = 32		 // maximum buffers per read
	};

};

#endif /* RTM_T_DEVICE_H_ */
