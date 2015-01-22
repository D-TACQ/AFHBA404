/*
 * RTM_T_Device.h
 *
 *  Created on: Apr 28, 2011
 *      Author: pgm
 */

#ifndef RTM_T_DEVICE_H_
#define RTM_T_DEVICE_H_

#include <map>
#include <string>

class RTM_T_Device {

	enum { CTRL_ROOT=-1, MINOR_REGREAD=253, MINOR_DMAREAD=254 };

	int devnum;
	int nbuffers;

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
	RTM_T_Device(int _devnum, int _nbuffers = MAXBUF);
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

	const char *getControlRoot(void) {
		return names[CTRL_ROOT].c_str();
	}
	int getDevnum(void) const {
		return devnum;
	}

	enum {
		MAXBUF = 16,		 // maximum buffers per read
		MAXLEN = 0x100000 	 // maximum length of buffer
	};
};

#endif /* RTM_T_DEVICE_H_ */
