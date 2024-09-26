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

#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

class RTM_T_Device {
  std::string name_ctlroot;
  std::string name_dmaread;
  std::string name_regread;
  std::vector<void *> host_buffers;
  int dmaread_fd = -1;
  int regread_fd = -1;

  inline int _open(const char *name, int mode = O_RDONLY);
  inline void _close(void);

public:
  static constexpr int MAXBUF = 32; // maximum buffers per read
  const unsigned devnum;
  const unsigned nbuffers;
  const unsigned maxlen;
  const unsigned transfer_buffers;

  RTM_T_Device(int _devnum);
  virtual ~RTM_T_Device();
  const char *getDevice(void) { return name_dmaread.c_str(); }
  const int getDeviceHandle(void) { return dmaread_fd; }
  const char *getRegsDevice(void) { return name_regread.c_str(); }
  const void *getHostBufferMapping(int ibuf = 0) { return host_buffers[ibuf]; }
  void *getHostBufferMappingW(int ibuf = 0) { return host_buffers[ibuf]; }

  const char *getControlRoot(void) { return name_ctlroot.c_str(); }
  int getDevnum(void) const { return devnum; }
  int next(int ibuf) { return ++ibuf == nbuffers ? 0 : ibuf; }
};

#endif /* RTM_T_DEVICE_H_ */
