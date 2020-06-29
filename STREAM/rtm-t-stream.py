#!/usr/bin/env python3

import os
import ctypes
import numpy
import time
import threading
import traceback

class STREAM(threading.Thread):
    lib = ctypes.CDLL(os.path.realpath(os.path.dirname(__file__)+'/librtm-t-stream.so'))
    def __init__(self, NBUF, NMOD, NCHAN):
        super(STREAM, self).__init__(name=self.__class__.__name__)
        self.HANDLE = ctypes.c_void_p()
        self.NBUF = NBUF
        self.NMOD = NMOD
        self.NCHAN = NCHAN

    def stop(self):
        self.lib.RtmStreamStop(self.HANDLE)

    def run(self):
        buflen = ctypes.c_int32(-1)
        if self.lib.RtmStreamStart(ctypes.byref(self.HANDLE), 0, self.NBUF*self.NMOD, ctypes.byref(buflen)):
            raise Exception("start_stream")
        try:
            BUFLEN = int(buflen.value)
            shape = (BUFLEN//2//self.NCHAN, self.NMOD*self.NCHAN)
            for i in range(self.NBUF):
                buf = (ctypes.c_char*(self.NMOD*BUFLEN))()
                for off in range(0, self.NMOD*BUFLEN, BUFLEN):
                    err = self.lib.RtmStreamGetBuffer(self.HANDLE, ctypes.byref(buf, off), buflen)
                    if err:
                        if err < 0:
                            raise Exception("get_buffer")
                        else:
                           return
                arr = numpy.frombuffer(buf, dtype=numpy.int16).reshape(shape).T
                print(arr[0],arr[1])
        except Exception:
            traceback.print_exc()
            print(self.lib.RtmStreamClose(self.HANDLE))

stream = STREAM(100, 6, 16)
stream.start()
stream.join(5)
if stream.is_alive():
    print("stop")
    stream.stop()
stream.join(1)
print(stream.is_alive())
