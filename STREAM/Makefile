
DEBUG = -g
#DEBUG += -Og
CXXFLAGS += -std=c++11 -fPIC -I.. -Ofast $(DEBUG)

APPS = rtm-t-stream-disk  \
	rtm-t-stream-disk-mux-AO-LLC rtm-t-stream-disk-mux-AO-STREAM rtm-t-stream-disk-mux-AO-STREAM_ALL \
	librtm-t-stream.so

apps: $(APPS)

all: apps

rtm-t-stream-disk: rtm-t-stream-disk.o RTM_T_Device.o InlineDataHandler.o
	$(CXX) $(DEBUG) -o $@ $^ -lpopt
	
rtm-t-stream-disk-mux-AO-LLC: rtm-t-stream-disk.o RTM_T_Device.o InlineDataHandlerMuxAO_LLC.o
	$(CXX) $(DEBUG) -o $@ $^ -lpopt
	
rtm-t-stream-disk-mux-AO-STREAM: rtm-t-stream-disk.o RTM_T_Device.o InlineDataHandlerMuxAO_STREAM.o
	$(CXX) $(DEBUG) -o $@ $^ -lpopt

rtm-t-stream-disk-mux-AO-STREAM_ALL: rtm-t-stream-disk.o RTM_T_Device.o InlineDataHandlerMuxAO_STREAM_ALL.o
	$(CXX) $(DEBUG) -o $@ $^ -lpopt
	
amx_dummy: amx_dummy.o RTM_T_Device.o
	$(CXX) $(DEBUG) -o $@ $^ -lpopt

librtm-t-stream.so: rtm-t-stream.o RTM_T_Device.o
	$(CXX) $(DEBUG) -shared -o librtm-t-stream.so rtm-t-stream.o RTM_T_Device.o -lpopt

clean:
	rm -f *.o $(APPS)


