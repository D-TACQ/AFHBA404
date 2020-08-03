
DEBUG = -g
#DEBUG += -Og
CXXFLAGS += -fPIC -I.. -Ofast $(DEBUG)

APPS = rtm-t-stream-disk  rtm-t-stream-disk-mux-AO librtm-t-stream.so

apps: $(APPS)

all: apps

rtm-t-stream-disk: rtm-t-stream-disk.o RTM_T_Device.o InlineDataHandler.o
	$(CXX) $(DEBUG) -o rtm-t-stream-disk $^ -lpopt
	
rtm-t-stream-disk-mux-AO: rtm-t-stream-disk.o RTM_T_Device.o InlineDataHandlerMuxAO_LLC.o
	$(CXX) $(DEBUG) -o rtm-t-stream-disk $^ -lpopt	

librtm-t-stream.so: rtm-t-stream.o RTM_T_Device.o
	$(CXX) $(DEBUG) -shared -o librtm-t-stream.so rtm-t-stream.o RTM_T_Device.o -lpopt

clean:
	rm -f *.o $(APPS)


