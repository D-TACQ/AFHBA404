
DEBUG = -g
#DEBUG += -Og
CXXFLAGS += -fPIC -I .. -Ofast $(DEBUG)

APPS = rtm-t-stream-disk librtm-t-stream.so

apps: $(APPS)

rtm-t-stream-disk: rtm-t-stream-disk.o RTM_T_Device.o
	$(CXX) $(DEBUG) -o rtm-t-stream-disk rtm-t-stream-disk.o RTM_T_Device.o -lpopt

librtm-t-stream.so: rtm-t-stream.o RTM_T_Device.o
	$(CXX) $(DEBUG) -shared -o librtm-t-stream.so rtm-t-stream.o RTM_T_Device.o -lpopt

clean:
	rm -f *.o $(APPS)


