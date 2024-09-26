
DEBUG = -g
#DEBUG += -Og
CXXFLAGS += -std=c++11 -fPIC -I.. -I../include -Ofast $(DEBUG)

APPS = librtm-t-stream.so

apps: $(APPS)

all: apps

librtm-t-stream.so: rtm-t-stream.o RTM_T_Device.o
	$(CXX) $(DEBUG) -shared -o librtm-t-stream.so rtm-t-stream.o RTM_T_Device.o -lpopt

clean:
	rm -f *.o $(APPS)


