
#DEBUG = -g -Og
#PROFILE = -pg
CXXFLAGS += -I .. -Ofast $(DEBUG) $(PROFILE)

APPS = rtm-t-stream-disk

apps: $(APPS)

rtm-t-stream-disk: rtm-t-stream-disk.o RTM_T_Device.o
	$(CXX) $(DEBUG) $(PROFILE) -o rtm-t-stream-disk rtm-t-stream-disk.o RTM_T_Device.o -lpopt

clean:
	rm -f *.o $(APPS)


