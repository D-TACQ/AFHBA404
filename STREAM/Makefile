


CXXFLAGS += -I .. -O2

APPS = rtm-t-stream-disk
 
apps: $(APPS)


rtm-t-stream-disk: rtm-t-stream-disk.o RTM_T_Device.o
	$(CXX) -g -o rtm-t-stream-disk rtm-t-stream-disk.o RTM_T_Device.o -lpopt
	

clean:
	rm -f *.o $(APPS)
 
	
