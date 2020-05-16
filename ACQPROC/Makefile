
CXXFLAGS+=-I. -std=c++11

APPS=acqproc acqproc_custom acqproc_broadcast_trigger

all: $(APPS)
#	acqproc_custom

clean:
	rm -f *.o $(APPS)


acqproc: acqproc.o AcqSys.o AcqHw.o DefaultSystemInterface.o afhba-llcontrol.o
	$(CXX) -O3 -o $@ $^	

# example custom acqproc. 
acqproc_custom: acqproc.o AcqSys.o AcqHw.o Custom1SystemInterface.o afhba-llcontrol.o
	$(CXX) -O3 -o $@ $^	
	
acqproc_broadcast_trigger: acqproc.o AcqSys.o AcqHw.o BroadcastSystemInterface.o afhba-llcontrol.o
	$(CXX) -O3 -o $@ $^
	
