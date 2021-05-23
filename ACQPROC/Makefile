
CXXFLAGS+=-I. -std=c++11 -O3

APPS=acqproc acqproc_custom acqproc_broadcast_trigger

all: $(APPS)
#	acqproc_custom

clean:
	rm -f *.o $(APPS)

doxygen:
	doxygen

acqproc: acqproc.o AcqSys.o AcqHw.o DefaultSystemInterface.o afhba-llcontrol.o Env.o
	$(CXX) -O3 -o $@ $^	

# example custom acqproc. 
acqproc_custom: acqproc.o AcqSys.o AcqHw.o Custom1SystemInterface.o afhba-llcontrol.o Env.o
	$(CXX) -O3 -o $@ $^	
	
acqproc_broadcast_trigger: acqproc.o AcqSys.o AcqHw.o BroadcastSystemInterface.o afhba-llcontrol.o Env.o
	$(CXX) -O3 -o $@ $^
	
