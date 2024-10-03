
CXXFLAGS+=-I. -I.. -I../include -std=c++11 -O3

APPS=acqproc acqproc_custom acqproc_broadcast_trigger acqproc_network acqproc_hpr

all: $(APPS)
#	acqproc_custom

clean:
	rm -f *.o $(APPS)

doxygen:
	doxygen

acqproc.o: AcqSys.h

acqproc: acqproc.o AcqSys.o AcqHw.o DefaultSystemInterface.o afhba-llcontrol.o Env.o
	$(CXX) -O3 -o $@ $^	

# example custom acqproc. 
acqproc_custom: acqproc.o AcqSys.o AcqHw.o Custom1SystemInterface.o afhba-llcontrol.o Env.o
	$(CXX) -O3 -o $@ $^	
	
acqproc_broadcast_trigger: acqproc.o AcqSys.o AcqHw.o BroadcastSystemInterface.o afhba-llcontrol.o Env.o
	$(CXX) -O3 -o $@ $^
	
acqproc_network: acqproc.o AcqSys.o AcqHw.o NetworkSystemInterface.o afhba-llcontrol.o Env.o
	$(CXX) -O3 -o $@ $^	
	

acqproc_hpr: acqproc.o AcqSys.o AcqHw.o HudpRelaySystemInterface.o afhba-llcontrol.o Env.o knobs.o
	$(CXX) -O3 -o $@ $^
	
