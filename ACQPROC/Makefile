
CXXFLAGS+=-I. -std=c++11

all: acqproc 
#	acqproc_custom

clean:
	rm -f *.o acqproc acqproc_custom


acqproc: acqproc.o AcqSys.o AcqHw.o DefaultSystemInterface.o afhba-llcontrol.o
	$(CXX) -O3 -o $@ $^	

# example custom acqproc. 
acqproc_custom: acqproc.o AcqSys.o AcqHw.o Custom1SystemInterface.o afhba-llcontrol.o
	$(CXX) -O3 -o $@ $^	
	
