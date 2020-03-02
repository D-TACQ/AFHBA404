
CXXFLAGS+=-I.

all: acqproc

clean:
	rm -f *.o acqproc


acqproc: acqproc.o AcqSys.o AcqHw.o afhba-llcontrol.o
	$(CXX) -O3 -o $@ $^	
