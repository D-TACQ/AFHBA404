
CXXFLAGS+=-I.

all: acqproc_th


clean:
	rm -f *.o acqproc_th


acqproc_th: acqproc_th.o AcqSys.o AcqHw.o afhba-llcontrol.o
	$(CXX) -O3 -o $@ $^	
