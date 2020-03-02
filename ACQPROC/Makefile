
CXXFLAGS+=-I.

all: acqproc_th


clean:
	rm -f *.0 acqproc_th


acqproc_th: acqproc_th.o AcqSys.o AcqHw.o afhba-llcontrol.o
	$(CXX) -O3 -o $@ $^	
