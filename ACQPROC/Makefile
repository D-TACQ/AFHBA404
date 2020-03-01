
CXXFLAGS+=-I.

all: acqproc_th




acqproc_th: acqproc_th.o AcqSys.o AcqHw.o
	$(CXX) -O3 -o $@ $^	
