
CXXFLAGS=-O2 -std=c++11 -fopenmp 

all: hbfarm425 checkramp checkramp480 multisitecheckramp

clean:
	rm -f hbfarm425 checkramp


multisitecheckramp:
	g++ $(CXXFLAGS) multisitecheckramp.cpp -o multisitecheckramp -lpopt

myinstall:
	cp hbfarm425 checkramp checkramp480 multisitecheckramp ~/bin

install:
	cp hbfarm425 checkramp /usr/local/bin
	
