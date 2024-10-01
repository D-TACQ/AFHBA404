
CXXFLAGS=-O2 -std=c++11 -I../include
#CXXFLAGS+=-fopenmp
CFLAGS+=-O2 -std=c99 -I../include

all: hbfarm425 checkramp checkramp480 multisitecheckramp checkES isramp pcmp

clean:
	rm -f hbfarm425 checkramp


multisitecheckramp:
	g++ $(CXXFLAGS) multisitecheckramp.cpp -o multisitecheckramp -lpopt

pcmp: pcmp.cpp
	g++ $(CXXFLAGS) pcmp.cpp -o pcmp -lpopt

myinstall:
	cp hbfarm425 checkramp checkramp480 multisitecheckramp ~/bin

install:
	cp hbfarm425 checkramp /usr/local/bin
	
