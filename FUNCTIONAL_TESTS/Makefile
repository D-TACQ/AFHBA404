
CXXFLAGS=-O2 -std=c++11 -I../include
#CXXFLAGS+=-fopenmp
CFLAGS+=-O2 -std=c99 -I../include

APPS = hbfarm425 checkramp checkramp480 multisitecheckramp checkES isramp pcmp

all: $(APPS)

clean:
	rm -f hbfarm425 checkramp


multisitecheckramp:
	g++ $(CXXFLAGS) multisitecheckramp.cpp -o multisitecheckramp -lpopt

pcmp: pcmp.cpp
	g++ $(CXXFLAGS) pcmp.cpp -o pcmp -lpopt

install: all
	cp $(APPS) /usr/local/bin

install_user: all
	cp $(APPS) $(HOME)/bin


	
