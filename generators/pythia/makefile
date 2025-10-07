PYINC=$(shell pythia8-config --cxxflags)
PYLIB=$(shell pythia8-config --libs)

CXXFLAGS= -std=c++11
ifeq ($(mode),debug)
CXXFLAGS += -D MYDEBUG
endif

all: main_PhPy8_HZA_dohist

CmdLine.o: CmdLine.cc CmdLine.hh
	g++ $(CXXFLAGS) -c CmdLine.cc

main_PhPy8_HZA_dohist.o: main_PhPy8_HZA_dohist.cc
	g++ $(CXXFLAGS) $(PYINC) -c main_PhPy8_HZA_dohist.cc

main_PhPy8_HZA_dohist: main_PhPy8_HZA_dohist.o CmdLine.o
	g++ $(CXXFLAGS) main_PhPy8_HZA_dohist.o CmdLine.o $(PYLIB) -lHepMC -o main_PhPy8_HZA_dohist
