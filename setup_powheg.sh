#!/bin/bash
POWHEG_VERSION="3604"

if [ ! -d "POWHEG-BOX-V2" ]; then
    svn checkout --no-auth-cache --revision ${POWHEG_VERSION} --username anonymous --password anonymous svn://powhegbox.mib.infn.it/trunk/POWHEG-BOX-V2
fi

pushd POWHEG-BOX-V2
    if [ ! -d "gg_H_2HDM" ]; then
        svn co --no-auth-cache --revision ${POWHEG_VERSION} --username anonymous --password anonymous svn://powhegbox.mib.infn.it/trunk/User-Processes-V2/gg_H_2HDM
    fi    

    pushd gg_H_2HDM
        # modify the makefile to use -fallow-argument-mismatch compiler flag
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/FFLAGS= -Wall/FFLAGS= -Wall -fallow-argument-mismatch/" Makefile
            sed -i '' "s/FJCXXFLAGS+= $(shell \$(LHAPDF_CONFIG) --cxxflags)/FJCXXFLAGS+= $(shell \$(LHAPDF_CONFIG) --cxxflags) -std=c++11 /" Makefile
            sed -i '' "s/LIBS+= -lchaplin/LIBS+= -lchaplin -L lib/" Makefile
            # replace -lstdc++ with -lc++ for MAC OS
            sed -i '' "s/-lstdc++/-lc++/" Makefile

        else
            sed -i "s/FFLAGS= -Wall/FFLAGS= -Wall -fallow-argument-mismatch/" Makefile
            sed -i "s/FJCXXFLAGS+= $(shell \$(LHAPDF_CONFIG) --cxxflags)/FJCXXFLAGS+= $(shell \$(LHAPDF_CONFIG) --cxxflags) -std=c++11 /" Makefile
            sed -i "s/LIBS+= -lchaplin/LIBS+= -lchaplin -L lib/" Makefile
        fi
        # install lchaplin
        wget --no-verbose http://chaplin.hepforge.org/code/chaplin-1.2.tar
        tar xvf chaplin-1.2.tar
        pushd chaplin-1.2
            curl -o config.guess "https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD"
            curl -o config.sub "https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD"

            ./configure --prefix=`pwd`/..
            make install
        popd

        # compile powheg
        make pwhg_main
    popd
popd
