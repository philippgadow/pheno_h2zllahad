#!/bin/bash
POWHEG_VERSION="3686"

if [ ! -d "generators/POWHEG-BOX-V2" ]; then
    mkdir -p generators/ && cd generators
    svn checkout --no-auth-cache --revision ${POWHEG_VERSION} --username anonymous --password anonymous svn://powhegbox.mib.infn.it/trunk/POWHEG-BOX-V2
    cd ../
fi

pushd generators/POWHEG-BOX-V2
    if [ ! -d "gg_H_2HDM" ]; then
        svn co --no-auth-cache --revision ${POWHEG_VERSION} --username anonymous --password anonymous svn://powhegbox.mib.infn.it/trunk/User-Processes-V2/gg_H_2HDM
        svn co --no-auth-cache --revision ${POWHEG_VERSION} --username anonymous --password anonymous svn://powhegbox.mib.infn.it/trunk/User-Processes-V2/Z
    fi

    pushd gg_H_2HDM
        # modify the makefile to use -fallow-argument-mismatch compiler flag
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "Mac OS detected, changing makefile of Powheg gg_H_2HDM"
	    ## on my system I need the -ffixed-line-length-none flag
            gsed -i "s/FFLAGS= -Wall/FFLAGS= -Wall -fallow-argument-mismatch -ffixed-line-length-none/" Makefile
	    ## using single quotes and escaping the dollar to avoid gsed error
            gsed -i 's/FJCXXFLAGS+= \$(shell \$(LHAPDF_CONFIG) --cxxflags)/FJCXXFLAGS+= \$(shell \$(LHAPDF_CONFIG) --cxxflags) -std=c++11 /' Makefile
            gsed -i "s/LIBS+= -lchaplin/LIBS+= -lchaplin -L lib/" Makefile
            # replace -lstdc++ with -lc++ for MAC OS
            gsed -i "s/-lstdc++/-lc++/" Makefile
        else
            echo "Linux detected, changing makefile of Powheg"
            sed -i "s/FFLAGS= -Wall/FFLAGS= -Wall -fallow-argument-mismatch/" Makefile
            sed -i "s/FJCXXFLAGS+= $(shell \$(LHAPDF_CONFIG) --cxxflags)/FJCXXFLAGS+= $(shell \$(LHAPDF_CONFIG) --cxxflags) -std=c++11 /" Makefile
            sed -i "s/LIBS+= -lchaplin/LIBS+= -lchaplin -L lib/" Makefile
        fi
        # install lchaplin
        wget --no-verbose http://chaplin.hepforge.org/code/chaplin-1.2.tar
        tar xvf chaplin-1.2.tar
        pushd chaplin-1.2
	    ## with curl I get empty files, switching to wget
            wget "https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD" -O config.guess
            wget "https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD" -O config.sub
	    ## Removing the default "-ipipe" argument from FFLAGS (with it, it uses clang instead of gfortran to compile .f files)
            export FFLAGS="-march=core2 -mtune=haswell -ftree-vectorize -fPIC -fstack-protector -O2 -isystem /Users/nitto/pheno_h2zllahad/micromamba/envs/micromamba_h2zllahad/include"
            ./configure --prefix=`pwd`/..
            make install
        popd

        # compile powheg
        make pwhg_main

        # compile input card: not there?
        #cp ../../configs/powheg/powheg.input powheg.input
    popd

    pushd Z
        # modify the makefile to use -fallow-argument-mismatch compiler flag
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "Mac OS detected, changing makefile of Powheg Z"
            gsed -i "s/#FFLAGS= -Wall/FFLAGS= -Wall -fallow-argument-mismatch -ffixed-line-length-none/" Makefile
            gsed -i 's/FJCXXFLAGS+= \$(shell \$(LHAPDF_CONFIG) --cxxflags)/FJCXXFLAGS+= \$(shell \$(LHAPDF_CONFIG) --cxxflags) -std=c++11 /' Makefile
            # replace -lstdc++ with -lc++ for MAC OS
            gsed -i "s/-lstdc++/-lc++/" Makefile
        else
            echo "Linux detected, changing makefile of Powheg"
            sed -i "s/FFLAGS= -Wall/FFLAGS= -Wall -fallow-argument-mismatch/" Makefile
            sed -i "s/FJCXXFLAGS+= $(shell \$(LHAPDF_CONFIG) --cxxflags)/FJCXXFLAGS+= $(shell \$(LHAPDF_CONFIG) --cxxflags) -std=c++11 /" Makefile
            sed -i "s/LIBS+= -lchaplin/LIBS+= -lchaplin -L lib/" Makefile
        fi

        # compile powheg
        make pwhg_main

        # compile input card: not there?
        #cp ../../configs/powheg/powheg.input powheg.input
    popd
popd

# install PDF sets with LHAPDF
lhapdf install NNPDF30_nlo_as_0118
lhapdf install MSTW2008nlo68cl
lhapdf install NNPDF23_lo_as_0130_qed
