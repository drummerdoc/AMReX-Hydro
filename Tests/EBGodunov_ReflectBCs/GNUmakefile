AMREX_HOME ?= ../../../amrex
AMREX_HYDRO_HOME = ../..

USE_MPI  = FALSE
USE_OMP  = FALSE

COMP = gnu

DIM = 3

DEBUG = FALSE

# Nothing here needs the Fortran interfaces.
BL_NO_FORT = TRUE

# The point of this test is the AMREX_ASSERTs in EBGodunovBC::Set*BCs, so keep
# assertions on even in an optimized build.
TEST = TRUE
USE_ASSERTION = TRUE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

# hydro_ebgodunov_bcs_K.H is header-only and pulls in nothing but AMReX Base
# (plus hydro_constants.H), so there is no need to build the rest of the
# library, or EB, to exercise it.
Pdirs := Base

Ppack += $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

Blocs := $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir))
Blocs += $(AMREX_HYDRO_HOME)/EBGodunov
Blocs += $(AMREX_HYDRO_HOME)/Utils

INCLUDE_LOCATIONS += $(Blocs)
VPATH_LOCATIONS   += $(Blocs)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
