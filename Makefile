# Define compiler 
FC		:= gfortran

# Compile flags
FCFlags	:= -fPIC -shared

# Check operating system for proper names
ifeq ($(OS), Windows_NT)
	RM := del /s /q *.dll
else
	RM := find . -type f -name '*.dll' -exec rm {} \;
endif

#===============================================================================
#===============================================================================
all: pdpr.dll radiation.dll 

pdpr.dll:
	$(FC) $(FCFlags) -o pdpr/functions.dll pdpr/functions.f90

radiation.dll:
	$(FC) $(FCFlags) -o radiation/radiation_calculator.dll radiation/radiation_calculator.f90

clean:
	@$(RM)  