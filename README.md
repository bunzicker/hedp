```hedp``` is a collection of functions I have written in the course of my PhD in Physics at The Ohio State University. They are centered around studying light-matter interactions using ultra-high intensity lasers. The main purpose of this package is to be a convenient place to store commonly used functions, but I hope it can be useful to others in the high energy density physics community. 

# Installation
To clone this repository from GitHub, use the command:
```
git clone https://github.com/bunzicker/hedp.git
```
Several packages in ```hedp``` call code written in Fortran90 to leverage its speed while maintaining Python's ease-of-use. Therefore, users must compile the code before calling it. To make this easy, I have included "Makefile". On most UNIX-based systems, simply calling ```make```
is sufficient to compile the codes. On Windows, it is recommended to use MinGW-make via ```mingw32-make```. I have written "Makefile" to use GNU's ```gfortran```, but it should be easy enough to change for different compilers.

If you would like to recompile the shared libraries use ```make clean``` and then call ```make``` again.


# Packages
```hedp``` contains several subpackages--all centered around relativistic laser-plasma interactions. A short description of each subpackage is given below. I try to keep this list updated as I add new features, but the most up-to-date information will be found by reading the code. 

## Laser
[```hedp.laser```](laser) has several functions to generate the electric field of a Laguerre-Gaussian laser. It also has functions to approximate the three-dimensional vector electric and magnetic fields of scalar solutions to the paraxial Maxwell equations.

## Radiation
[```hedp.radiation```](radiation) calculates the far field radiation from a relativistic particle. It follows the algorithm implemented in Pardal, *et al.* ([doi: 10.1016/j.cpc.2022.108634](https://doi.org/10.1016/j.cpc.2022.108634)) and in [the EPOCH fork on my profile.](https://github.com/bunzicker/epoch) It calculates the Lienerd-Weichert fields at some "virtual detector" and interpolates this field onto an arbitrary time array. This method can increase the detector's temporal resolution by a factor of $1/2\gamma^2$.

## PDPR
[```hedp.pdpr```](pdpr) implements the phase diversity-phase retrieval (PD-PR) method described in [Nick Czapla's PhD. thesis.](http://rave.ohiolink.edu/etdc/view?acc_num=osu1658486928321502) It uses an iterative algorithm to determine the magnitude and phase of the laser's electric field using several images taken different locations relative to the focal plane. 

> [!WARNING]
> This is still a work in progress. Expect updates soon. 

## TPS
[```hedp.tps```](tps) includes some simple code to analyze Thomson Parabola Spectrometer (TPS) data. Currently, this module's main function is to get the ion energy spectrum from 2D images. Details of the specific implementation of OSU's TPS can be found in Connor Winter's bachelor's thesis ['Development of a new Thomson parabola spectrometer for analysis of laser accelerated ions'](https://core.ac.uk/outputs/323062055/). More sophisticated analysis may be added later. 

## FROG
[```hedp.frog'''](frog) recovers the intensity and phase variations over time using the frequency-resolved optical gating technique. This implementation is based on [froglib](https://github.com/xmhk/froglib) written by Christoph Mahnke. It uses the principle components generalized projections recovery algorithm to reconstruct an experimentally measured SHG FROG trace.

> [!WARNING]
> This is still a work in progress. Expect updates soon. 
