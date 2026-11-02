#ifndef DEFINE_SEEN
#define DEFINE_SEEN

/*
Daniel: SHORTCUT_FFT is not safe at this moment
It should not be used when tracking pairs
#ifndef SHORTCUT_FFT
#define SHORTCUT_FFT
#endif
*/

/*
Daniel: This flag can be set in order to enable FFTW to perform
        measurements of the fastest solution. On some systems this can give
	a speedup by a factor of a few. On other systems there is no difference or a slow down.
	In the longer run this should be modified to load and store WISDOM, see FFTW documentation.
  #define DO_MEASURE
*/

#ifndef SCALE_ENERGY
#define SCALE_ENERGY
#endif

#ifndef PAIR_SYN
#define PAIR_SYN
#endif

const int ZPOS=1;

const int MCROSS = 1;
const int AVERCROSS = 0;

#endif
