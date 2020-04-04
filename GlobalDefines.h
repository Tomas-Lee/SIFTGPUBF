#pragma once
#ifndef _GLOBAL_DEFINES_
    #include "glog/logging.h"
    #define _GLOBAL_DEFINES_

    #define MINF __int_as_float(0xff800000)

    #define MAX_MATCHES_PER_IMAGE_PAIR_RAW 128
    #define MAX_MATCHES_PER_IMAGE_PAIR_FILTERED 25


    #define USE_LIE_SPACE



#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		getchar();
		exit(-1);
	}
}


#endif //_GLOBAL_DEFINES_