#ifndef _GLOBAL_UTILITY_H
#define _GLOBAL_UTILITY_H
#include <iostream>
// 存储参数
class GlobalUtil
{
public:
	static int		_texMaxDim; //3200
    static int      _texMinDim; //16
	static int		_MemCapGPU; //0
	static int		_FitMemoryCap;  //0
	static int		_MaxFilterWidth;    //-1
	static int		_MaxOrientation;    //2
	static int      _OrientationPack2;  //0
	static float	_MaxFeaturePercent; //0.005
	static int		_MaxLevelFeatureNum;    //4096
	static int		_SubpixelLocalization;  //0
    static int      _TruncateMethod;    //0
	static int		_octave_min_default;    //0
	static int		_octave_num_default;    //4
	static int		_InitPyramidWidth;      //2048
	static int		_InitPyramidHeight;     //1536
	static int		_FixedOrientation;  //0
	static int		_LoweOrigin;    //0
	static int		_NormalizedSIFT;    //1
	static int		_FeatureCountThreshold;      //150
	static bool		_EnableDetailedTimings;     //0
	static float	_SiftDepthMin;  //0.1
	static float	_SiftDepthMax;  //4
};



#endif

