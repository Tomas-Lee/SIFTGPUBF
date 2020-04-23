#include <cuda_runtime.h>   //定义这个能解决__align__关键词找不到的问题
#include <device_functions.h>

__align__(16) //has to be aligned to 16 bytes
 struct  SiftCameraParams {

	unsigned int m_depthWidth;
	unsigned int m_depthHeight;
	unsigned int m_intensityWidth;
	unsigned int m_intensityHeight;

	//float4x4 m_siftIntrinsics;
	//float4x4 m_siftIntrinsicsInv;

	//float4x4 m_downSampIntrinsics;
	//float4x4 m_downSampIntrinsicsInv;

	float m_minKeyScale;

	unsigned int dummy0;
	unsigned int dummy1;
	unsigned int dummy2;
};
