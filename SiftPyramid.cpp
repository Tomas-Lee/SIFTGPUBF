#include <string.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <fstream>
#include <math.h>


#include "GlobalUtil.h"
#include "SiftPyramid.h"
#include "SiftGPU.h"
#include "CuTexImage.h"
#include "ProgramCU.h"

#include "CUDATimer.h"

SiftPyramid::SiftPyramid(SiftParam&sp) :param(sp)
{
	_featureNum = 0;
	_levelFeatureNum = NULL;

	//image size
	_octave_num = 0;
	_octave_min = 0;
	_pyramid_octave_num = _pyramid_octave_first = 0;
	_pyramid_width = _pyramid_height = 0;
	_allocated = false;

	_allPyramid = NULL;
	_featureTexRaw = NULL;
	_featureTexFinal = NULL;
	_orientationTex = NULL;
	_inputTex = new CuTexImage();

	d_featureCount = NULL;

	d_outDescriptorList = NULL;

	_timer = new CUDATimer();
}


SiftPyramid::~SiftPyramid()
{
	DestroyPerLevelData();
	DestroyPyramidData();
	if (_inputTex) delete _inputTex;
	if (_timer) delete _timer;

	if (d_featureCount) cutilSafeCall(cudaFree(d_featureCount));
	if (d_outDescriptorList) cutilSafeCall(cudaFree(d_outDescriptorList));
}


void SiftPyramid::BuildPyramid(float* d_data)   //d_data is in GPU, 这里读入的d_data是对的
{
	int i, j;
	for (i = _octave_min; i < _octave_min + _octave_num; i++)   //octave_min=0; octave_num=4,总共4组
	{
		float* filter_sigma = param._sigma;
		CuTexImage *tex = GetBaseLevel(i);
		CuTexImage *buf = GetBaseLevel(i, DATA_KEYPOINT) + 2;   //tex和buf这两个变量都是对的

		j = param._level_min + 1;
		if (i == _octave_min)   //对于最下面一组
		{
			_inputTex->setImageData(_pyramid_width, _pyramid_height, 1, d_data);    //_pyramid_width=2048
			//ConvertInputToCU(input);
			if (i == 0)
			{
				ProgramCU::FilterImage(tex, _inputTex, buf,
					param.m_filterWidths[0], 0);    //_inputTex有23%值不一样
			}
			else
			{
				if (i < 0)	ProgramCU::SampleImageU(tex, _inputTex, -i);
				else		ProgramCU::SampleImageD(tex, _inputTex, i);

				ProgramCU::FilterImage(tex, tex, buf,
					param.m_filterWidths[0], 0);
			}
		}
		else    //对于其他组
		{
			ProgramCU::SampleImageD(tex, GetBaseLevel(i - 1) + param._level_ds - param._level_min);
			if (param._sigma_skip1 > 0)
			{
				std::cout << "ERROR" << std::endl;
				//ProgramCU::FilterImage(tex, tex, buf, param._sigma_skip1);
			}
		}
		for (; j <= param._level_max; j++, tex++, filter_sigma++)   //对于一组的所有层
		{
			// filtering
			ProgramCU::FilterImage(tex + 1, tex, buf, param.m_filterWidths[j + 1], j + 1);
		}
	}

	{ // resize the feature images
		unsigned int idx = 0;
		for (i = 0; i < _octave_num; i++) {
			CuTexImage * tex = GetBaseLevel(i + _octave_min);
			int fmax = int(tex->GetImgWidth() * tex->GetImgHeight()*GlobalUtil::_MaxFeaturePercent);
			//
			if (fmax > GlobalUtil::_MaxLevelFeatureNum) fmax = GlobalUtil::_MaxLevelFeatureNum;
			else if (fmax < 32) fmax = 32;	//give it at least a space of 32 feature

			for (j = 0; j < param._dog_level_num; j++, idx++) {
				_featureTexRaw[idx].InitTexture(fmax, 1, 4);
			}
		}
	}
	ProgramCU::CheckErrorCUDA(__FUNCTION__);
}


void SiftPyramid::RunSIFT(float* d_colorData, const float* d_depthData)
{
	//*****************
	//build the pyramid
	//*****************
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->startEvent("BuildPyramid");
	}
	BuildPyramid(d_colorData);
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->endEvent();
	}

	//*****************
	//detect key points
	//*****************
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->startEvent("DetectKeypoints");
	}
	DetectKeypoints(d_depthData);
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->endEvent();
	}

	//********************
	//limit feature count
	//********************
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->startEvent("LimitFeatureCount");
	}
	LimitFeatureCount(0);
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->endEvent();
	}

	//************************
	//get feature orientations
	//************************
	//some extra tricks are done to handle existing key point list
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->startEvent("GetFeatureOrientations");
	}
	GetFeatureOrientations();
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->endEvent();
	}

	//************************
	//reshape feature list
	//************************
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->startEvent("ReshapeFeatureList");
	}
	ReshapeFeatureList(); // actually get key pos/size/orientation	
	LimitFeatureCount(1);
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->endEvent();
	}

	//************************
	//compute feature descriptors
	//************************
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->startEvent("GetFeatureDescriptors");
	}
	GetFeatureDescriptors();
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->endEvent();
	}
}


void SiftPyramid::LimitFeatureCount(int have_keylist)
{

	if (GlobalUtil::_FeatureCountThreshold <= 0) return;
	///////////////////////////////////////////////////////////////
	//skip the lowest levels to reduce number of features. 

	if (GlobalUtil::_TruncateMethod == 2)
	{
		int i = 0, new_feature_num = 0, level_num = param._dog_level_num * _octave_num;
		for (; new_feature_num < _FeatureCountThreshold && i < level_num; ++i) new_feature_num += _levelFeatureNum[i];
		for (; i < level_num; ++i)            _levelFeatureNum[i] = 0;

		if (new_feature_num < _featureNum)
		{
			_featureNum = new_feature_num;
		}
	}
	else
	{
		int i = 0, num_to_erase = 0;
		while (_featureNum - _levelFeatureNum[i] > _FeatureCountThreshold)
		{
			num_to_erase += _levelFeatureNum[i];
			_featureNum -= _levelFeatureNum[i];
			_levelFeatureNum[i++] = 0;
		}
	}
}

int SiftPyramid::GetRequiredOctaveNum(int inputsz)
{
	//[2 ^ i,  2 ^ (i + 1)) -> i - 3...
	//768 in [2^9, 2^10)  -> 6 -> smallest will be 768 / 32 = 24
	int num = (int)floor(log(inputsz * 2.0 / GlobalUtil::_texMinDim) / log(2.0));
	return num <= 0 ? 1 : num;
}

//void SiftPyramid::CopyFeaturesToCPU(float* keys, float *descriptors)
//{
//	if (keys && d_outKeypointList) {
//		cutilSafeCall(cudaMemcpy(keys, d_outKeypointList, sizeof(float4) * _featureNum, cudaMemcpyDeviceToHost));
//	}
//	if (descriptors && d_outDescriptorList) {
//		cutilSafeCall(cudaMemcpy(descriptors, d_outDescriptorList, sizeof(float) * 128 * _featureNum, cudaMemcpyDeviceToHost));
//	}
//}

void SiftPyramid::GetFeatureDescriptors()
{
	//descriptors...
	unsigned int descOffset = 0;
	CuTexImage * got;
	CuTexImage *ftex = _featureTexFinal;
	for (int i = 0, idx = 0; i < _octave_num; i++)
	{
		got = GetBaseLevel(i + _octave_min, DATA_GRAD) + 1;
		for (int j = 0; j < param._dog_level_num; j++, ftex++, idx++, got++)
		{
			if (_levelFeatureNum[idx] == 0) continue;
			ProgramCU::ComputeDescriptor(ftex, got, d_outDescriptorList + descOffset, IsUsingRectDescription());//process

			descOffset += 128 * _levelFeatureNum[idx];
		}
	}

	ProgramCU::CheckErrorCUDA(__FUNCTION__);
}


void SiftPyramid::ReshapeFeatureList()
{
	//!!!TODO HAVE OPTION FOR NUM_ORIENTATION = 1 (no need to expand or decrease) -> no atomic

	int n = param._dog_level_num*_octave_num;
	float os = _octave_min >= 0 ? float(1 << _octave_min) : 1.0f / (1 << (-_octave_min));

	_featureNum = 0;

	for (int i = 0; i < n; i++) {
		if (_levelFeatureNum[i] == 0) continue;
		float keyLocScale = os * (1 << (i / param._dog_level_num));

		unsigned int numFeatures = ProgramCU::ReshapeFeatureList(&_featureTexRaw[i], &_featureTexFinal[i], d_featureCount, keyLocScale);
		SetLevelFinalFeatureNum(i, numFeatures);
		_featureNum += numFeatures;
	}
}

void SiftPyramid::DestroyPerLevelData()
{
	//integers vector to store the feature numbers.
	if (_levelFeatureNum) {
		delete[] _levelFeatureNum;
		_levelFeatureNum = NULL;
	}
	//texture used to store features
	if (_featureTexRaw) {
		delete[] _featureTexRaw;
		_featureTexRaw = NULL;
	}
	if (_featureTexFinal) {
		delete[] _featureTexFinal;
		_featureTexFinal = NULL;
	}
	//texture used for multi-orientation 
	if (_orientationTex) {
		delete[] _orientationTex;
		_orientationTex = NULL;
	}
	int no = _octave_num* param._dog_level_num;
}

void SiftPyramid::DestroyPyramidData()
{
	if (_allPyramid)
	{
		delete[] _allPyramid;
		_allPyramid = NULL;
	}
}



void SiftPyramid::DetectKeypoints(const float* d_depthData)
{
	cutilSafeCall(cudaMemset(d_featureCount, 0, sizeof(int) * _octave_num * param._dog_level_num)); //set gpu array d_featureCount to 0=int[12]

	float os = _octave_min >= 0 ? float(1 << _octave_min) : 1.0f / (1 << (-_octave_min));
	float keyLocOffset = GlobalUtil::_LoweOrigin ? 0 : 0.5f;

	for (int i = _octave_min; i < _octave_min + _octave_num; i++)   //计算每一个组的DoG金字塔,_octave_min=0; _octave_num=4，这里循环4组
	{
		CuTexImage * gus = GetBaseLevel(i) + 1; //data_gaussian
		CuTexImage * dog = GetBaseLevel(i, DATA_DOG) + 1;
		CuTexImage * got = GetBaseLevel(i, DATA_GRAD) + 1;  //data_grad
		//compute the gradient
        for (int j = param._level_min + 1; j <= param._level_max; j++, gus++, dog++, got++)     //_level_min=-1, _level_max=4,这里循环三轮
        {
			//input: gus and gus -1
			//output: gradient, dog, orientation
			ProgramCU::ComputeDOG(gus, dog, got);
		}
	}

	for (int i = _octave_min; i < _octave_min + _octave_num; i++)   //计算每一组的关键点,_octave_min=0; _octave_num=4，这里循环4组
	{
		CuTexImage * dog = GetBaseLevel(i, DATA_DOG) + 2;
		CuTexImage * key = GetBaseLevel(i, DATA_KEYPOINT) + 2;
		for (int j = param._level_min + 2; j < param._level_max; j++, dog++)    //_level_min=-1, _level_max=4,这里循环三轮
		{
			int featureOctLevelIndex = (i - _octave_min) * param._dog_level_num + j - 1;
			float keyLocScale = os * (1 << (featureOctLevelIndex / param._dog_level_num));  //_dog_level_num=3
//			LOG(INFO)<<"featureOctLevelIndex = "<<featureOctLevelIndex;     //=0
//			LOG(INFO)<<"keyLocScale = "<<keyLocScale;   //1
			//input, dog, dog + 1, dog -1
			//output, key
			/*
			输入：
			    CuTexImage* dog 一样,输入输出都一样, mydog1=mydog2=dog1=dog2
			    const float* d_depthData 一样, 输入是一样的,输出也一样myd1=myd2=d1=d2
			    CuTexImage* featureList 这是输入，在输入的时候因为随机初始化，所以两者不一样，但是输出好像也不一样。f1=f2, myf1=myf2, f1!=myf1
			输出： ???
			    CuTexImage* key 输入是一样的，输出也是一样的，输入均不等于输出, myk1!=myk2 k1!=k2 myk1=k1; myk2=k2

			    int* d_featureCount 不一样，这肯定是输出
			input param: 所有都一样
			    float Tdog;
			    float Tedge;
			    unsigned int featureOctLevelidx,
			    float keyLocScale;
			    float keyLocOffset;
			    float siftDepthMin;
			    float siftDepthMax;
            */
//			LOG(INFO)<<"Tdog = "<<param._dog_threshold; //=0.00666667
//            LOG(INFO)<<"Tedge = "<<param._edge_threshold;   //=1
//            LOG(INFO)<<"featureOctLevelidx = "<<featureOctLevelIndex;   //=0
//            LOG(INFO)<<"keyLocScale = "<<keyLocScale;   //1
//            LOG(INFO)<<"keyLocOffset = "<<keyLocOffset; //0.5
//            LOG(INFO)<<"siftDepthMin = "<<GlobalUtil::_SiftDepthMin;    //=0.1
//            LOG(INFO)<<"siftDepthMax = "<<GlobalUtil::_SiftDepthMax;    //4
//            SaveTexImage(&_featureTexRaw[featureOctLevelIndex],"/home/atom/RobotProjects/SIFTGPUS/siftgpuBF/tex/f1.bin");
//            SaveTexImage(key,"/home/atom/RobotProjects/SIFTGPUS/siftgpuBF/tex/k1.bin");
//            SaveTexImage(dog,"/home/atom/RobotProjects/SIFTGPUS/siftgpuBF/tex/dog1.bin");
//            SaveImage(d_depthData,2048*1536,"/home/atom/RobotProjects/SIFTGPUS/siftgpuBF/tex/myd_data1.bin");
//            LoadTexImage(&_featureTexRaw[featureOctLevelIndex],"/home/atom/RobotProjects/SIFTGPUS/siftgpuBF/tex/f1.bin");
			ProgramCU::ComputeKEY(dog, key, param._dog_threshold, param._edge_threshold, &_featureTexRaw[featureOctLevelIndex], d_featureCount, featureOctLevelIndex,
				keyLocScale, keyLocOffset, d_depthData, GlobalUtil::_SiftDepthMin, GlobalUtil::_SiftDepthMax);
//			cudaDeviceSynchronize();
//            SaveImage(d_depthData,2048*1536,"/home/atom/RobotProjects/SIFTGPUS/siftgpuBF/tex/myd_data2.bin");
//			SaveTexImage(&_featureTexRaw[featureOctLevelIndex],"/home/atom/RobotProjects/SIFTGPUS/siftgpuBF/tex/f2.bin");
//            SaveTexImage(key,"/home/atom/RobotProjects/SIFTGPUS/siftgpuBF/tex/k2.bin");
//            SaveTexImage(dog,"/home/atom/RobotProjects/SIFTGPUS/siftgpuBF/tex/dog2.bin");
//			LOG(INFO)<<"------------- i= "<<i<<" j= "<<j;
//			int tmp[12];
//            cutilSafeCall(cudaMemcpy(tmp, d_featureCount, sizeof(int) * _octave_num * param._dog_level_num, cudaMemcpyDeviceToHost));
//            for(int i=0;i<12;i++) cout<<tmp[i]<<" ";
//            cout<<endl;
//            exit(0);

		}
	}
	cutilSafeCall(cudaMemcpy(_levelFeatureNum, d_featureCount, sizeof(int) * _octave_num * param._dog_level_num, cudaMemcpyDeviceToHost));
	_featureNum = 0;
	for (int i = 0; i < _octave_num * param._dog_level_num; i++) {
		_levelFeatureNum[i] = std::min(_levelFeatureNum[i], _featureTexRaw[i].GetImgWidth() * _featureTexRaw[i].GetImgHeight());
		_featureTexRaw[i].SetImageSize(_levelFeatureNum[i], 1);
		_featureNum += _levelFeatureNum[i];
	}
}

void SiftPyramid::CopyGradientTex()
{
	for (int i = 0, idx = 0; i < _octave_num; i++)
	{
		CuTexImage * got = GetBaseLevel(i + _octave_min, DATA_GRAD) + 1;
		//compute the gradient
		for (int j = 0; j < param._dog_level_num; j++, got++, idx++)
		{
			if (_levelFeatureNum[idx] > 0)	got->CopyToTexture2D();
		}
	}
}

void SiftPyramid::ComputeGradient()
{
	for (int i = _octave_min; i < _octave_min + _octave_num; i++)
	{
		CuTexImage * gus = GetBaseLevel(i) + 1;
		CuTexImage * dog = GetBaseLevel(i, DATA_DOG) + 1;
		CuTexImage * got = GetBaseLevel(i, DATA_GRAD) + 1;

		//compute the gradient
		for (int j = 0; j < param._dog_level_num; j++, gus++, dog++, got++)
		{
			ProgramCU::ComputeDOG(gus, dog, got);
		}
	}
}


void SiftPyramid::GetFeatureOrientations()
{
	CuTexImage * ftex = _featureTexRaw;
	int * count = _levelFeatureNum;
	float sigma, sigma_step = powf(2.0f, 1.0f / param._dog_level_num);

	for (int i = 0; i < _octave_num; i++)
	{
		CuTexImage* got = GetBaseLevel(i + _octave_min, DATA_GRAD) + 1;
		CuTexImage* key = GetBaseLevel(i + _octave_min, DATA_KEYPOINT) + 2;

		for (int j = 0; j < param._dog_level_num; j++, ftex++, count++, got++, key++)
		{
			if (*count <= 0) continue;

			//if(ftex->GetImgWidth() < *count) ftex->InitTexture(*count, 1, 4);

			sigma = param.GetLevelSigma(j + param._level_min + 1);

			ProgramCU::ComputeOrientation(ftex, got, key, sigma, sigma_step);
		}
	}

	ProgramCU::CheckErrorCUDA(__FUNCTION__);
}
//这个金字塔中对于每个图像都有5个备份，分别是DATA_GAUSSIAN，DATA_DOG，DATA_KEYPOINT，DATA_GRAD，DATA_ROT，该函数获取某一组的第一层的某个备份
CuTexImage* SiftPyramid::GetBaseLevel(int octave, int dataName /*= DATA_GAUSSIAN*/)
{
	if (octave <_octave_min || octave > _octave_min + _octave_num) return NULL; //如果组号越界，则返回错误
	int offset = (_pyramid_octave_first + octave - _octave_min) * param._level_num; //_pyramid_octave_first=0
	int num = param._level_num * _pyramid_octave_num;   //6*4=24
	if (dataName == DATA_ROT) dataName = DATA_GRAD;
	return _allPyramid + num * dataName + offset;   //_allPyramid是CuTexImage*类型的，即是一个指针=24*dataName+ x*6
	//_allPyramid总共4*6*5个空间
}

void SiftPyramid::InitPyramid(int w, int h)
{
	int wp, hp, toobig = 0;
	//
	w = TruncateWidth(w);
	////
	if (GlobalUtil::_octave_min_default >= 0)
	{
		wp = w >> _octave_min_default;
		hp = h >> _octave_min_default;
	}
	else
	{
		//can't upsample by more than 8
		_octave_min_default = std::max(-3, _octave_min_default);
		//
		wp = w << (-_octave_min_default);
		hp = h << (-_octave_min_default);
	}
	_octave_min = _octave_min_default;

	while (wp > GlobalUtil::_texMaxDim || hp > GlobalUtil::_texMaxDim)
	{
		_octave_min++;
		wp >>= 1;
		hp >>= 1;
		toobig = 1;
	}

	while (GlobalUtil::_MemCapGPU > 0 && GlobalUtil::_FitMemoryCap && (wp > _pyramid_width || hp > _pyramid_height) &&
		std::max(std::max(wp, hp), std::max(_pyramid_width, _pyramid_height)) > 1024 * sqrt(GlobalUtil::_MemCapGPU / 110.0))
	{
		_octave_min++;
		wp >>= 1;
		hp >>= 1;
		toobig = 2;
	}


	if (toobig && _octave_min > 0)
	{
		std::cout << (toobig == 2 ? "[**SKIP OCTAVES**]:\tExceeding Memory Cap (-nomc)\n" :
			"[**SKIP OCTAVES**]:\tReaching the dimension limit(-maxd)!\n");
	}
	//ResizePyramid(wp, hp);
	if (wp == _pyramid_width && hp == _pyramid_height && _allocated)    //这里wp=2048;hp=1536;_pyramid_width=_pyramid_height=0=_allocated=0
	{
		FitPyramid(wp, hp);
	}
	else if (!_allocated)
	{
		ResizePyramid(wp, hp);  //如果金字塔尚未分配空间，则这里给金字塔分配空间,默认进入此步骤
	}
	else if (wp > _pyramid_width || hp > _pyramid_height)
	{
		ResizePyramid(std::max(wp, _pyramid_width), std::max(hp, _pyramid_height));
		if (wp < _pyramid_width || hp < _pyramid_height)  FitPyramid(wp, hp);
	}
	else
	{
		//try use the pyramid allocated for large image on small input images
		FitPyramid(wp, hp);
	}
}
//给金字塔分配空间
void SiftPyramid::ResizePyramid(int w, int h)
{
	//
	unsigned int totalkb = 0;
	int _octave_num_new, input_sz, i, j;
	//

	if (_pyramid_width == w && _pyramid_height == h && _allocated) return;

	if (w > GlobalUtil::_texMaxDim || h > GlobalUtil::_texMaxDim) return;

	//first octave does not change
	_pyramid_octave_first = 0;

	//compute # of octaves

	input_sz = std::min(w, h);
	_pyramid_width = w;
	_pyramid_height = h;

	//reset to preset parameters

	_octave_num_new = GlobalUtil::_octave_num_default;

	if (_octave_num_new < 1)
	{
		_octave_num_new = (int)floor(log(double(input_sz)) / log(2.0)) - 3;
		if (_octave_num_new < 1) _octave_num_new = 1;
	}

	if (_pyramid_octave_num != _octave_num_new)
	{
		//destroy the original pyramid if the # of octave changes
		if (_octave_num > 0)
		{
			DestroyPerLevelData();
			DestroyPyramidData();
		}
		_pyramid_octave_num = _octave_num_new;
	}

	_octave_num = _pyramid_octave_num;

	int noct = _octave_num;     //4
	int nlev = param._level_num;       //6
 	//	//initialize the pyramid
	if (_allPyramid == NULL)	_allPyramid = new CuTexImage[noct* nlev * DATA_NUM];    //4*6*5个空间

	CuTexImage * gus = GetBaseLevel(_octave_min, DATA_GAUSSIAN);
	CuTexImage * dog = GetBaseLevel(_octave_min, DATA_DOG);
	CuTexImage * got = GetBaseLevel(_octave_min, DATA_GRAD);
	CuTexImage * key = GetBaseLevel(_octave_min, DATA_KEYPOINT);

	////////////there could be "out of memory" happening during the allocation

	for (i = 0; i < noct; i++)
	{
		int wa = ((w + 3) / 4) * 4;

		totalkb += ((nlev * 8 - 19)* (wa * h) * 4 / 1024);
		for (j = 0; j < nlev; j++, gus++, dog++, got++, key++)
		{
			gus->InitTexture(wa, h); //nlev
			if (j == 0)continue;
			dog->InitTexture(wa, h);  //nlev -1
			if (j >= 1 && j < 1 + param._dog_level_num)
			{
				got->InitTexture(wa, h, 2); //2 * nlev - 6
				got->InitTexture2D();
			}
			if (j > 1 && j < nlev - 1)	key->InitTexture(wa, h, 4); // nlev -3 ; 4 * nlev - 12
		}
		w >>= 1;
		h >>= 1;
	}

	totalkb += ResizeFeatureStorage(); // inits _histoPyramidTex & _featureTex & _orientationTex & _descriptorTex

	ProgramCU::CheckErrorCUDA(__FUNCTION__);

	_allocated = true;
}

void SiftPyramid::FitPyramid(int w, int h)
{
	_pyramid_octave_first = 0;
	//
	_octave_num = GlobalUtil::_octave_num_default;

	int _octave_num_max = std::max(1, (int)floor(log(double(std::min(w, h))) / log(2.0)) - 3);

	if (_octave_num < 1 || _octave_num > _octave_num_max)
	{
		_octave_num = _octave_num_max;
	}


	int pw = _pyramid_width >> 1, ph = _pyramid_height >> 1;
	while (_pyramid_octave_first + _octave_num < _pyramid_octave_num &&
		pw >= w && ph >= h)
	{
		_pyramid_octave_first++;
		pw >>= 1;
		ph >>= 1;
	}

	//////////////////
	int nlev = param._level_num;
	CuTexImage * gus = GetBaseLevel(_octave_min, DATA_GAUSSIAN);
	CuTexImage * dog = GetBaseLevel(_octave_min, DATA_DOG);
	CuTexImage * got = GetBaseLevel(_octave_min, DATA_GRAD);
	CuTexImage * key = GetBaseLevel(_octave_min, DATA_KEYPOINT);
	for (int i = 0; i < _octave_num; i++)
	{
		int wa = ((w + 3) / 4) * 4;

		for (int j = 0; j < nlev; j++, gus++, dog++, got++, key++)
		{
			gus->InitTexture(wa, h); //nlev
			if (j == 0)continue;
			dog->InitTexture(wa, h);  //nlev -1
			if (j >= 1 && j < 1 + param._dog_level_num)
			{
				got->InitTexture(wa, h, 2); //2 * nlev - 6
				got->InitTexture2D();
			}
			if (j > 1 && j < nlev - 1)	key->InitTexture(wa, h, 4); // nlev -3 ; 4 * nlev - 12
		}
		w >>= 1;
		h >>= 1;
	}
}

//int SiftPyramid::CheckCudaDevice(int device)
//{
//	return ProgramCU::CheckCudaDevice(device);
//}

void SiftPyramid::EvaluateTimings()
{
	if (!GlobalUtil::_EnableDetailedTimings) {
		std::cout << "Error timings not enabled" << std::endl;
		return;
	}
	else {
		_timer->evaluate(true);
	}
}

//void SiftPyramid::SetLevelFeatureNum(int idx, int fcount)
//{
//	_featureTexRaw[idx].InitTexture(fcount, 1, 4);
//	_levelFeatureNum[idx] = fcount;
//}

void SiftPyramid::SetLevelFinalFeatureNum(int idx, int fcount)
{
	_featureTexFinal[idx].InitTexture(fcount, 1, 4);
	_levelFeatureNum[idx] = fcount;
}

int SiftPyramid::ResizeFeatureStorage()
{
	int totalkb = 0;
	if (_levelFeatureNum == NULL)	_levelFeatureNum = new int[_octave_num * param._dog_level_num];
	std::fill(_levelFeatureNum, _levelFeatureNum + _octave_num * param._dog_level_num, 0);

	cutilSafeCall(cudaMalloc(&d_featureCount, sizeof(int) * _octave_num * param._dog_level_num));
	cutilSafeCall(cudaMalloc(&d_outDescriptorList, sizeof(float) * 128 * GlobalUtil::_MaxLevelFeatureNum));

	//initialize the feature texture
	int idx = 0, n = _octave_num * param._dog_level_num;
	if (_featureTexRaw == NULL)	_featureTexRaw = new CuTexImage[n];
	if (_featureTexFinal == NULL) _featureTexFinal = new CuTexImage[n];
	if (GlobalUtil::_MaxOrientation > 1 && GlobalUtil::_OrientationPack2 == 0 && _orientationTex == NULL)
		_orientationTex = new CuTexImage[n];


	for (int i = 0; i < _octave_num; i++)
	{
		CuTexImage * tex = GetBaseLevel(i + _octave_min);
		int fmax = int(tex->GetImgWidth() * tex->GetImgHeight()*GlobalUtil::_MaxFeaturePercent);
		//
		if (fmax > GlobalUtil::_MaxLevelFeatureNum) fmax = GlobalUtil::_MaxLevelFeatureNum;
		else if (fmax < 32) fmax = 32;	//give it at least a space of 32 feature

		for (int j = 0; j < param._dog_level_num; j++, idx++)
		{
			_featureTexRaw[idx].InitTexture(fmax, 1, 4);
			_featureTexFinal[idx].InitTexture(fmax, 1, 4);
			totalkb += fmax * 16 / 1024;
			//
			if (GlobalUtil::_MaxOrientation > 1 && GlobalUtil::_OrientationPack2 == 0)
			{
				_orientationTex[idx].InitTexture(fmax, 1, 4);
				totalkb += fmax * 16 / 1024;
			}
		}
	}

	return totalkb;
}

void SiftPyramid::CreateGlobalKeyPointList(float4* d_keypoints, const float* d_depthData, unsigned int maxNumKeyPoints)
{
	float os = _octave_min >= 0 ? float(1 << _octave_min) : 1.0f / (1 << (-_octave_min));

	int n = param._dog_level_num*_octave_num;
	unsigned int numKeyOffset = 0;
	//for (int i = 0; i < n; i++) {
	//	if (_levelFeatureNum[i] == 0) continue;

	//	float keyLocScale = os * (1 << (i / param._dog_level_num));
	//	float keyLocOffset = GlobalUtil::_LoweOrigin ? 0 : 0.5f;

	//	ProgramCU::CreateGlobalKeyPointList(&_featureTexFinal[i], d_keypoints + numKeyOffset, keyLocScale, keyLocOffset, d_depthData, 9999);
	//	numKeyOffset += _levelFeatureNum[i];
	//}

	//if needed, eliminate lower level keypoints first
	std::vector<int> numKeysPerLevel(n, 0); int cur = 0;
	const bool bHasMax = maxNumKeyPoints == (unsigned int)-1;
	for (int i = n - 1; i >= 0; i--) {
		if (!bHasMax) numKeysPerLevel[i] = _levelFeatureNum[i];
		else {
			int curnum = std::min(_levelFeatureNum[i], (int)maxNumKeyPoints - cur);
			numKeysPerLevel[i] = curnum;
			cur += curnum;
			if (cur == maxNumKeyPoints) break;
		}
	}

	for (int i = 0; i < n; i++) {
		if (numKeysPerLevel[i] == 0) continue;

		float keyLocScale = os * (1 << (i / param._dog_level_num));
		float keyLocOffset = GlobalUtil::_LoweOrigin ? 0 : 0.5f;

		ProgramCU::CreateGlobalKeyPointList(&_featureTexFinal[i], d_keypoints + numKeyOffset, keyLocScale, keyLocOffset, d_depthData, numKeysPerLevel[i]);
		numKeyOffset += numKeysPerLevel[i];
	}
}

void SiftPyramid::GetFeatureVectorCUDA(unsigned char* d_descriptor, unsigned int maxNumKeyPoints)
{
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->startEvent("ConvertDescriptorToUChar");
	}
	int numDescriptors = (maxNumKeyPoints == (unsigned int)-1) ? _featureNum : std::min(_featureNum, (int)maxNumKeyPoints);
	if (numDescriptors > 0) ProgramCU::ConvertDescriptorToUChar(d_outDescriptorList, numDescriptors * 128, d_descriptor);
	//if (_featureNum > 0) ProgramCU::ConvertDescriptorToUChar(d_outDescriptorList, _featureNum * 128, d_descriptor);
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->endEvent();
	}
}

void SiftPyramid::GetKeyPointsCUDA(float4* d_keypoints, const float* d_depthData, unsigned int maxNumKeyPoints)
{
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->startEvent("CreateGlobalKeyPointList");
	}
	CreateGlobalKeyPointList(d_keypoints, d_depthData, maxNumKeyPoints);
	if (GlobalUtil::_EnableDetailedTimings) {
		_timer->endEvent();
	}
}

void SiftPyramid::SaveTexImage(CuTexImage* TexImage, std::string path) {
    FILE* pFile3;
    if((pFile3=fopen(path.c_str(),"wb"))==NULL){
        printf("can't open the save path CuTexImage.bin");
        exit(0);
    }
    int numBytes=TexImage->GetDataSize();
    std::cout<<"numBytes = "<<numBytes<<std::endl;
    float* floatptr=new float[numBytes];
    TexImage->CopyToHost(floatptr);
    fwrite(floatptr,sizeof(float),numBytes,pFile3);
    fclose(pFile3);
    delete floatptr;
}

void SiftPyramid::LoadTexImage(CuTexImage* TexImage, std::string path) {
    FILE* pFile3;
    if((pFile3=fopen(path.c_str(),"rb"))==NULL){
        printf("can't open the load path CuTexImage.bin");
        exit(0);
    }
    int numBytes=TexImage->GetDataSize();
    std::cout<<"Load numBytes = "<<numBytes<<std::endl;
    float* floatptr=new float[numBytes];
    fread(floatptr,sizeof(float),numBytes,pFile3);
    TexImage->CopyFromHost(floatptr);
    fclose(pFile3);
    delete floatptr;
}

void SiftPyramid::SaveImage(const void* d_image, int size, std::string path) {
    float* depthf2_ptr=new float[size];
    std::cout<<"save float size = "<<size<<std::endl;
    cutilSafeCall(cudaMemcpy(depthf2_ptr, d_image, sizeof(float)*size, cudaMemcpyDeviceToHost));
    FILE* pFile3;
    if((pFile3=fopen(path.c_str(),"wb"))==NULL){
        printf("can't open the depth.bin");
        exit(0);
    }
    fwrite(depthf2_ptr,sizeof(float),size,pFile3);
    fclose(pFile3);
    delete depthf2_ptr;
}
