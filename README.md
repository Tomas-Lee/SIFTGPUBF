# SIFTGPUBF
基于wuchangchang的SIFTGPU代码和BundleFusion作者对其进行修改封装的代码，为便于使用，便于模块化处理，本人做了封装和精简，可以适配目前最新CUDA环境的SIFTGPU代码，并且代码非常精简，不含其他依赖和冗余库。

# 为何特征提取要使用depth图
1. SIFT特征提取，检测关键点ProgramCU::ComputeKEY()函数中，深度信息仅用于过滤掉那些深度值超出值域的关键点。
2. 在获取关键点位置的时候，深度图用于和rgb信息一起返回该关键点的深度值。仅此而已。可以删除。

# 遗留的问题
- SiftCameraParam.h文件注释掉了__align__(16)代码，这是因为不然会报错，原因待查
- main函数仍然需要cu后缀，没能把分离编译弄完
- 如果该程序连续多次运行，则SIFTImageManager.h文件166行的cudaMemcpy会报错，显示Runtime api error77 an illegal memory access was encountered， 原因待查
- SIFTImageManager需要把实现跟声明分开，时间不够，暂时没有做，而且此类需要更加完善。
