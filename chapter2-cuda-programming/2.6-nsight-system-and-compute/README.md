## 4. Nsight System 和 Nsight Compute
我自己是使用windows端SSH连接远程服务器(Ubuntu 20.04), 然后访问服务器上的容器, 所以就没有办法直接使用, 这个时候就可以在容器用指令生成report, 然后下载下来用windows打开


### Nsight Systems

Nsight Systems偏向于可视化整个应用程序的性能分析，它关注多个系统层面的性能指标，包括但不限于：

- PCIe带宽
- DRAM带宽
- SM Warp占用率
- 核函数（Kernel）的调度和执行时间
- 多个Stream和队列之间的调度信息
- CPU和GPU间的数据传输时间
- 整体应用程序的时间消耗排序

这样全面的信息可以帮助开发者从宏观的角度理解应用程序的性能瓶颈，并据此进行相应的优化。

### 4.1 Nsight Compute

与Nsight Systems相比，Nsight Compute则更加专注于单个CUDA核函数的性能分析。它能提供非常细致的信息，例如：

- SM中的计算吞吐量
- L1和L2缓存的数据传输吞吐量
- DRAM数据传输吞吐量
- 核函数是计算密集型还是内存访问密集型
- Roofline model分析
- L1缓存的命中率和失效率
- 核函数中各代码部分的延迟
- 内存访问的调度信息

这些信息可以让开发者针对特定的CUDA核函数进行深度优化。

### 4.2 区别和应用场景

总结一下，两者的主要区别在于它们的焦点和应用场景：

- **Nsight Systems**：更多用于初步诊断和宏观优化，当你需要了解整个系统或应用程序的性能瓶颈时，这是一个很好的起点。
- **Nsight Compute**：当你需要深入到特定的CUDA核函数进行细粒度的分析和优化时，这是一个更适合的工具。

通常，开发者会先使用Nsight Systems进行初步的性能分析，找出可能存在的瓶颈，然后再针对这些瓶颈使用Nsight Compute进行深入的优化。这两个工具往往是相互补充的。

```bash
# 打开容器的指令加一个--cap-add=SYS_ADMIN才能跑nsight compute
docker run --cap-add=SYS_ADMIN --gpus all -it --name easonbob_trt -v $(pwd):/app easonbob/my_trt-tensorrt:nsight_system

# nsight system
nsys profile --trace=cuda,nvtx -o nsight_systems_report ./trt-cuda

# nsight compute
nv-nsight-cu-cli -o nsight_compute_report ./trt-cuda
```

然后下载下来就可以直接在最新版本的nsight system和nsight compute里面打开(我自己测试过), 使用File->open, 这里下载的是windows的版本就好, 这里也附上NVIDIA的Download Center(https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2023-3)

![在这里插入图片描述](https://img-blog.csdnimg.cn/cee90e5889c54817bd17894edc480f5b.png)![在这里插入图片描述](https://img-blog.csdnimg.cn/c874094278014e828844138ceb3dcd11.png)


![在这里插入图片描述](https://img-blog.csdnimg.cn/3884575cae1f441fa2364eb0fea2ee79.png)