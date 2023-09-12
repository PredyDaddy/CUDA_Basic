## 7. Stream and Event 
CUDA Stream（CUDA流）是一种在NVIDIA的GPU编程环境中使用的并行执行机制。它允许您将不同的GPU任务划分为多个独立的流，并在GPU上并行执行这些流中的任务。CUDA流的主要目的是提高GPU的利用率和性能，特别是在处理多个并发任务时。

### 7.1 **启动顺序和重叠操作**：
   - 同一个流的启动顺序通常与设计时的顺序相同。
   - 目标是实现操作的重叠，以提高效率和性能。

### 7.2 **memcpy和多流的比较**：
   - memcpy在单位时间内只能启动一个操作。如果资源被占满，多流和单流的性能差异不大。

### 7.3 **数据传输和核函数重叠**：
   - 通过优化数据传输，如H2D（Host to Device）和D2H（Device to Host）以及核函数的启动，可以实现高效的重叠。
   - 可以逐步搬移数据，执行核函数，然后再搬移回数据，以实现高效率。
   - 这种优化可以将性能提高到单流默认流的三倍速，前提是资源没有被占满。

### 7.4 **流的启动顺序和延迟**：
   - 流的启动是有顺序和延迟的，不同流之间的操作不一定同时开始。
   - 这个顺序和延迟是需要考虑的因素，特别是在依赖关系重要的情况下。

### 7.5 **cudaMalloc和页锁定内存**：
   - `cudaMalloc`分配的是页锁定内存，也称为固定内存。
   - 页锁定内存不会被分页到磁盘，因此对于GPU访问非常高效。
   - 在某些情况下，人们更喜欢直接使用`cudaMallocHost`来分配页锁定内存，因为它更容易使用。
   - 像之前矩阵乘法并没有直接cudaMallocHost是因为使用的是左边的而不是右边的页锁定内存
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/662f6e9d3f654653bffea12cc6b5fd18.png)

### 7.6 **两个比较直观的例子**
kernel1和kernel2是不相关的
```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 在stream1中启动操作
kernel1<<<grid1, block1, 0, stream1>>>();
cudaMemcpyAsync(devPtr1, hostPtr1, size1, cudaMemcpyHostToDevice, stream1);

// 在stream2中启动另一组操作
kernel2<<<grid2, block2, 0, stream2>>>();
cudaMemcpyAsync(devPtr2, hostPtr2, size2, cudaMemcpyHostToDevice, stream2);
```
kernel1和第一个cudaMemcpyAsync调用是在stream1中执行的。

同时，kernel2和第二个cudaMemcpyAsync调用是在stream2中执行的。

**kernel2需要kernel1的输出作为输入**
```cpp
cudaStream_t stream1, stream2;
cudaEvent_t event;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaEventCreate(&event);

// 在stream1中启动操作
kernel1<<<grid1, block1, 0, stream1>>>();
cudaMemcpyAsync(devPtr1, hostPtr1, size1, cudaMemcpyHostToDevice, stream1);

// 记录stream1中的事件
cudaEventRecord(event, stream1);

// 在stream2中启动另一组操作，但只有在event被记录后才开始
cudaStreamWaitEvent(stream2, event, 0);
kernel2<<<grid2, block2, 0, stream2>>>();
cudaMemcpyAsync(devPtr2, hostPtr2, size2, cudaMemcpyHostToDevice, stream2);
```

### 7.7 我们自己的代码：

- 在多流实验中，可以人为地创建一个没有充分利用计算资源的操作，以便重叠核函数执行和内存访问。
- 多流的优势不仅仅是核函数的重叠，更是核函数与内存访问的重叠。
- 通过优化，可以减少延迟并将不依赖于彼此的操作放在一起，以提高性能。

总之，CUDA流是一种重要的并行计算概念，它可以用于提高GPU应用程序的性能。通过合理的流管理和数据传输优化，可以实现更好的并行性和效率，以充分利用GPU的计算资源。

**代码实战**
先检查设备是否支持重叠操作, 支持重叠才可以进行后面的操作，来自main.cpp
```cpp
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 需要先确认自己的GPU是否支持overlap计算
    if (!prop.deviceOverlap) {
        LOG("device does not support overlap");
    } else {
        LOG("device supports overlap");
    }
```

在main.cpp中调用stream.cu的kernel function
```cpp
SleepSingleStream(src_host, tar_host, width, blockSize, taskCnt);
SleepMultiStream(src_host, tar_host, width, blockSize, taskCnt);
```

**核函数 __global__ void SleepKernel** 这个内核函数接收一个num_cycles参数，并使用一个循环来“睡眠”一段时间，通过不断检查当前时钟周期直到达到num_cycles。这里设置了三个MAX_ITER常量在代码里可以拿来测试
```cpp
__global__ void SleepKernel(
    int64_t num_cycles)
{
    int64_t cycles = 0;
    int64_t start = clock64();
    while(cycles < num_cycles) {
        cycles = clock64() - start;
    }
}
```

这里我写了一个显示的流, 其实跟默认流也没什么区别的, 唯一区别是默认流用cudaDeviceSynchronize()而显示的流用cudaStreamSynchronize(), 最后用完了需要把流释放了，这里单流多流都是一次cudaMalloc(), cudaMemcpy然后执行n次kernel, 代码如下
```cpp
void MySleepSingleStream(
    float* src_host, float* tar_host, 
    int width, int blockSize, 
    int count) 
{
    int size = width * width * sizeof(float);

    float *src_device;
    float *tar_device;

    CUDA_CHECK(cudaMalloc((void**)&src_device, size));
    CUDA_CHECK(cudaMalloc((void**)&tar_device, size));

    // 创建一个新的CUDA流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < count ; i++) {
        for (int j = 0; j < 1; j ++) 
            // 使用创建的流进行异步内存拷贝
            CUDA_CHECK(cudaMemcpyAsync(src_device, src_host, size, cudaMemcpyHostToDevice, stream));

        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(width / blockSize, width / blockSize);

        // 使用创建的流来启动内核
        SleepKernel <<<dimGrid, dimBlock, 0, stream>>> (MAX_ITER);
        // 使用创建的流进行异步内存拷贝
        CUDA_CHECK(cudaMemcpyAsync(src_host, src_device, size, cudaMemcpyDeviceToHost, stream));
    }

    // 同步创建的流
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 销毁创建的流
    CUDA_CHECK(cudaStreamDestroy(stream));

    cudaFree(tar_device);
    cudaFree(src_device);
}
```

多流的代码也好理解, 这里可以同时执行多个内存拷贝和内核执行。我们可以看到流优化了三个地方，**H2D, kernel, D2H**, 对应的图![](https://img-blog.csdnimg.cn/b70ec29675d54ef690f12e4e6350aaab.png)
```cpp
/* n stream，处理一次memcpy，以及n个kernel */
void SleepMultiStream(
    float* src_host, float* tar_host,
    int width, int blockSize, 
    int count) 
{
    int size = width * width * sizeof(float);

    float *src_device;
    float *tar_device;

    CUDA_CHECK(cudaMalloc((void**)&src_device, size));
    CUDA_CHECK(cudaMalloc((void**)&tar_device, size));


    /* 先把所需要的stream创建出来 */
    cudaStream_t stream[count];
    for (int i = 0; i < count ; i++) {
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }

    for (int i = 0; i < count ; i++) {
        for (int j = 0; j < 1; j ++) 
            CUDA_CHECK(cudaMemcpyAsync(src_device, src_host, size, cudaMemcpyHostToDevice, stream[i]));
        dim3 dimBlock(blockSize, blockSize);
        dim3 dimGrid(width / blockSize, width / blockSize);

        /* 这里面我们把参数写全了 <<<dimGrid, dimBlock, sMemSize, stream>>> */
        SleepKernel <<<dimGrid, dimBlock, 0, stream[i]>>> (MAX_ITER);
        CUDA_CHECK(cudaMemcpyAsync(src_host, src_device, size, cudaMemcpyDeviceToHost, stream[i]));
    }


    CUDA_CHECK(cudaDeviceSynchronize());


    cudaFree(tar_device);
    cudaFree(src_device);

    for (int i = 0; i < count ; i++) {
        // 使用完了以后不要忘记释放
        cudaStreamDestroy(stream[i]);
    }

}
```

最后效果如下
```bash
device supports overlap
Input size is 1048576
sleep <<<(64,64), (16,16)>>>,  1 stream,  1 memcpy,  4 kernel uses 5.000320 ms
sleep <<<(64,64), (16,16)>>>,  4 stream,  1 memcpy,  4 kernel uses 3.653024 ms
```
