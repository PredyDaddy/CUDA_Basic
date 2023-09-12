## 5 共享内存

```bash
Input size is 4096 x 4096
matmul in gpu(warmup)                                        uses 102.768669 ms
matmul in gpu(without shared memory)<<<256, 16>>>            uses 101.848831 ms
matmul in gpu(with shared memory(static))<<<256, 16>>>       uses 63.545631 ms
```

在之前的案例中, 我们把M, N两个矩阵通过cudaMalloc()开辟然后cudaMemcpy()把数据从Host搬到Device上, 这里其实用的是Global Memory, 从图上可以看到的是Global Memory其实很慢, 因为在图中离Threads越近, 他会有一个更高的带宽, 所以在CUDA编程中我们需要更多的去使用L1 Cache和Share Memory。**共享内存是每个线程块（block）专用的**。
![在这里插入图片描述](https://img-blog.csdnimg.cn/bbb7c9c621004f0a855748371eb16460.png)


## 5.1 MatmulSharedStaticKernel()
静态共享内存, 这里的设计是给每一个block设置跟线程数同等大小的共享内存, 最后的P_element跟之前一样还是把全部的block里面计算的都加起来, 这里的思想跟之前一样。 唯一的区别就是每一个block访问的内存。

每一个block中, 线程先是从Global Memory(M_device, N_device)中拿到对应的内存去填上共享内存, 全部填完了(同步)之后再从共享内存依次取出来去做对应的计算。

__syncthreads();  这个是跟共享内存绑定的, 这里出现两次, 第一次是每个线程块（block）中的线程首先将一小块（tile）的数据从全局内存（M_device 和 N_device）复制到共享内存。第二次是等待全部计算完成。

**M的共享内存往右边遍历**, 拿的是行, 这里可以想象成是为了拿到每一行, 也就是在y++的情况下怎么拿到每一行的每一个元素, 用tx和y
```cpp
M_deviceShared[ty][tx] = M_device[y * width + (m * BLOCKSIZE + tx)];
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/8d6823911e6c4da09ae63557b3c51f9c.jpeg)

**M的共享内存往下边遍历**, 拿的是列, 这里可以想象成是为了拿到每一列, 也就是在x++的情况下拿到每一列的元素, 用tx和y
```cpp
N_deviceShared[ty][tx] = N_device[(m * BLOCKSIZE + ty)* width + x];
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/1a755e2d388a44cb8143cf276e060e19.jpeg)


```cpp
__global__ void MatmulSharedStaticKernel(float *M_device, float *N_device, float *P_device, int width){
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];
    /* 
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0.0;

    // 这里出现的是block里面的索引, 因为共享内存是block专属的东西
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了，这里有点绕，画图理解一下*/
    for (int m = 0; m < width / BLOCKSIZE; m ++) {
        M_deviceShared[ty][tx] = M_device[y * width + (m * BLOCKSIZE + tx)];
        N_deviceShared[ty][tx] = N_device[(m * BLOCKSIZE + ty)* width + x];
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k ++) {
            P_element += M_deviceShared[ty][k] * N_deviceShared[k][tx];
        }
        __syncthreads();
    }

    P_device[y * width + x] = P_element;
}
```

P_device的结果是全部m加起来的结果


## 5.2 动态共享内存

一般没有什么特殊需求就不要用共享动态内存了，也未必见得会快多少 By 韩导

```cpp
__global__ void MatmulSharedDynamicKernel(float *M_device, float *N_device, float *P_device, int width, int blockSize){
    /* 
        声明动态共享变量的时候需要加extern，同时需要是一维的 
        注意这里有个坑, 不能够像这样定义： 
            __shared__ float M_deviceShared[];
            __shared__ float N_deviceShared[];
        因为在cuda中定义动态共享变量的话，无论定义多少个他们的地址都是一样的。
        所以如果想要像上面这样使用的话，需要用两个指针分别指向shared memory的不同位置才行
    */

    extern __shared__ float deviceShared[];
    int stride = blockSize * blockSize;
    /* 
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * blockSize + threadIdx.x;
    int y = blockIdx.y * blockSize + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了 */
    for (int m = 0; m < width / blockSize; m ++) {
        deviceShared[ty * blockSize + tx] = M_device[y * width + (m * blockSize + tx)];
        deviceShared[stride + (ty * blockSize + tx)] = N_device[(m * blockSize + ty)* width + x];
        __syncthreads();

        for (int k = 0; k < blockSize; k ++) {
            P_element += deviceShared[ty * blockSize + k] * deviceShared[stride + (k * blockSize + tx)];
        }
        __syncthreads();
    }

    if (y < width && x < width) {
        P_device[y * width + x] = P_element;
    }
}
```