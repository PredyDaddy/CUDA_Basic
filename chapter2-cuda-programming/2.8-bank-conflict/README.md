## 6. Bank Conflict

使用共享内存的时候可能会遇到的问题

## 6.1 Bank Conflict 
1. 共享内存的Bank组织

共享内存被组织成若干bank（例如，32或64），每个bank可以在一个时钟周期内服务一个内存访问。因此，理想情况下，如果32个线程（一个warp）访问32个不同的bank中的32个不同的字（word），则所有这些访问可以在一个时钟周期内完成。

2. 什么是Bank Conflict？

当多个线程在同一个时钟周期中访问同一个bank中的不同字时，就会发生bank conflict。这会导致访问被序列化，增加总的访问时间。例如，如果两个线程访问同一个bank中的两个不同字，则需要两个时钟周期来服务这两个访问。

3. 如何避免Bank 

避免bank conflict的一种策略是通过确保线程访问的内存地址分布在不同的bank上。这可以通过合理的数据布局和访问模式来实现。例如，在矩阵乘法中，可以通过使用共享内存的块来重新排列数据访问模式来减少bank conflicts。

总结
理解和避免bank conflicts是优化CUDA程序的一个重要方面，特别是当使用共享内存来存储频繁访问的数据时。你可以通过修改你的数据访问模式和数据结构来尽量减少bank conflicts，从而提高程序的性能。



## 6.2 案例

最简单的理解就是之前是[ty][tx] =====> [tx][ty] 

```cpp
/* 
    使用shared memory把计算一个tile所需要的数据分块存储到访问速度快的memory中
*/
__global__ void MatmulSharedStaticConflictKernel(float *M_device, float *N_device, float *P_device, int width){
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE];
    /* 
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0.0;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    /* 对于每一个P的元素，我们只需要循环遍历width / tile_width 次就okay了，这里有点绕，画图理解一下*/
    for (int m = 0; m < width / BLOCKSIZE; m ++) {
        /* 这里为了实现bank conflict, 把tx与tx的顺序颠倒，同时索引也改变了*/
        M_deviceShared[tx][ty] = M_device[x * width + (m * BLOCKSIZE + ty)];
        N_deviceShared[tx][ty] = M_device[(m * BLOCKSIZE + tx)* width + y];
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; k ++) {
            P_element += M_deviceShared[tx][k] * N_deviceShared[k][ty];
        }
        __syncthreads();
    }

    /* 列优先 */
    P_device[x * width + y] = P_element;
}
```
