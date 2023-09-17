
## 1. CUDA中的grid和block基本的理解

![在这里插入图片描述](https://img-blog.csdnimg.cn/0cf6284070f44f948deac812f3207b4f.png)


1. **Kernel**: Kernel不是CPU，而是在GPU上运行的特殊函数。你可以把Kernel想象成GPU上并行执行的任务。当你从主机（CPU）调用Kernel时，它在GPU上启动，并在许多线程上并行运行。

2. **Grid**: 当你启动Kernel时，你会定义一个网格（grid）。网格是一维、二维或三维的，代表了block的集合。

3. **Block**: 每个block内部包含了许多线程。block也可以是一维、二维或三维的。

4. **Thread**: 每个线程是Kernel的单个执行实例。在一个block中的所有线程可以共享一些资源，并能够相互通信。

你正确地指出，grid、block和thread这些概念在硬件级别上并没有直接对应的实体，它们是抽象的概念，用于组织和管理GPU上的并行执行。然而，GPU硬件是专门设计来支持这种并行计算模型的，所以虽然线程在物理硬件上可能不是独立存在的，但是它们通过硬件架构和调度机制得到了有效的支持。

另外，对于线程的管理和调度，GPU硬件有特定的线程调度单元，如NVIDIA的warp概念。线程被组织成更小的集合，称为warps（在NVIDIA硬件上），并且这些warps被调度到硬件上以供执行。

所以，虽然这些概念是逻辑和抽象的，但它们与硬件的实际执行密切相关，并由硬件特性和架构直接支持。

一般来说：

• 一个kernel对应一个grid

• 一个grid可以有多个block，一维~三维

• 一个block可以有多个thread，一维~三维

### 1.1. 1D traverse

![在这里插入图片描述](https://img-blog.csdnimg.cn/3f7fd337f91242ee8e277b6c697f4ed1.png)

```cpp
void print_one_dim(){
    int inputSize = 8;
    int blockDim = 4;
    int gridDim = inputSize / blockDim; // 2

    // 定义block和grid的维度
    dim3 block(blockDim);  // 说明一个block有多少个threads
    dim3 grid(gridDim);    // 说明一个grid里面有多少个block 

    /* 这里建议大家吧每一函数都试一遍*/
    print_idx_kernel<<<grid, block>>>();
    // print_dim_kernel<<<grid, block>>>();
    // print_thread_idx_per_block_kernel<<<grid, block>>>();
    // print_thread_idx_per_grid_kernel<<<grid, block>>>();

    cudaDeviceSynchronize();
}
```
我觉得重点在这两行

1. **`dim3 block(blockDim);`**: 这一行创建了一个三维向量`block`，用来定义每个block的大小。在这个例子中，`blockDim`是一个整数值4，所以每个block包含4个线程。`dim3`数据类型是CUDA中的一个特殊数据类型，用于表示三维向量。在这个情况下，你传递了一个整数值，所以`block`的其余维度将被默认设置为1。这意味着你将有一个包含4个线程的一维block。

2. **`dim3 grid(gridDim);`**: 这一行创建了一个三维向量`grid`，用来定义grid的大小。`gridDim`的计算基于输入大小（`inputSize`）和每个block的大小（`blockDim`）。在这个例子中，`inputSize`是8，`blockDim`是4，所以`gridDim`会是2。这意味着整个grid将包含2个block。与`block`一样，你传递了一个整数值给`grid`，所以其余维度将被默认设置为1，得到一个一维grid。

总体来说，这两行代码定义了内核的执行配置，将整个计算空间划分为2个block，每个block包含4个线程。你可以想象这个配置如下：

- **Block 0**: 线程0, 线程1, 线程2, 线程3
- **Block 1**: 线程4, 线程5, 线程6, 线程7

然后，当你调用内核时，这些线程将被用来执行你的代码。每个线程可以通过其线程索引和block索引来访问自己在整个grid中的唯一位置。这些索引用于确定每个线程应处理的数据部分。

```bash
block idx:   1, thread idx in block:   0, thread idx:   4
block idx:   1, thread idx in block:   1, thread idx:   5
block idx:   1, thread idx in block:   2, thread idx:   6
block idx:   1, thread idx in block:   3, thread idx:   7
block idx:   0, thread idx in block:   0, thread idx:   0
block idx:   0, thread idx in block:   1, thread idx:   1
block idx:   0, thread idx in block:   2, thread idx:   2
block idx:   0, thread idx in block:   3, thread idx:   3
```



### 1.2 2D打印
```cpp
// 8个线程被分成了两个
void print_two_dim(){
    int inputWidth = 4;

    int blockDim = 2;  
    int gridDim = inputWidth / blockDim;

    dim3 block(blockDim, blockDim);
    dim3 grid(gridDim, gridDim);

    /* 这里建议大家吧每一函数都试一遍*/
    // print_idx_kernel<<<grid, block>>>();
    // print_dim_kernel<<<grid, block>>>();
    // print_thread_idx_per_block_kernel<<<grid, block>>>();
    print_thread_idx_per_grid_kernel<<<grid, block>>>();

    cudaDeviceSynchronize();
}
```

1. **`dim3 block(blockDim, blockDim);`**: 这里创建了一个二维的`block`，每个维度的大小都是`blockDim`，在这个例子中是2。因此，每个block都是2x2的，包含4个线程。由于`dim3`定义了一个三维向量，没有指定的第三维度会默认为1。

2. **`dim3 grid(gridDim, gridDim);`**: 同样，`grid`也被定义为二维的，每个维度的大小都是`gridDim`。由于`inputWidth`是4，并且`blockDim`是2，所以`gridDim`会是2。因此，整个grid是2x2的，包括4个block。第三维度同样默认为1。

因此，整个执行配置定义了2x2的grid，其中包括4个2x2的block，总共16个线程。你可以将整个grid可视化如下：

- **Block (0,0)**:
  - 线程(0,0), 线程(0,1)
  - 线程(1,0), 线程(1,1)

- **Block (0,1)**:
  - 线程(2,0), 线程(2,1)
  - 线程(3,0), 线程(3,1)

- **Block (1,0)**:
  - 线程(4,0), 线程(4,1)
  - 线程(5,0), 线程(5,1)

- **Block (1,1)**:
  - 线程(6,0), 线程(6,1)
  - 线程(7,0), 线程(7,1)

输出中的“block idx”是整个grid中block的线性索引，而“thread idx in block”是block内线程的线性索引。最后的“thread idx”是整个grid中线程的线性索引。

请注意，执行的顺序仍然是不确定的。你看到的输出顺序可能在不同的运行或不同的硬件上有所不同。

```bash
block idx:   3, thread idx in block:   0, thread idx:  12
block idx:   3, thread idx in block:   1, thread idx:  13
block idx:   3, thread idx in block:   2, thread idx:  14
block idx:   3, thread idx in block:   3, thread idx:  15
block idx:   2, thread idx in block:   0, thread idx:   8
block idx:   2, thread idx in block:   1, thread idx:   9
block idx:   2, thread idx in block:   2, thread idx:  10
block idx:   2, thread idx in block:   3, thread idx:  11
block idx:   1, thread idx in block:   0, thread idx:   4
block idx:   1, thread idx in block:   1, thread idx:   5
block idx:   1, thread idx in block:   2, thread idx:   6
block idx:   1, thread idx in block:   3, thread idx:   7
block idx:   0, thread idx in block:   0, thread idx:   0
block idx:   0, thread idx in block:   1, thread idx:   1
block idx:   0, thread idx in block:   2, thread idx:   2
block idx:   0, thread idx in block:   3, thread idx:   3
```

### 1.3 3D grid
```cpp
dim3 block(3, 4, 2);
dim3 grid(2, 2, 2);
```

1. **Block布局** (`dim3 block(3, 4, 2)`):
   - 这定义了每个block的大小为3x4x2，所以每个block包含24个线程。
   - 你可以将block视为三维数组，其中`x`方向有3个元素，`y`方向有4个元素，`z`方向有2个元素。

2. **Grid布局** (`dim3 grid(2, 2, 2)`):
   - 这定义了grid的大小为2x2x2，所以整个grid包含8个block。
   - 你可以将grid视为三维数组，其中`x`方向有2个元素，`y`方向有2个元素，`z`方向有2个元素。
   - 由于每个block包括24个线程，所以整个grid将包括192个线程。

整体布局可以视为8个3x4x2的block，排列为2x2x2的grid。

如果我们想用文字来表示整个结构，可能会是这样的：

- **Grid[0][0][0]**:
   - Block(3, 4, 2) -- 24个线程
- **Grid[0][0][1]**:
   - Block(3, 4, 2) -- 24个线程
- **Grid[0][1][0]**:
   - Block(3, 4, 2) -- 24个线程
- **Grid[0][1][1]**:
   - Block(3, 4, 2) -- 24个线程
- **Grid[1][0][0]**:
   - Block(3, 4, 2) -- 24个线程
- **Grid[1][0][1]**:
   - Block(3, 4, 2) -- 24个线程
- **Grid[1][1][0]**:
   - Block(3, 4, 2) -- 24个线程
- **Grid[1][1][1]**:
   - Block(3, 4, 2) -- 24个线程

这种三维结构允许在物理空间中进行非常自然的映射，尤其是当你的问题本身就具有三维的特性时。例如，在处理三维物理模拟或体素数据时，这种映射可能非常有用。

#### 5. 通过维度打印出来对应的thread

![在这里插入图片描述](https://img-blog.csdnimg.cn/e8e014f01d7343f885bc7d1b4f368cf0.png)


**比较推荐的打印方式**

```cpp
__global__ void print_cord_kernel(){
    int index = threadIdx.z * blockDim.x * blockDim.y + \
              threadIdx.y * blockDim.x + \
              threadIdx.x;

    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;

    printf("block idx: (%3d, %3d, %3d), thread idx: %3d, cord: (%3d, %3d)\n",
         blockIdx.z, blockIdx.y, blockIdx.x,
         index, x, y);
}
```
index是线程索引的问题，首先，考虑z维度。对于每一层z，都有blockDim.x * blockDim.y个线程。所以threadIdx.z乘以该数量给出了前面层中的线程总数，**从图上看也就是越过了多少个方块**

然后，考虑y维度。对于每一行y，都有blockDim.x个线程。所以threadIdx.y乘以该数量给出了当前层中前面行的线程数，**也就是在当前方块的xy面我们走了几个y, 几行**

最后加上thread x完成索引的坐标

```cpp
void print_cord(){
    int inputWidth = 4;

    int blockDim = 2;
    int gridDim = inputWidth / blockDim;

    dim3 block(blockDim, blockDim);
    dim3 grid(gridDim, gridDim);

    print_cord_kernel<<<grid, block>>>();
    // print_thread_idx_per_grid_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
}
```

```bash
block idx: (  0,   1,   0), thread idx:   0, cord: (  0,   2)
block idx: (  0,   1,   0), thread idx:   1, cord: (  1,   2)
block idx: (  0,   1,   0), thread idx:   2, cord: (  0,   3)
block idx: (  0,   1,   0), thread idx:   3, cord: (  1,   3)
block idx: (  0,   1,   1), thread idx:   0, cord: (  2,   2)
block idx: (  0,   1,   1), thread idx:   1, cord: (  3,   2)
block idx: (  0,   1,   1), thread idx:   2, cord: (  2,   3)
block idx: (  0,   1,   1), thread idx:   3, cord: (  3,   3)
block idx: (  0,   0,   1), thread idx:   0, cord: (  2,   0)
block idx: (  0,   0,   1), thread idx:   1, cord: (  3,   0)
block idx: (  0,   0,   1), thread idx:   2, cord: (  2,   1)
block idx: (  0,   0,   1), thread idx:   3, cord: (  3,   1)
block idx: (  0,   0,   0), thread idx:   0, cord: (  0,   0)
block idx: (  0,   0,   0), thread idx:   1, cord: (  1,   0)
block idx: (  0,   0,   0), thread idx:   2, cord: (  0,   1)
block idx: (  0,   0,   0), thread idx:   3, cord: (  1,   1)
```

**跟之前2D的一样， 同样看起来有点乱，是因为是异步执行的**

### 1.4 最后看一个多个grid的案例
```cpp
void print_coordinates() {
    dim3 block(3, 4, 2);
    dim3 grid(2, 2, 2);

    print_cord_kernel<<<grid, block>>>();

    cudaDeviceSynchronize(); // 确保内核完成后才继续执行主机代码
}
```

```bash
block idx: (  0,   1,   0), thread idx:   0, cord: (  0,   4)
block idx: (  0,   1,   0), thread idx:   1, cord: (  1,   4)
block idx: (  0,   1,   0), thread idx:   2, cord: (  2,   4)
block idx: (  0,   1,   0), thread idx:   3, cord: (  0,   5)
block idx: (  0,   1,   0), thread idx:   4, cord: (  1,   5)
block idx: (  0,   1,   0), thread idx:   5, cord: (  2,   5)
block idx: (  0,   1,   0), thread idx:   6, cord: (  0,   6)
block idx: (  0,   1,   0), thread idx:   7, cord: (  1,   6)
block idx: (  0,   1,   0), thread idx:   8, cord: (  2,   6)
block idx: (  0,   1,   0), thread idx:   9, cord: (  0,   7)
block idx: (  0,   1,   0), thread idx:  10, cord: (  1,   7)
block idx: (  0,   1,   0), thread idx:  11, cord: (  2,   7)
block idx: (  0,   1,   0), thread idx:  12, cord: (  0,   4)
block idx: (  0,   1,   0), thread idx:  13, cord: (  1,   4)
block idx: (  0,   1,   0), thread idx:  14, cord: (  2,   4)
block idx: (  0,   1,   0), thread idx:  15, cord: (  0,   5)
block idx: (  0,   1,   0), thread idx:  16, cord: (  1,   5)
block idx: (  0,   1,   0), thread idx:  17, cord: (  2,   5)
block idx: (  0,   1,   0), thread idx:  18, cord: (  0,   6)
block idx: (  0,   1,   0), thread idx:  19, cord: (  1,   6)
block idx: (  0,   1,   0), thread idx:  20, cord: (  2,   6)
block idx: (  0,   1,   0), thread idx:  21, cord: (  0,   7)
block idx: (  0,   1,   0), thread idx:  22, cord: (  1,   7)
block idx: (  0,   1,   0), thread idx:  23, cord: (  2,   7)
block idx: (  1,   1,   1), thread idx:   0, cord: (  3,   4)
block idx: (  1,   1,   1), thread idx:   1, cord: (  4,   4)
block idx: (  1,   1,   1), thread idx:   2, cord: (  5,   4)
block idx: (  1,   1,   1), thread idx:   3, cord: (  3,   5)
block idx: (  1,   1,   1), thread idx:   4, cord: (  4,   5)
block idx: (  1,   1,   1), thread idx:   5, cord: (  5,   5)
block idx: (  1,   1,   1), thread idx:   6, cord: (  3,   6)
block idx: (  1,   1,   1), thread idx:   7, cord: (  4,   6)
block idx: (  1,   1,   1), thread idx:   8, cord: (  5,   6)
block idx: (  1,   1,   1), thread idx:   9, cord: (  3,   7)
block idx: (  1,   1,   1), thread idx:  10, cord: (  4,   7)
block idx: (  1,   1,   1), thread idx:  11, cord: (  5,   7)
block idx: (  1,   1,   1), thread idx:  12, cord: (  3,   4)
block idx: (  1,   1,   1), thread idx:  13, cord: (  4,   4)
block idx: (  1,   1,   1), thread idx:  14, cord: (  5,   4)
block idx: (  1,   1,   1), thread idx:  15, cord: (  3,   5)
block idx: (  1,   1,   1), thread idx:  16, cord: (  4,   5)
block idx: (  1,   1,   1), thread idx:  17, cord: (  5,   5)
block idx: (  1,   1,   1), thread idx:  18, cord: (  3,   6)
block idx: (  1,   1,   1), thread idx:  19, cord: (  4,   6)
block idx: (  1,   1,   1), thread idx:  20, cord: (  5,   6)
block idx: (  1,   1,   1), thread idx:  21, cord: (  3,   7)
block idx: (  1,   1,   1), thread idx:  22, cord: (  4,   7)
block idx: (  1,   1,   1), thread idx:  23, cord: (  5,   7)
block idx: (  0,   1,   1), thread idx:   0, cord: (  3,   4)
block idx: (  0,   1,   1), thread idx:   1, cord: (  4,   4)
block idx: (  0,   1,   1), thread idx:   2, cord: (  5,   4)
block idx: (  0,   1,   1), thread idx:   3, cord: (  3,   5)
block idx: (  0,   1,   1), thread idx:   4, cord: (  4,   5)
block idx: (  0,   1,   1), thread idx:   5, cord: (  5,   5)
block idx: (  0,   1,   1), thread idx:   6, cord: (  3,   6)
block idx: (  0,   1,   1), thread idx:   7, cord: (  4,   6)
block idx: (  0,   1,   1), thread idx:   8, cord: (  5,   6)
block idx: (  0,   1,   1), thread idx:   9, cord: (  3,   7)
block idx: (  0,   1,   1), thread idx:  10, cord: (  4,   7)
block idx: (  0,   1,   1), thread idx:  11, cord: (  5,   7)
block idx: (  0,   1,   1), thread idx:  12, cord: (  3,   4)
block idx: (  0,   1,   1), thread idx:  13, cord: (  4,   4)
block idx: (  0,   1,   1), thread idx:  14, cord: (  5,   4)
block idx: (  0,   1,   1), thread idx:  15, cord: (  3,   5)
block idx: (  0,   1,   1), thread idx:  16, cord: (  4,   5)
block idx: (  0,   1,   1), thread idx:  17, cord: (  5,   5)
block idx: (  0,   1,   1), thread idx:  18, cord: (  3,   6)
block idx: (  0,   1,   1), thread idx:  19, cord: (  4,   6)
block idx: (  0,   1,   1), thread idx:  20, cord: (  5,   6)
block idx: (  0,   1,   1), thread idx:  21, cord: (  3,   7)
block idx: (  0,   1,   1), thread idx:  22, cord: (  4,   7)
block idx: (  0,   1,   1), thread idx:  23, cord: (  5,   7)
block idx: (  1,   0,   0), thread idx:   0, cord: (  0,   0)
block idx: (  1,   0,   0), thread idx:   1, cord: (  1,   0)
block idx: (  1,   0,   0), thread idx:   2, cord: (  2,   0)
```

## 2. 对比GPU 和 CPU的矩阵乘法的结果

这里对比一下1024x1024的矩阵相乘的速度，下面是对main函数分段的解析

### 2.1 CPU上的矩阵相乘的方法
cpu的办法会简单一些
```cpp
void MatmulOnHost(float *M, float *N, float *P, int width)
{
    for (int i = 0; i < width; i ++)
    {
        for (int j = 0; j < width; j++)
        {
            float sum = 0;
            for (int k = 0; k < width; k ++)
            {
                // M的行乘N的列, 这个循环M行每一个乘N的一个
                float a = M[i * width + k];
                float b = N[k * width + j];
                sum += a * b;
            }
            P[i * width + j] = sum;   // 
        }
    }     
}
```

### 2.2 GPU举证相乘的流程

MatmulOnDevice() 是给cpp文件调用的 MatmulKernel()用来写

看一下函数的输入, case里面width设置的是1024, M_host, h_host都是1024x1024的矩阵, 填充是0-1之前的浮点数, 这里假设矩阵相乘都是方阵的(height = width)

```cpp
#ifndef MATMUL_GPU_BASIC_H
#define MATMUL_GPU_BASIC_H

// CUDA运行时库
#include "cuda_runtime.h"
#include "cuda.h"

// 函数声明

/**
 * 用于矩阵乘法的CUDA内核函数。
 * 
 * @param M_device 指向设备上第一个矩阵的指针。
 * @param N_device 指向设备上第二个矩阵的指针。
 * @param P_device 指向设备上输出矩阵的指针。
 * @param width 矩阵的宽度（假设是方阵）。
 */
__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int width);

/**
 * 在设备上执行两个矩阵相乘的主机函数。
 * 
 * @param M_host 指向主机上第一个矩阵的指针。
 * @param N_host 指向主机上第二个矩阵的指针。
 * @param P_host 指向主机上输出矩阵的指针。
 * @param width 矩阵的宽度（假设是方阵）。
 * @param blockSize CUDA块的大小。
 */
void MatmulOnDevice(float *M_host, float *N_host, float* P_host, int width, int blockSize);

#endif // MATMUL_GPU_BASIC_H
```

**MatmulOnDevice()** 

```bash
- 设置size, 矩阵大小, 用来分配内存
- 分配GPU内存，输入输出
-  设置grid, block的布局
```

在之前的Grid, Block布局分析中提到过, block和grid的布局最好跟计算的内容是一致的, 例如说图像和这里的矩阵是2D, 所以block的中的线程设置是2D, 一个block里面包含16x16=256, 32x32=1024个线程, 然后grid里面包含多少个block是基于这个计算出来的, 可以做一个向上取整确保有足够的线程计算

设计布局的时候，如果处理的是矩阵，或者是二维度的图像,  先设计好好block里面的线程规划，然后基于这个设计好grid中的block规划

这里的设计方案就是把一个矩阵切分成多个block来计算, 这里的case是1024x1024的, 用**blockSize** = 32 刚好够, 如果用16的话就是把1024x1024分成多个

这里其实就是计算每一个线程的计算, 之前知道, 这里会堆出一大堆线程索引例如说(0, 0, 1)....(2, 1, 2) 对应的是第2个block块, x = 1, y = 2 的线程, 这些线程会同时计算但是并不会按顺序计算, 所以后面会有一个同步等待其他的线程一次性做完这些操作
```cpp
void MatmulOnDevice(float *M_host, float *N_host, 
                    float* P_host, int width, int blockSize)
{
    /*
    M_host: First Matrix ptr at host 
    h_host: second matrix ptr at host
    P_host: output matrix ptr at host 
    */
   // 设置矩阵尺寸
    int size = width * width* sizeof(float);
    // 开辟GPU内存
    float *M_device;
    float *N_device;
    float *P_device;

    cudaMalloc(&M_device, size);
    cudaMalloc(&N_device, size);
    cudaMalloc(&P_device, size);

    // 把输入输出的矩阵信息从host搬到device
    cudaMemcpy(M_device, M_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_device, N_host,  size, cudaMemcpyHostToDevice);

    // 分配grid, block
    dim3 dimBlock(blockSize, blockSize);
    int gridDim = (width + blockSize - 1) / blockSize;
    dim3 dimGrid(gridDim, gridDim);

    // 调用kernel function计算
    MatmulKernel<<<dimGrid, dimBlock>>>(M_device, N_device, P_device, width);

    // 计算结果从device搬到host
    cudaMemcpy(P_host, P_device, size , cudaMemcpyDeviceToHost);

    // 等待全部线程完成计算
    cudaDeviceSynchronize();

    // Free
    cudaFree(P_device);
    cudaFree(M_device);
    cudaFree(N_device);

}
```


**MatmulKernel()**

这里的int x, int y是一个数字, 因为在GPU上的内存是连续的, 我们之前分配的block, grid就是用来管理我自己的理解是索引写完就拿一个case出来写一个线程的计算, 写完就明白了。

以这个case为例，总共有1024x1024个元素需要处理, 如果blockSize设置的是32, 每个block里面就有32x32=1024个线程处理这个项目, 根据计算就有(32, 32)个block, 也就是1024个

M_element, N_element, p_element属于是每一个线程的局部变量, P_element在每一个线程都会有0, 然后M_element, N_element, P_device的数都是通过

这里以(3, 2) 为案例, 就可以很好理解下面的M_element, N_element, p_element。 
```cpp
__global__ void MatmulKernel(float *M_device, float *N_device, float *P_device, int width){
    /* 
        我们设定每一个thread负责P中的一个坐标的matmul
        所以一共有width * width个thread并行处理P的计算
    */
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float P_element = 0;
    for (int k = 0; k < width; k++){
        float M_element = M_device[y * width + k]; // 行
        float N_element = N_device[k * width + x]; // 列
        P_element += M_element * N_element;  // 这个结束就是行列相乘
    }

    P_device[y * width + x] = P_element; // 第几行 + 第几列
}
```

### 2.3 实验测试
我自己这边跟韩导的实验结果不一样，他的卡上面实现了一个1500倍的加速但是我这边实现的是414倍的加速，在blockSize = 16的情况下实现的, 这里也说明了blockSize不是越大越好的
```bash
matmul in cpu                  uses 4149.35 ms
matmul in GPU Warmup           uses 173.9 ms
matmul in GPU blockSize = 16   uses 9.90609 ms
matmul in GPU blockSize = 32   uses 13.2933 ms
Matmul result is same, precision is 1.0E-4
```


## 3. CUDA Error Handle和获取信息

一个良好的cuda编程习惯里，我们习惯在调用一个cuda runtime api时，例如cudaMalloc() cudaMemcpy()我们就用error handler进行包装。这样
可以方便我们排查错误的来源

具体来说，CUDA的runtime API都会返回一个cudaError(枚举类), 可以通过枚举类来查看到它里面要么是成功了要么就是各种错误

```__FILE__, __LINE__```这两个指的是当前文件，下面的行和文件名就是这里来的
```bash
ERROR: src/matmul_gpu_basic.cu:62, CODE:cudaErrorInvalidConfiguration, DETAIL:invalid configuration argument
```

至于这里两个，宏定义, 一个是用来检查CUDA Runtime API, 一个是检查核函数的。检查kernel function的时候，用```LAST_KERNEL_CHECK()```, 这个放在同步后面, 确保之前的所有CUDA操作（包括kernel的执行）都已经完成,Z再来检查

有cudaPeekAtLastError或者cudaGetLastError, 区别是是否传播错误
```cpp
kernelFunction<<<numBlocks, numThreads>>>();
cudaError_t err1 = cudaPeekAtLastError();  // 只查看，不清除错误状态
cudaError_t err2 = cudaGetLastError();  // 查看并清除错误状态
```

```cpp
#include <cuda_runtime.h>
#include <system_error>

#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK()          __kernelCheck(__FILE__, __LINE__)
#define BLOCKSIZE 16

inline static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

inline static void __kernelCheck(const char* file, const int line) {
    /* 
     * 在编写CUDA是，错误排查非常重要，默认的cuda runtime API中的函数都会返回cudaError_t类型的结果，
     * 但是在写kernel函数的时候，需要通过cudaPeekAtLastError或者cudaGetLastError来获取错误
     */
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}
```



### 3.1 两个错误案例
#### EX1: 
这里分配之前矩阵乘法的blockSize = 64, 那么一个线程块里面有64x64=4096个线程，超出了1024的限制, 下面是不用KernelCheck()和用了的区别

不加是不会报错的
```bash
matmul in cpu                  uses 4092.84 ms
matmul in GPU Warmup           uses 199.453 ms
matmul in GPU blockSize = 1    uses 13.1558 ms
matmul in GPU blockSize = 16   uses 13.0716 ms
matmul in GPU blockSize = 32   uses 13.0694 ms
matmul in GPU blockSize = 64   uses 2.00626 ms
res is different in 0, cpu: 260.89050293, gpu: 0.00000000
Matmul result is different
```

**加了会出现报错**, 这个错误 cudaErrorInvalidConfiguration 表示在执行CUDA kernel时，传递给 kernel 的配置参数无效。具体来说，CUDA kernel的配置包括线程块的数量、线程块内线程的数量等。
```bash
matmul in cpu                  uses 4115.42 ms
matmul in GPU Warmup           uses 201.464 ms
matmul in GPU blockSize = 1    uses 13.1182 ms
matmul in GPU blockSize = 16   uses 13.0607 ms
matmul in GPU blockSize = 32   uses 13.0602 ms
ERROR: src/matmul_gpu_basic.cu:69, CODE:cudaErrorInvalidConfiguration, DETAIL:invalid configuration argument
```

#### EX2: 
```cpp
    // 分配grid, block
    dim3 dimBlock(blockSize, blockSize);
    int gridDim = (width + blockSize - 1) / blockSize;
    dim3 dimGrid(gridDim, gridDim);
```
**写成了**
```cpp
    // 分配grid, block
    dim3 dimBlock(blockSize, blockSize);
    int gridDim = (width + blockSize - 1) / blockSize;
    dim3 dimGrid(gridDim);
```

```bash
matmul in cpu                  uses 4152.26 ms
matmul in GPU Warmup           uses 189.667 ms
matmul in GPU blockSize = 1    uses 2.92747 ms
matmul in GPU blockSize = 16   uses 2.85372 ms
matmul in GPU blockSize = 32   uses 2.86483 ms
res is different in 32768, cpu: 260.76977539, gpu: 0.00000000
```

这个没有报错, 这里grid(网格)只有一个,  然后这里不够块去计算了, 所以计算了一部分他就不计算了, 所以运行的速度快了很多, 以后如果CUDA编程中速度快了很多，要参考是否是没有完整的计算。



### 3.4 为什么要获取硬件信息

当进行CUDA编程时，了解硬件规格是非常重要的，因为这些规格限制了你可以使用的并行策略和优化方式。

```cpp
*********************Architecture related**********************
Device id:                              7
Device name:                            NVIDIA GeForce RTX 3090
Device compute capability:              8.6
GPU global meory size:                  23.70GB
L2 cache size:                          6.00MB
Shared memory per block:                48.00KB
Shared memory per SM:                   100.00KB
Device clock rate:                      1.69GHz
Device memory clock rate:               9.75Ghz
Number of SM:                           82
Warp size:                              32
*********************Parameter related************************
Max block numbers:                      16
Max threads per block:                  1024
Max block dimension size:               1024:1024:64
Max grid dimension size:                2147483647:65535:65535
```

### 3.5 代码
```cpp
#include <cuda_runtime.h>
#include <system_error>
#include <stdarg.h>

#define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KERNEL_CHECK(call)      __kernelCheck(__FILE__, __LINE__)
#define LOG(...)                     __log_info(__VA_ARGS__)

#define BLOCKSIZE 16

static void __cudaCheck(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

static void __kernelCheck(const char* file, const int line) {
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s:%d, ", file, line);
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
        exit(1);
    }
}

// 使用变参进行LOG的打印。比较推荐的打印log的写法
static void __log_info(const char* format, ...) {
    char msg[1000];
    va_list args;
    va_start(args, format);

    vsnprintf(msg, sizeof(msg), format, args);

    fprintf(stdout, "%s\n", msg);
    va_end(args);
}
```

```cpp
#include <stdio.h>
#include <cuda_runtime.h>
#include <string>

#include "utils.hpp"

int main(){
    int count;
    int index = 0;
    cudaGetDeviceCount(&count);
    while (index < count) {
        cudaSetDevice(index);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, index);
        LOG("%-40s",             "*********************Architecture related**********************");
        LOG("%-40s%d%s",         "Device id: ",                   index, "");
        LOG("%-40s%s%s",         "Device name: ",                 prop.name, "");
        LOG("%-40s%.1f%s",       "Device compute capability: ",   prop.major + (float)prop.minor / 10, "");
        LOG("%-40s%.2f%s",       "GPU global meory size: ",       (float)prop.totalGlobalMem / (1<<30), "GB");
        LOG("%-40s%.2f%s",       "L2 cache size: ",               (float)prop.l2CacheSize / (1<<20), "MB");
        LOG("%-40s%.2f%s",       "Shared memory per block: ",     (float)prop.sharedMemPerBlock / (1<<10), "KB");
        LOG("%-40s%.2f%s",       "Shared memory per SM: ",        (float)prop.sharedMemPerMultiprocessor / (1<<10), "KB");
        LOG("%-40s%.2f%s",       "Device clock rate: ",           prop.clockRate*1E-6, "GHz");
        LOG("%-40s%.2f%s",       "Device memory clock rate: ",    prop.memoryClockRate*1E-6, "Ghz");
        LOG("%-40s%d%s",         "Number of SM: ",                prop.multiProcessorCount, "");
        LOG("%-40s%d%s",         "Warp size: ",                   prop.warpSize, "");

        LOG("%-40s",             "*********************Parameter related************************");
        LOG("%-40s%d%s",         "Max block numbers: ",           prop.maxBlocksPerMultiProcessor, "");
        LOG("%-40s%d%s",         "Max threads per block: ",       prop.maxThreadsPerBlock, "");
        LOG("%-40s%d:%d:%d%s",   "Max block dimension size:",     prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], "");
        LOG("%-40s%d:%d:%d%s",   "Max grid dimension size: ",     prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2], "");
        index ++;
        printf("\n");
    }
    return 0;
}
```

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

上图是nsight compute, 下图是nsight system。 


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

最简单的理解就是之前是[ty][tx] =====> [tx][ty] , 左图是bank conflict, 右图是解决bank conflict的分布
![在这里插入图片描述](https://img-blog.csdnimg.cn/9088c6317a954a19ab39908804b2504a.png)
#### 6.2.1 创造bank conflict
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

#### 6.2.2 用pad的方式解决bank conflict
```cpp
__global__ void MatmulSharedStaticConflictPadKernel(float *M_device, float *N_device, float *P_device, int width){
    /* 添加一个padding，可以防止bank conflict发生，结合图理解一下*/
    __shared__ float M_deviceShared[BLOCKSIZE][BLOCKSIZE + 1];
    __shared__ float N_deviceShared[BLOCKSIZE][BLOCKSIZE + 1];
    /* 
        对于x和y, 根据blockID, tile大小和threadID进行索引
    */
    int x = blockIdx.x * BLOCKSIZE + threadIdx.x;
    int y = blockIdx.y * BLOCKSIZE + threadIdx.y;

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
虽然说

```bash
Input size is 4096 x 4096
matmul in gpu(warmup)                                        uses 113.364067 ms
matmul in gpu(general)                                       uses 114.303902 ms
matmul in gpu(shared memory(static))                         uses 73.318878 ms
matmul in gpu(shared memory(static, bank conf))              uses 141.755173 ms
matmul in gpu(shared memory(static, pad resolve bank conf))  uses 107.326782 ms
matmul in gpu(shared memory(dynamic))                        uses 90.047234 ms
matmul in gpu(shared memory(dynamic, bank conf)              uses 191.804550 ms
matmul in gpu(shared memory(dynamic, pad resolve bank conf)) uses 108.733856 ms
```
在设计核函数时候通过选择合适的数据访问模式来避免bank conflicts是一种常用的优化策略。

在CUDA编程中，通常推荐的做法是：

1. 行优先访问：因为CUDA的内存是按行优先顺序存储的，所以采用行优先访问可以更好地利用内存带宽，减少bank conflicts。

2. 合适的数据对齐：通过确保数据结构的对齐也可以减少bank conflicts。例如，可以通过padding来确保矩阵的每行都是一个固定数量的word长。

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

## 8. 通过深度学习的前处理案例学习CUDA

双线性插值

![在这里插入图片描述](https://img-blog.csdnimg.cn/b8f55b6350a842e8bd97fa4a49e91540.png)


### 8.1 CPU版本的前处理
这里直接用一个resize在CPU上做对比，当然这里其实就是简单resize了一下, 不做过多的分析
```cpp
cv::Mat preprocess_cpu(cv::Mat &src, const int &tar_h, const int &tar_w, Timer timer, int tactis) {
    cv::Mat tar;

    timer.start_cpu();

    /*BGR2RGB*/
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);

    /*Resize*/
    cv::resize(src, tar, cv::Size(tar_w, tar_h), 0, 0, cv::INTER_LINEAR);

    timer.stop_cpu();
    timer.duration_cpu<Timer::ms>("Resize(bilinear) in cpu takes:");

    return tar;
}
```

### 8.2 调用kernel的主函数

这里看一个GPU正儿八经的GPU版本的核函数, 先看调用这个的主函数

这里是target图像的宽高, 还有src图像的宽高, tactis不用管。 
```cpp
void resize_bilinear_gpu(
    uint8_t* d_tar, uint8_t* d_src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    int tactis) 
```

这里是grid和block的分布, 这里有一个+1是为了保证有足够的资源去处理这个问题, 这里也是因为之前分析过，16是比较好的性能, 
```cpp
dim3 dimBlock(16, 16, 1);
dim3 dimGrid(tarW / 16 + 1, tarH / 16 + 1, 1);
```

计算缩放因子,为了满足条件，缩多了的h/w就填充, 下面让一个缩放因子代表h,w是为了不让图像变形, 这里有点抽象，画个图理解一下，
```cpp
//scaled resize
float scaled_h = (float)srcH / tarH;
float scaled_w = (float)srcW / tarW;
float scale = (scaled_h > scaled_w ? scaled_h : scaled_w);
scaled_h = scale;
scaled_w = scale;
resize_bilinear_BGR2RGB_shift_kernel <<<dimGrid, dimBlock>>> (d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w, scaled_h);
```

### 8.3 kernel function

```cpp
__global__ void resize_bilinear_BGR2RGB_shift_kernel(
    uint8_t* tar, uint8_t* src, 
    int tarW, int tarH, 
    int srcW, int srcH, 
    float scaled_w, float scaled_h) 
```

y是行的索引, 这个容易弄混淆
```cpp
// bilinear interpolation -- resized之后的图tar上的坐标
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

这里是通过target image的图片来寻找不同的坐标, 因为我们在上面统一了缩放因子, 所以这里会有一些超出边界的，就不计算了，这也是目标图为什么会有填充，因为映射回去越界了就给他填充起来了

- (y + 0.5): 中心对齐, 把像素值从左上角移动到中心
- * scaled_h：映射回去
-  - 0.5: 重新把像素移动到左上角
- 公式如图所示
![**](https://img-blog.csdnimg.cn/f792ee9b0d46436e81b7e97c7c2fdd86.png)


```cpp
    // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
    int src_y1 = floor((y + 0.5) * scaled_h - 0.5);
    int src_x1 = floor((x + 0.5) * scaled_w - 0.5);
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    if (src_y1 < 0 || src_x1 < 0 || src_y1 > srcH || src_x1 > srcW) {
        // bilinear interpolation -- 对于越界的坐标不进行计算
    } else {
        // 这里计算的是双线性插值的东西
```

这里计算的是左上角面积的tw, th, 可以理解为之前的floor了, 这里取的就是floor余下的

![在这里插入图片描述](https://img-blog.csdnimg.cn/8d6c018c617444468bee9e083472b4db.png)


```cpp
float th   = ((y + 0.5) * scaled_h - 0.5) - src_y1;
float tw   = ((x + 0.5) * scaled_w - 0.5) - src_x1;
```

计算四个单位的面积
```cpp
// bilinear interpolation -- 计算面积(这里建议自己手画一张图来理解一下)
float a1_1 = (1.0 - tw) * (1.0 - th);  //右下
float a1_2 = tw * (1.0 - th);          //左下
float a2_1 = (1.0 - tw) * th;          //右上
float a2_2 = tw * th;                  //左上
```

计算原图和目标图的索引, 这里要知道rgb是连续的，所以要 * 3, 还有这边便宜的时候是先把每一个像素移动到最上面/最左边, 然后再弄到中间，
```cpp
// bilinear interpolation -- 计算4个坐标所对应的索引
int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;  //左上
int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;  //右上
int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;  //左下
int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;  //右下


y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);
int tarIdx    = (y * tarW  + x) * 3;
```

计算目标图的像素顺便,  rgb转一下，因为opencv读的是bgr, 在这里转了
```cpp
// bilinear interpolation -- 实现bilinear interpolation的resize + BGR2RGB
tar[tarIdx + 0] = round(
                    a1_1 * src[srcIdx1_1 + 2] + 
                    a1_2 * src[srcIdx1_2 + 2] +
                    a2_1 * src[srcIdx2_1 + 2] +
                    a2_2 * src[srcIdx2_2 + 2]);

tar[tarIdx + 1] = round(
                    a1_1 * src[srcIdx1_1 + 1] + 
                    a1_2 * src[srcIdx1_2 + 1] +
                    a2_1 * src[srcIdx2_1 + 1] +
                    a2_2 * src[srcIdx2_2 + 1]);

tar[tarIdx + 2] = round(
                    a1_1 * src[srcIdx1_1 + 0] + 
                    a1_2 * src[srcIdx1_2 + 0] +
                    a2_1 * src[srcIdx2_1 + 0] +
                    a2_2 * src[srcIdx2_2 + 0]);
```