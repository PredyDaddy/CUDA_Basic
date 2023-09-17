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