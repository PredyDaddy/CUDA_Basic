## 4. Nsight System 和 Nsight Compute
我自己是使用windows端SSH连接远程服务器(Ubuntu 20.04), 然后访问服务器上的容器, 所以就没有办法直接使用, 这个时候就可以在容器用指令生成report, 然后下载下来用windows打开
```bash
# nsight system
nsys profile --trace=cuda,nvtx -o nsight_systems_report ./trt-cuda

# nsight compute
nv-nsight-cu-cli -o nsight_compute_report ./trt-cuda
```
