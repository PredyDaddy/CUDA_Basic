#ifndef __STREAM_HPP__
#define __STREAM_HPP__

void SleepSingleStream(
    float* src_host, float* tar_host, 
    int width, int blockSize, 
    int count);

void SleepMultiStream(
    float* src_host, float* tar_host,
    int width, int blockSize, 
    int count);

void MySleepSingleStream(
    float* src_host, float* tar_host,
    int width, int blockSize, 
    int count);

void experiment_local_overlap(float *src_host, float *tar_host, int width, int blockSize);

#endif // __STREAM_HPP__
