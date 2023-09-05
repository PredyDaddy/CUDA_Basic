#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <cstdio>
#include <ratio>
#include <string>
#include <iostream>
#include <stdio.h>

class Timer 
{
public:
    // 这些就是后面用到的span的类型
    using s  = std::ratio<1, 1>;
    using ms = std::ratio<1, 1000>;
    using us = std::ratio<1, 1000000>;
    using ns = std::ratio<1, 1000000000>;

// 构造函数
public:
    Timer(){};

public:
    void start(){mStart = std::chrono::system_clock::now();} 
    void stop(){mStop = std::chrono::system_clock::now();} 

    template<typename span>
    void duration(std::string msg);
    
private:
    // std::chrono::time_point 是类模板
    // std::chrono::high_resolution_clock 是C++的高性能时钟
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> mStop;
    
};

// 构造函数外实现duration
template <typename span>
void Timer::duration(std::string msg)
{
    // sprintf用于格式化内容，准备这个span，然后
    std::string str;
    char fMsg[100];
    std::sprintf(fMsg, "%-30s", msg.c_str());

    if(std::is_same<span, s>::value) { str = " s"; }
    else if(std::is_same<span, ms>::value) { str = " ms"; }
    else if(std::is_same<span, us>::value) { str = " us"; }
    else if(std::is_same<span, ns>::value) { str = " ns"; }

    std::chrono::duration<double, span> time = mStop - mStart;
    std::cout << fMsg << " uses " << time.count() << str << std::endl;
}

#endif 
