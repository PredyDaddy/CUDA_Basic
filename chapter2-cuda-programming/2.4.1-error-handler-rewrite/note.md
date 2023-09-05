### 1. 模板的复习
简单看下函数模版的定义

函数模版的意义: 对类型也可以进行参数化了，不仅仅是减少了工作量，从汇编的角度不用重复构造compare()函数，从而减小了可执行文件的大小

函数模版的语法:

template 关键字用于声明开始进行泛型编程
typename 关键字用于声明泛指类型
需要注意的是：函数模板是不允许隐式类型转换的，调用时类型必须严格匹配
```cpp
#include <iostream>
using namespace std;

template<typename T>
bool compare(T a, T b)
{
    cout << "Template compare" << endl;
    return a > b;
}

int main()
{
    // 函数的调用点
    compare<int>(10, 20);
    compare<double>(10.1, 20.5);
    return 0;
}
```

复习模版用来理解代码里面的计时器

### 2. 计时器Timer
 
```cpp
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

```