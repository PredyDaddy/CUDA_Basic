#include "utils.hpp"

void Timer_Check() {
    Timer timer;  // 创建 Timer 对象

    // 开始计时
    timer.start();

    // 模拟执行一些操作，比如这里可以是矩阵运算、CUDA kernel 执行等
    // 在这里写你想要测试的代码块

    // 结束计时
    timer.stop();

    // 输出执行时间，选择适当的时间单位（s、ms、us、ns）
    timer.duration<Timer::s>("Task execution time:");
}