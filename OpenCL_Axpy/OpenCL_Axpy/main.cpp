#include "axpy.h"
#include <chrono>

typedef float FPType;

int main() {
    const size_t n = static_cast<size_t>(1e+8), incx = 1, incy = 1;
    const FPType a = static_cast<FPType>(1);
    FPType *x = new FPType[n], *y = new FPType[n];

    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<FPType>(1);
        y[i] = static_cast<FPType>(2);
    }

    std::cout << "y = a*x + y\na: " << a << "\nx:";
    for (size_t i = 0; i < 10; ++i)
        std::cout << " " << x[i];
    std::cout << "\ny:";
    for (size_t i = 0; i < 10; ++i)
        std::cout << " " << y[i];
    std::cout << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    cpu_axpy(n, a, x, incx, y, incy);
    auto cpuTime = std::chrono::steady_clock::now() - t0;

    std::cout << "CPU result:";
    for (size_t i = 0; i < 10; ++i)
        std::cout << " " << y[i];
    std::cout << std::endl;

    for (size_t i = 0; i < n; ++i)
        y[i] = static_cast<FPType>(2);

    t0 = std::chrono::steady_clock::now();
    gpu_axpy(n, a, x, incx, y, incy);
    auto gpuTime = std::chrono::steady_clock::now() - t0;

    std::cout << "GPU result:";
    for (size_t i = 0; i < 10; ++i)
        std::cout << " " << y[i];
    std::cout << std::endl;

    for (size_t i = 0; i < n; ++i)
        y[i] = static_cast<FPType>(2);

    t0 = std::chrono::steady_clock::now();
    omp_axpy(n, a, x, incx, y, incy);
    auto ompTime = std::chrono::steady_clock::now() - t0;

    std::cout << "OpenMP result:";
    for (size_t i = 0; i < 10; ++i)
        std::cout << " " << y[i];
    std::cout << std::endl;

    std::cout << "Time:\n"
              << "CPU    " << std::chrono::duration_cast<std::chrono::milliseconds>(cpuTime).count() << " ms\n"
              << "GPU    " << std::chrono::duration_cast<std::chrono::milliseconds>(gpuTime).count() << " ms\n"
              << "OpenMP " << std::chrono::duration_cast<std::chrono::milliseconds>(ompTime).count() << " ms\n";

    delete[] x, y;

    return 0;
}
