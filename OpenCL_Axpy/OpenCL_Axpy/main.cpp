#include "axpy.h"

typedef float FPType;

int main() {
    const size_t n = static_cast<size_t>(10e+7), incx = 1, incy = 1;
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

    // CPU
    auto cpuTime = cpu_axpy(n, a, x, incx, y, incy);

    std::cout << "CPU result:";
    for (size_t i = 0; i < 10; ++i)
        std::cout << " " << y[i];
    std::cout << std::endl;

    for (size_t i = 0; i < n; ++i)
        y[i] = static_cast<FPType>(2);

    // OpenCL CPU
    auto openCLCPUTime = opencl_axpy(n, a, x, incx, y, incy, CL_DEVICE_TYPE_CPU);

    std::cout << "OpenCL CPU result:";
    for (size_t i = 0; i < 10; ++i)
        std::cout << " " << y[i];
    std::cout << std::endl;

    for (size_t i = 0; i < n; ++i)
        y[i] = static_cast<FPType>(2);

    // OpenCL GPU
    auto openCLGPUTime = opencl_axpy(n, a, x, incx, y, incy);

    std::cout << "OpenCL GPU result:";
    for (size_t i = 0; i < 10; ++i)
        std::cout << " " << y[i];
    std::cout << std::endl;

    for (size_t i = 0; i < n; ++i)
        y[i] = static_cast<FPType>(2);

    // OpenMP
    auto ompTime = omp_axpy(n, a, x, incx, y, incy);

    std::cout << "OpenMP result:";
    for (size_t i = 0; i < 10; ++i)
        std::cout << " " << y[i];
    std::cout << std::endl;

    // Total
    std::cout << "Time:\n"
              << "CPU        " << std::chrono::duration_cast<std::chrono::milliseconds>(cpuTime).count() << " ms\n"
              << "OpenCL CPU " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLCPUTime).count() << " ms\n"
              << "OpenCL GPU " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLGPUTime).count() << " ms\n"
              << "OpenMP     " << std::chrono::duration_cast<std::chrono::milliseconds>(ompTime).count() << " ms\n";

    delete[] x, y;

    return 0;
}
