#include "axpy.h"

typedef float FPType;

int main() {
    const size_t n = 1024, incx = 1, incy = 1;
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

    cpu_axpy(n, a, x, incx, y, incy);

    std::cout << "CPU result:";
    for (size_t i = 0; i < 10; ++i)
        std::cout << " " << y[i];
    std::cout << std::endl;

    for (size_t i = 0; i < n; ++i)
        y[i] = static_cast<FPType>(2);

    gpu_axpy(n, a, x, incx, y, incy);

    std::cout << "GPU result:";
    for (size_t i = 0; i < 10; ++i)
        std::cout << " " << y[i];
    std::cout << std::endl;

    delete[] x, y;

    return 0;
}
