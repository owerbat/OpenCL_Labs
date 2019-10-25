__kernel void axpy(const size_t n, const float a, const float *x, const size_t incx, float *y, const size_t incy) {
    y[i * incy] = y[i * incy] + a * x[i * incx];
}
