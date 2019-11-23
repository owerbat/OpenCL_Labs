__kernel void gemm(const uint n, __global const float *a,
                   __global const float *b, __global float *c) {
    const uint iRow = get_global_id(1);
    const uint iCol = get_global_id(0);

    if (iRow < n && iCol < n) {
        float result = 0.0f;
        for (uint k = 0; k < n; ++k)
            result += a[iRow * n + k] * b[k * n + iCol];
        c[iRow * n + iCol] = result;
    }
}
