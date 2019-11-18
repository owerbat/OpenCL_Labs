__kernel void gemm_block(const uint rhs_columns_count, const uint n,
                         __global float* a, __global float* b, __global float* c) {
    const int row = get_local_id(0);
    const int col = get_local_id(1);

    const int globalRow = BLOCK_SIZE * get_group_id(0) + row;
    const int globalCol = BLOCK_SIZE * get_group_id(1) + col;

    __local float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __local float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    float result = 0.0f;
    const int numBlocks = n / BLOCK_SIZE;

    for (int t = 0; t < numBlocks; t++) {
        const int rowOfBlock = BLOCK_SIZE * t + row;
        const int columnOfBlock = BLOCK_SIZE * t + col;
        Asub[col][row] = a[globalRow * n + columnOfBlock];
        Bsub[col][row] = b[rowOfBlock * rhs_columns_count + globalCol];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < BLOCK_SIZE; i++) {
              result += Asub[i][row] * Bsub[col][i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[globalRow * rhs_columns_count + globalCol] = result;
}
