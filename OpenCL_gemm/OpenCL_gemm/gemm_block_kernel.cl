#define BLOCK_SIZE 16

__kernel void gemm_block(const uint n, __global float* a,
                         __global float* b, __global float* c) {
    const uint row = get_local_id(0);
    const uint col = get_local_id(1);

    const uint globalRow = BLOCK_SIZE * get_group_id(0) + row;
    const uint globalCol = BLOCK_SIZE * get_group_id(1) + col;

    __local float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __local float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    float result = 0.0f;
    const uint nBlocks = n / BLOCK_SIZE;

    for (uint iBlock = 0; iBlock < nBlocks; ++iBlock) {
        const uint rowOfBlock = BLOCK_SIZE * iBlock + row;
        const uint columnOfBlock = BLOCK_SIZE * iBlock + col;
        Asub[col][row] = a[globalRow * n + columnOfBlock];
        Bsub[col][row] = b[rowOfBlock * n + globalCol];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint i = 0; i < BLOCK_SIZE; i++) {
              result += Asub[i][row] * Bsub[col][i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[globalRow * n + globalCol] = result;
}
