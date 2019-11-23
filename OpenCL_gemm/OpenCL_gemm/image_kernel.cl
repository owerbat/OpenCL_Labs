#define BLOCK_SIZE 16

__kernel void matrixMulImg(__write_only image2d_t C, __read_only image2d_t A, __read_only image2d_t B) {
    int row = get_local_id(0);
    int col = get_local_id(1);
    const int globalRow = BLOCK_SIZE * get_group_id(0) + row;
    const int globalCol = BLOCK_SIZE * get_group_id(1) + col;
    int n = get_global_size(0);
    local float Asub[BLOCK_SIZE][BLOCK_SIZE];
    local float Bsub[BLOCK_SIZE][BLOCK_SIZE];


    float total = 0.0f;
    const int numTiles = n / BLOCK_SIZE;
    for (int t = 0; t < numTiles; t++) {
        const int tiledRow = BLOCK_SIZE * t + row;
        const int tiledCol = BLOCK_SIZE * t + col;
        const int2 idA = {tiledCol, globalRow};
        const int2 idB = {globalCol, tiledRow};
        Asub[col][row] = read_imagef(A, idA).x;
        Bsub[col][row] = read_imagef(B, idB).x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k=0; k < BLOCK_SIZE; k++) {
            total += Asub[k][row] * Bsub[col][k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
	//if (row == col) printf("%f ", total);
    const int2 idC = {globalCol, globalRow};
    write_imagef(C, idC, total);
}
