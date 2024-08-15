// Grid, Block, Thread

/*
Thread Organization
- A grid has blocks and a block has threads
- A kernel is executed as a grid of thread blocks
- A grid is a 3D array of thread blocks (gridDim.x, gridDim.y, gridDim.z)
- Each thread block has a unique index within the grid (blockIdx.x, blockIdx.y, blockIdx.z)
- A thread block is a 3D array of threads (blockDim.x, blockDim.y, blockDim.z)
- Each thread has a unique index within the block (threadIdx.x, threadIdx.y, threadIdx.z)


Typical Configuration
- 1-5 blocks per SM( Streaming Multiprocessor)
- 128-1024 threads per block
- Total 2k-100k threads
- You can launch a kernel with millions of threads
*/