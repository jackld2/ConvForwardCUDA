#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

const int constMemSize = 16*4*7*7;  //M*C*K*K
__constant__ float Kc[constMemSize];//filter-bank

__global__ void shared_mem_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) Kc[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(W_out*1.0 / TILE_WIDTH);  //# of tiles in width
    int n, m, h, w, c, p, q;
    n = blockIdx.x; //current channel
    m = blockIdx.y; //current ouput feature map
    h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

      float acc = 0.;
      if(h < H_out && w < W_out){
        for(c=0; c<C; c++)
        {
          for(p=0; p<K; p++)
          {
            for(q=0; q<K; q++)
            {
              acc+= x4d(n, c, h+p, w+q) * k4d(m, c, p, q);
            }
          }
        }
        y4d(n, m, h, w) = acc;
      }

#undef y4d
#undef x4d
#undef k4d
}

__global__ void combined_unroll_mm_kernel(float *y, float *x, float *w, int B, int C, int H, int K, int W, int M)
{
    __shared__ float MaskTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float InputTile[TILE_WIDTH][TILE_WIDTH];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;


    #define k4d(i3, i2, i1, i0) w[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int column = blockIdx.x * TILE_WIDTH + tx;
    int unrollColumn = C*K*K;

    float acc = 0.0;
    int num_iterations = ceil(unrollColumn/(1.0*TILE_WIDTH));

    for (int i = 0; i < num_iterations; i++) {
      int lx = i*TILE_WIDTH + tx;
      int ly = i*TILE_WIDTH + ty;

      MaskTile[ty][tx] = 0;
      InputTile[ty][tx] = 0;

      int W_m = row;
      int W_c = lx/(K*K);
      int W_h = (lx%(K*K))/K;
      int W_w = (lx%(K*K))%K;

      if ((lx < unrollColumn) && (row < M)){
        MaskTile[ty][tx] = k4d(W_m, W_c, W_h, W_w);
      }
      else{
        MaskTile[ty][tx] = 0;
      }

      int X_b = b;
      int X_c = ly/(K*K);
      int X_p = (ly%(K*K))/K;
      int X_q = (ly%(K*K))%K;
      int X_h = column/W_out;
      int X_w = column%W_out;

      if (ly < unrollColumn && column < H_out*W_out){
        InputTile[ty][tx] = x4d(X_b, X_c, X_h + X_p, X_w + X_q);
      }
      else{
        InputTile[ty][tx] = 0;
      }
      __syncthreads();

      for (int q = 0; q < TILE_WIDTH; q++){
        acc += MaskTile[ty][q] * InputTile[q][tx];
      }

      __syncthreads();
    }

    int Y_b = b;
    int Y_m = row;
    int Y_h = column / W_out;
    int Y_w = column % W_out;

    if (row < M && column < W_out*H_out)
      y4d(Y_b, Y_m, Y_h, Y_w) = acc;
}

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float* device_y;
    float* device_x;
    float* device_k;

    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void**) &device_y, B*M*(H-K+1)*(W-K+1)*sizeof(float));
    cudaMalloc((void**) &device_x, B*C*H*W*sizeof(float));
    cudaMalloc((void**) &device_k, M*C*K*K*sizeof(float));
    std::cout<< "M: "<<M<<"\n";
    std::cout<< "C: "<<C<<"\n";
    std::cout<< "K: "<<K<<"\n";
    std::cout<< "H: "<<H<<"\n";
    std::cout<< "W: "<<W<<"\n";
    std::cout<< "B: "<<B<<"\n";
    get_device_properties();

    //copy input to GPU
    cudaMemcpy(device_x, host_x, B*C*H*W*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_k, host_k, M*C*K*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Kc, host_k, 7*7*4*16*sizeof(float));

    // Set the kernel dimensions and call the kernel
    int W_out = H - K + 1;  //output feature map width
    int H_out = W - K + 1;  //output feature map height

    if (C == 1) {
      int W_grid = ceil(W_out*1.0 / TILE_WIDTH);  //# of tiles in width
      int H_grid = ceil(H_out*1.0 / TILE_WIDTH);  //# of titls in height
      int Z = H_grid * W_grid;                    //total number of tile
      dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);   //thread block size
      dim3 gridDim(B, M, Z);                      //batch_size, # of output feature maps, total number of tiles
      shared_mem_kernel<<<gridDim, blockDim>>>(device_y, device_x, B, M, C, H, W, K);
    }
    else {
      dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
      dim3 gridDim(ceil(H_out*W_out/(1.0*TILE_WIDTH)), ceil(M/(1.0*TILE_WIDTH)), B);
      combined_unroll_mm_kernel<<<gridDim, blockDim>>>(device_y,device_x,device_k,B,C,H,K,W,M);

    }


    // Copy the output back to host
    cudaMemcpy(host_y, device_y, B*M*(H-K+1)*(W-K+1)*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_k);

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
