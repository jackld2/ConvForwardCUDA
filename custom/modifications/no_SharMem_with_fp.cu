#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"
#define TILE_WIDTH 16

const int constMemSize = 16*4*7*7;  //M*C*K*K
__constant__ half Kc[constMemSize];//filter-bank

__global__ void shared_mem_kernel(half *y, const half *x, const int B, const int M, const int C, const int H, const int W, const int K)
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

      half acc = 0.0;
      if(h < H_out && w < W_out){
        for(c=0; c<C; c++)
        {
          for(p=0; p<K; p++)
          {
            for(q=0; q<K; q++)
            {
              acc = __hadd(acc, __hmul(x4d(n, c, h+p, w+q), k4d(m, c, p, q)));
            }
          }
        }
        y4d(n, m, h, w) = acc;
      }

#undef y4d
#undef x4d
#undef k4d
}


//acc = __hadd(acc, __hmul(x4d(n, c, h+p, w+q), k4d(m, c, p, q)));
__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    half* device_y;
    half* device_x;
    half* device_k;

    half* half_x;
    half* half_y;
    half* half_k;

    half_y = (half*)malloc( B*M*(H-K+1)*(W-K+1)*sizeof(half));
    half_x = (half*)malloc( B*C*H*W*sizeof(half));
    half_k = (half*)malloc( M*C*K*K*sizeof(half));

    for (int i = 0; i < B*M*(H-K+1)*(W-K+1); i++) {
      half_y[i] = __float2half(host_y[i]);
    }
    for (int i = 0; i < B*C*H*W; i++) {
      half_x[i] = __float2half(host_x[i]);
    }
    for (int i = 0; i < M*C*K*K; i++) {
      half_k[i] = __float2half(host_k[i]);
    }





    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void**) &device_y, B*M*(H-K+1)*(W-K+1)*sizeof(half));
    cudaMalloc((void**) &device_x, B*C*H*W*sizeof(half));
    cudaMalloc((void**) &device_k, M*C*K*K*sizeof(half));
    std::cout<< "M: "<<M<<"\n";
    std::cout<< "C: "<<C<<"\n";
    std::cout<< "K: "<<K<<"\n";
    std::cout<< "H: "<<H<<"\n";
    std::cout<< "W: "<<W<<"\n";
    std::cout<< "B: "<<B<<"\n";
    get_device_properties();

    //copy input to GPU
    cudaMemcpy(device_x, half_x, B*C*H*W*sizeof(half), cudaMemcpyHostToDevice);
    //cudaMemcpy(device_k, half_k, M*C*K*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Kc, half_k, 7*7*4*16*sizeof(half));

    // Set the kernel dimensions and call the kernel
    int W_out = H - K + 1;  //output feature map width
    int H_out = W - K + 1;  //output feature map height
    int W_grid = ceil(W_out*1.0 / TILE_WIDTH);  //# of tiles in width
    int H_grid = ceil(H_out*1.0 / TILE_WIDTH);  //# of titls in height
    int Z = H_grid * W_grid;                    //total number of tile
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);   //thread block size
    dim3 gridDim(B, M, Z);                      //batch_size, # of output feature maps, total number of tiles
    shared_mem_kernel<<<gridDim, blockDim>>>(device_y, device_x, B, M, C, H, W, K);


    // Copy the output back to host
    cudaMemcpy(half_y, device_y, B*M*(H-K+1)*(W-K+1)*sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < B*M*(H-K+1)*(W-K+1); i++) {
      host_y[i] = __half2float(half_y[i]);
    }

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
    free(half_x);
    free(half_y);
    free(half_k);
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
