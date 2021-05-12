#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"
#define TILE_WIDTH 16

const int constMemSize = (16*4*7*7)/2;  //M*C*K*K
__constant__ half2 Kc[constMemSize];//filter-bank

__global__ void shared_mem_kernel(half *y, const half2 *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[((i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0) / 2]
#define k4d(i3, i2, i1, i0) Kc[((i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0) / 2]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(W_out*1.0 / TILE_WIDTH);  //# of tiles in width
    int n, m, h, w, c, p, q;
    n = blockIdx.x; //current channel
    m = blockIdx.y; //current ouput feature map
    h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;

    half2 acc = __floats2half2_rn(0.0f,0.0f);
    half total;
      if(h < H_out && w < W_out){
        for(c=0; c<C; c++)
        {


          int d;
          for (d = 0; d < K*K - 1; d+=2) {
            p = d / 7;
            q = d % 7;
            half2 x_half = x4d(n, c, h+p, w+q);

            acc = __hadd2(acc, __hmul2(x4d(n, c, h+p, w+q), k4d(m, c, p, q)));
          }
          total = __hadd(acc.x, acc.y);
          total = __hadd(total, __hmul(x4d(n, c, h+6, w+6).x, k4d(m, c, 6, 6).x));
        }
        y4d(n, m, h, w) = total;
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
    half2* device_x;
    half2* device_k;

    half2* half_x;
    half* half_y;
    half2* half_k;

    const int y_size = B*M*(H-K+1)*(W-K+1);
    const int x_size = ceil((1.0*B*C*H*W)/2);
    const int k_size = ceil((1.0*M*C*K*K)/2);

    half_y = (half*)malloc( y_size*sizeof(half));
    half_x = (half2*)malloc( x_size*sizeof(half2));
    half_k = (half2*)malloc( k_size*sizeof(half2));


    for (int i = 0; i < x_size; i++) {
      if (2*i+1 < B*C*H*W) {
        half_x[i] = __floats2half2_rn(host_x[2*i],host_x[2*i+1]);
      }
      else {
        half_x[i] = __floats2half2_rn(host_x[2*i], 0.0f);
      }
    }

    for (int i = 0; i < k_size; i++) {
      if (2*i+1 < M*C*K*K) {
        half_k[i] = __floats2half2_rn(host_k[2*i],host_k[2*i+1]);
      }
      else {
        half_k[i] = __floats2half2_rn(host_k[2*i], 0.0f);
      }
    }





    // Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void**) &device_y, y_size*sizeof(half));
    cudaMalloc((void**) &device_x, x_size*sizeof(half2));
    cudaMalloc((void**) &device_k, k_size*sizeof(half2));
    std::cout<< "M: "<<M<<"\n";
    std::cout<< "C: "<<C<<"\n";
    std::cout<< "K: "<<K<<"\n";
    std::cout<< "H: "<<H<<"\n";
    std::cout<< "W: "<<W<<"\n";
    std::cout<< "B: "<<B<<"\n";
    get_device_properties();

    //copy input to GPU
    cudaMemcpy(device_x, half_x, x_size*sizeof(half2), cudaMemcpyHostToDevice);
    //cudaMemcpy(device_k, half_k, M*C*K*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Kc, half_k, (7*7*4*16/2)*sizeof(half));

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
