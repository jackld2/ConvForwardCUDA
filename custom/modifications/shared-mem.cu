#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

const int constMemSize = 7*7*4*16;
__constant__ float Kc[constMemSize];

__global__ void conv_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define kC4d(i3, i2, i1, i0) Kc[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int n, m, h0, w0, h_base, w_base, h, w;
    //int X_tile_width = TILE_WIDTH + K - 1; // K=7, tile_width = 16
    __shared__ float X_shared[TILE_WIDTH][TILE_WIDTH];


    int W_grid = ceil(W_out*1.0 / TILE_WIDTH);  //# of tiles in width
    n = blockIdx.x; //current batch
    m = blockIdx.y; //current ouput feature map
    h0 = threadIdx.x; //current height of pixel of the input tile
    w0 = threadIdx.y; //current width of pixel of the input tile
    h_base = (blockIdx.z / W_grid)*TILE_WIDTH; // vertical base out data index for the block
    w_base = (blockIdx.z % W_grid)*TILE_WIDTH; // horizontal base out data index for the block
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0.0;
    int c, p, q;

    if (h < H_out && w < W_out) {
      for (c = 0; c < C; c++) { // for each input feature map
        X_shared[h0][w0]= x4d(n, c, h, w);

        __syncthreads();
        for (p = 0; p < K; p++) {
          for (q = 0; q < K; q++){
            if (((h0+p < TILE_WIDTH) && (w0+q < TILE_WIDTH)) && ((h+p < H_out) && (w+q < W_out)) ) {
              acc += X_shared[h0+p][w0+q] * kC4d(m,c,p,q);
            }
            else {
              acc += x4d(n, c, h+p, w+q) * kC4d(m,c,p,q);
            }
            //acc += x4d(n, c, h+p, w+q) * kC4d(m,c,p,q);
          }
        }
      __syncthreads();
    }
    y4d(n, m, h, w) = acc;
  }

#undef y4d
#undef x4d
#undef kC4d
}

// __global__ void conv_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
// {
//     /*
//     Function paramter definitions:
//     y - output
//     x - input
//     k - kernel
//     B - batch_size (number of images in x)
//     M - number of output feature maps
//     C - number of input feature maps
//     H - input height dimension
//     W - input width dimension
//     K - kernel height and width (K x K)
//     */
//
//     const int H_out = H - K + 1;
//     const int W_out = W - K + 1;
//
// #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define kC4d(i3, i2, i1, i0) Kc[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
//     // Insert your GPU convolution kernel code here
//     int n, m, h0, w0, h_base, w_base, h, w;
//     int X_tile_width = TILE_WIDTH + K - 1; // K=7, tile_width = 16
//     __shared__ float X_shared[TILE_WIDTH + 6][TILE_WIDTH + 6];
//
//
//     int W_grid = ceil(W_out*1.0 / TILE_WIDTH);  //# of tiles in width
//     n = blockIdx.x; //current channel
//     m = blockIdx.y; //current ouput feature map
//     h0 = threadIdx.x; //current height of pixel of the input tile
//     w0 = threadIdx.y; //current width of pixel of the input tile
//     h_base = (blockIdx.z / W_grid)*TILE_WIDTH; // vertical base out data index for the block
//     w_base = (blockIdx.z % W_grid)*TILE_WIDTH; // horizontal base out data index for the block
//     h = h_base + h0;
//     w = w_base + w0;
//
//     float acc = 0.0;
//     int c, p, q;
//     if (h < H_out && w < W_out) {
//       for (c = 0; c < C; c++) {
//
//         //write into shared memory
//         for (int i = h; i < h_base + X_tile_width; i+= TILE_WIDTH) {
//           for (int j = w; j < w_base + X_tile_width; j+= TILE_WIDTH) {
//             if (i < H && j < W){
//               X_shared[i - h_base][j - w_base] = x4d(n, c, i, j);
//             }
//             else {
//               X_shared[i - h_base][j - w_base] = 0.0f;
//             }
//           }
//         }
//         __syncthreads();
//
//         for (p = 0; p < K; p++) {
//           for (q = 0; q < K; q++){
//             acc += X_shared[h0+p][w0+q] * kC4d(m,c,p,q);
//           }
//         }
//         __syncthreads();
//       }
//       y4d(n, m, h, w) = acc;
//     }
//
// #undef y4d
// #undef x4d
// #undef kC4d
// }

__host__ void GPUInterface::conv_forward_gpu(float *host_y, const float *host_x, const float *host_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Declare relevant device pointers
    float* device_y;
    float* device_x;
    float* device_k;

    //get_device_properties();

    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int inputLength = B*C*H*W;
    int outputLength = B*M*H_out*W_out;
    int kernelLength = M*C*K*K;

    // std::cout<< "M: "<<M<<"\n";
    // std::cout<< "C: "<<C<<"\n";
    // std::cout<< "K: "<<K<<"\n";
     //std::cout<< "H: "<<H<<"\n";
     //std::cout<< "W: "<<W<<"\n";


    cudaMalloc((void**) &device_y, outputLength*sizeof(float));
    cudaMalloc((void**) &device_x, inputLength*sizeof(float));
    //cudaMalloc((void**) &device_k, kernelLength*sizeof(float));

    cudaMemcpy(device_x, host_x, inputLength*sizeof(float), cudaMemcpyHostToDevice);

    //cudaMemcpy(device_k, host_k, kernelLength*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Kc, host_k, 7*7*4*16*sizeof(float));

    // Set the kernel dimensions and call the kernel

    int W_grid = ceil(W_out*1.0 / TILE_WIDTH);  //# of tiles in width
    int H_grid = ceil(H_out*1.0 / TILE_WIDTH);  //# of titls in height
    int Z = H_grid * W_grid;                    //total number of tiles
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);   //thread block size
    //dim3 blockDim(TILE_WIDTH + K - 1, TILE_WIDTH + K - 1, 1);
    dim3 DimGrid(B, M, Z);  //batch_size, # of output feature maps, total number of tiles
    conv_forward_kernel<<<DimGrid, DimBlock>>>(device_y, device_x, B, M, C, H, W, K);

    //float shmem_size = sizeof(float)*((TILE_WIDTH + K - 1)*(TILE_WIDTH + K - 1)+ K*K);
    //conv_forward_kernel<<<gridDim, blockDim, shmem_size>>>(device_y, device_x, device_k, B, M, C, H, W, K);


    // Copy the output back to host
    cudaMemcpy(host_y, device_y, B*M*(H-K+1)*(W-K+1)*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_y);
    cudaFree(device_x);

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
