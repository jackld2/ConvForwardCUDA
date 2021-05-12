#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

const int constMemSize = 7*7*4*16;
__constant__ float Kc[constMemSize];

__global__ void unroll_kernel(float *x_unroll, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
  // int H_out = H - K + 1;
  // int W_out = W - K + 1;
  // for (int b = 0; b < B; ++b) { // for each image in the batch
  //   for (int c = 0; c < C; ++c) { // for each input channel
  //     int w_base = c * (K*K); //find the beginning of the unrolled section in column
  //     for (int p = 0; p < K; ++p) {
  //       for (int q = 0; q < K; ++q) { // loop over all positions of convolution mask
  //         for (int h = 0; h < H_out; ++h) {
  //           for (int w = 0; w < W_out; ++w) { // for each output feature map element
  //             int h_unroll =
  //
  //           }
  //         }
  //       }
  //     }
  //   }
  //
  // }
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
  int c, s, h_out, w_out, h_unroll, w_base, p, q;
  int t = blockIdx.x * 1024 + threadIdx.x;
  int H_out = H-K+1;
  int W_out = W-K+1;
  int W_unroll = H_out * W_out;

  if (t < C * W_unroll) {
    c = t / W_unroll;
    s = t % W_unroll;
    h_out = s / W_out;
    w_out = s % W_out;
    h_unroll = h_out * W_out + w_out;
    w_base = c*K*K;
    for (p = 0; p < K; p++) {
      for (q = 0; q < K; q++) {
        w_unroll = w_base + p * K + q;
        X_unroll[]
      }
    }
  }

#undef x4d
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
//         for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
//           for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
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
    float* device_x_unroll;
    float* host_x_unroll;


    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;


    int num_threads = C*H_out*W_out;
    int num_blocks = ceil((1.0*C*H_out*W_out) / 1024);

    int inputLength = B*C*H*W;
    int outputLength = B*M*H_out*W_out;
    int kernelLength = M*C*K*K;
    int unrollLength = C*K*K*H_out*W_out;

    host_x_unroll = (float *)malloc(unrollLength * sizeof(float));

    cudaMalloc((void**) &device_y, outputLength*sizeof(float));
    cudaMalloc((void**) &device_x, inputLength*sizeof(float));
    cudaMalloc((void**) &device_x_unroll, unrollLength*sizeof(float));

    //cudaMalloc((void**) &device_k, kernelLength*sizeof(float));

    cudaMemcpy(device_x, host_x, inputLength*sizeof(float), cudaMemcpyHostToDevice);

    //cudaMemcpy(device_k, host_k, kernelLength*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(Kc, host_k, 7*7*4*16*sizeof(float));

    // Set the kernel dimensions and call the kernel

    int W_grid = ceil(W_out*1.0 / TILE_WIDTH);  //# of tiles in width
    int H_grid = ceil(H_out*1.0 / TILE_WIDTH);  //# of titls in height
    //int Z = H_grid * W_grid;
    dim3 DimBlock(1024, 1, 1);   //thread block size
    dim3 DimGrid(num_blocks, 1, 1);  //batch_size, # of output feature maps, total number of tiles


    unroll_kernel<<<DimGrid, DimBlock>>>(device_y, device_x, B, M, C, H, W, K);

    // Copy the output back to host
    cudaMemcpy(host_y, device_y, B*M*(H-K+1)*(W-K+1)*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    free(host_x_unroll);
    cudaFree(device_y);
    cudaFree(device_x);
    cudaFree(device_x_unroll);

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
