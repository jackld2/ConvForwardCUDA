__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

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

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int n, m, h0, w0, h_base, w_base, h, w;
    int X_tile_width = TILE_WIDTH + K - 1; // K=7, tile_width = 16
    __shared__ float X_shared[TILE_WIDTH+6][TILE_WIDTH+6];
    __shared__ float K_shared[7][7];

    //extern __shared__ float shmem[];//input tile and filter
    //float* X_shared = &shmem[0];  //pointer to input tile
    //float* K_shared = &shmem[X_tile_width * X_tile_width];  //pointer to filter
    int W_grid = ceil(W_out*1.0 / TILE_WIDTH);  //# of tiles in width
    n = blockIdx.x; //current channel
    m = blockIdx.y; //current ouput feature map
    w0 = threadIdx.x; //current height of pixel of the input tile
    h0 = threadIdx.y; //current width of pixel of the input tile
    //h_base = (blockIdx.z / W_grid) * X_tile_width; // vertical base out data index for the block
    //w_base = (blockIdx.z % W_grid) * X_tile_width; // horizontal base out data index for the block
    h_base = (blockIdx.z / W_grid) * TILE_WIDTH; // vertical base out data index for the block
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH; // horizontal base out data index for the block
    h = h_base + h0;
    w = w_base + w0;

    float acc = 0.;
    int c, p, q;
    if(h < H_out && w < W_out){
    for (c = 0; c < C; c++) { // sum over all input channels
                              // load weights for W [m, c,..],
                              // h0 and w0 used as shorthand for threadIdx.x
                              // and threadIdx.y

       // load tile from X[n, c,…] into shared memory

       if (( h0 < K) && ( w0 < K)){
         K_shared[h0][w0]= k4d(m, c, h0, w0);
       }
       __syncthreads();

       for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
         for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH){
         //   if(i < H_out && j < W_out){
         //   X_shared[i - h_base][j - w_base] = x4d(n, c, h, w);
         // }else{
         //   X_shared[i - h_base][j - w_base] = 0.0f;
         // }
         if(i < H && j < W){
         X_shared[i - h_base][j - w_base] = x4d(n, c, i, j);
       }else{
         X_shared[i - h_base][j - w_base] = 0.0f;
       }

         }
       }
          __syncthreads();

       for (p = 0; p < K; p++) {
         for (q = 0; q < K; q++){
           acc = acc + X_shared[h0+p][w0+q] * K_shared[p][q];
             //acc = acc + X_shared[(h + p)*TILE_WIDTH+w+q] * K_shared[p*K+q];
         }
       }

       __syncthreads();
    }
     y4d(n, m, h, w) = acc;
   }
#undef y4d
#undef x4d
#undef k4d
}
