#include <cstdint>
#include <cuda_runtime.h>

// This runs on the GPU - one thread per output pixel
// Takes the BGR grame (from OpenCV), resizes with letterboxing
// normalizes to [0,1], and outputs float NCHW layout
__global__ void preprocess_kernel(const uint8_t *__restrict__ src, // BGR frame from OpenCV, on GPU
                                  float *__restrict__ dst,         // output: float32 NCHW
                                  int   src_w,
                                  int   src_h,
                                  int   dst_w,
                                  int   dst_h,
                                  float scale,
                                  int   pad_x,
                                  int   pad_y,
                                  int   norm_type) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_w || dy >= dst_h)
        return;
    float r = 0.447f, g = 0.447f, b = 0.447f; // gray padding (114/255)

    int sx = dx - pad_x;
    int sy = dy - pad_y;

    if (sx >= 0 && sy >= 0 && sx < (int)(src_w * scale) && sy < (int)(src_h * scale)) {
        int ox = (int)(sx / scale);
        int oy = (int)(sy / scale);

        ox = min(ox, src_w - 1);
        oy = min(oy, src_h - 1);

        int idx = (oy * src_w + ox) * 3;

        b = src[idx + 0] / 255.0f;
        g = src[idx + 1] / 255.0f;
        r = src[idx + 2] / 255.0f;
    }

    // Apply ImageNet normalization if requested
    if (norm_type == 1) {
        r = (r - 0.485f) / 0.229f;
        g = (g - 0.456f) / 0.224f;
        b = (b - 0.406f) / 0.225f;
    }

    // Write to planer NCHW: channel planes laid out sequentially
    int plane = dst_w * dst_h;

    dst[0 * plane + dy * dst_w + dx] = r;
    dst[1 * plane + dy * dst_w + dx] = g;
    dst[2 * plane + dy * dst_w + dx] = b;
}

// Called from C++ code -  this is the launch wrapper
void launch_preprocess(const uint8_t *d_bgr,
                       float         *d_out,
                       int            src_w,
                       int            src_h,
                       int            dst_w,
                       int            dst_h,
                       cudaStream_t   stream,
                       int            norm_type = 0) {
    // Compute letterbox scale and padding
    float scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
    int   new_w = (int)(src_w * scale);
    int   new_h = (int)(src_h * scale);
    int   pad_x = (dst_w - new_w) / 2;
    int   pad_y = (dst_h - new_h) / 2;

    dim3 threads(32, 8);
    dim3 blocks((dst_w + 31) / 32, (dst_h + 7) / 8);
    preprocess_kernel<<<blocks, threads, 0, stream>>>(
        d_bgr, d_out, src_w, src_h, dst_w, dst_h, scale, pad_x, pad_y, norm_type);
}
