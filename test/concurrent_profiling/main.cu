#include <cstddef>
#include <cuda.h>
#include <string>
#include <wuk/cupti_wrapper.hh>

template <typename T> __global__ void stupid_kernel(T *x, size_t n) {
  for (size_t i = blockDim.x * (size_t)blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    x[i] += 1;
  }
}

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      std::fprintf(stderr,                                                     \
                   "{\"File\": "                                               \
                   "\"%s\", \"Line\": %d, \"Error\": \"%s\",\"Code\": %d, "    \
                   "\"Msg\": \"%s\"}\n",                                       \
                   __FILE__, __LINE__, #x, (int)result, msg);                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

template <typename T> class StupidWordload {
private:
  CUdeviceptr x;
  size_t n;
  CUstream s;

public:
  StupidWordload(size_t num = 1 << 20) : n(num) {
    CUDA_SAFE_CALL(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
    CUDA_SAFE_CALL(cuMemAllocAsync(&x, sizeof(T) * n, s));
  }
  void sync() { CUDA_SAFE_CALL(cuStreamSynchronize(s)); }
  void reset() {
    CUDA_SAFE_CALL(cuMemsetD8Async(x, 0, sizeof(T) * n, s));
    sync();
  }
  void run_async() {
    stupid_kernel<T><<<(n + 255) / 256, 256, 0, s>>>((T *)x, n);
  }
  ~StupidWordload() {
    CUDA_SAFE_CALL(cuMemFreeAsync(x, s));
    sync();
    CUDA_SAFE_CALL(cuStreamDestroy(s));
  }
};

int main() {
  CUdevice device;
  CUcontext ctx;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&device, 0));
  CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&ctx, device));
  CUDA_SAFE_CALL(cuCtxPushCurrent(ctx));
  do {
    StupidWordload<int> kernel0;
    StupidWordload<float> kernel1;
    auto reset = [&] {
      kernel0.reset();
      kernel1.reset();
    };
    auto run = [&] {
      kernel0.run_async();
      kernel1.run_async();
      kernel0.sync();
      kernel1.sync();
    };
    wuk::CuProfiler::init();
    do {
      // https://docs.nvidia.com/cupti/main/main.html#metrics-mapping-table
      std::vector<std::string> metricNames{"sm__cycles_elapsed.sum",
                                           "sm__cycles_active.sum"};
#if 0
      wuk::CuProfiler::ProfilingConfig cfg;
      cfg.maxRangeNameLength = 16; // the max length of "range_name"
      wuk::CuProfiler p(metricNames, cfg);
#else
      wuk::CuProfiler p(metricNames);
#endif
      p.ProfileKernels("range_name", reset, run);
      std::string res =
          wuk::CuProfiler::res_to_json(p.MetricValues(metricNames));
      std::fprintf(stdout, "%s", res.c_str());
    } while (0);
    wuk::CuProfiler::deinit();
  } while (0);
  CUDA_SAFE_CALL(cuCtxPopCurrent(&ctx));
  CUDA_SAFE_CALL(cuDevicePrimaryCtxRelease(device));
}