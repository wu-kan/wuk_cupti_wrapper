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

#define DRIVER_API_CALL(x)                                                     \
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
    DRIVER_API_CALL(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
    DRIVER_API_CALL(cuMemAllocAsync(&x, sizeof(T) * n, s));
  }
  void sync() { DRIVER_API_CALL(cuStreamSynchronize(s)); }
  void reset() {
    DRIVER_API_CALL(cuMemsetD8Async(x, 0, sizeof(T) * n, s));
    sync();
  }
  void run_async() {
    stupid_kernel<T><<<(n + 255) / 256, 256, 0, s>>>((T *)x, n);
  }
  ~StupidWordload() {
    DRIVER_API_CALL(cuMemFreeAsync(x, s));
    sync();
    DRIVER_API_CALL(cuStreamDestroy(s));
  }
};

int main() {
  CUdevice device;
  CUcontext ctx;
  DRIVER_API_CALL(cuInit(0));
  DRIVER_API_CALL(cuDeviceGet(&device, 0));
  DRIVER_API_CALL(cuDevicePrimaryCtxRetain(&ctx, device));
  DRIVER_API_CALL(cuCtxPushCurrent(ctx));
  do {
    StupidWordload<int> workload0;
    StupidWordload<float> workload1;
    auto reset = [&] {
      workload0.reset();
      workload1.reset();
    };
    auto run = [&] {
      workload0.run_async();
      workload1.run_async();
      workload0.sync();
      workload1.sync();
    };
    do {
      // https://docs.nvidia.com/cupti/main/main.html#metrics-mapping-table
      std::vector<std::string> metricNames{"sm__cycles_elapsed.sum",
                                           "sm__cycles_active.sum"};
      wuk::CuProfiler p(metricNames);
      p.ProfileKernels("range_name", reset, run);
      std::string res = wuk::CuProfiler::res_to_json(p.MetricValues());
      std::fprintf(stdout, "%s", res.c_str());
    } while (0);
  } while (0);
  DRIVER_API_CALL(cuCtxPopCurrent(&ctx));
  DRIVER_API_CALL(cuDevicePrimaryCtxRelease(device));
}