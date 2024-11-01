#include <cuda.h>
#include <wuk/cupti_wrapper.hh>

template <typename T> __global__ void stupid_kernel(T *x, size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    x[i] = 0;
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

template <typename T> struct StupidTester {
  T *x;
  size_t n;
  CUstream s;
  StupidTester(size_t num = 1 << 20) : n(num) {
    CUDA_SAFE_CALL(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
    CUDA_SAFE_CALL(cuMemAlloc((CUdeviceptr *)&x, n * sizeof(T)));
  }
  void reset() { CUDA_SAFE_CALL(cuStreamSynchronize(s)); }
  void run() {
    stupid_kernel<T><<<(n + 255) / 256, 256, 0, s>>>(x, n);
    CUDA_SAFE_CALL(cuStreamSynchronize(s));
  }
  ~StupidTester() {
    CUDA_SAFE_CALL(cuMemFree((CUdeviceptr)x));
    CUDA_SAFE_CALL(cuStreamDestroy(s));
  }
};

int main() {
  CUdevice device;
  CUcontext context;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&device, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, device));

  // https://docs.nvidia.com/cupti/main/main.html#metrics-mapping-table
  std::vector<std::string> metricNames{"sm__cycles_elapsed.sum",
                                       "sm__cycles_active.sum"};
  wuk::Profiler::init();
  wuk::Profiler p(metricNames);
  StupidTester<int> kernel0;
  StupidTester<float> kernel1;
  auto reset = [&] {
    kernel0.reset();
    kernel1.reset();
  };
  auto run = [&] {
    kernel0.run();
    kernel1.run();
  };
  p.ProfileKernels("RangeName", reset, run);
  auto res = p.MetricValuesToJSON(metricNames);
  std::fprintf(stdout, "%s", res.c_str());
  wuk::Profiler::deinit();
  // CUDA_SAFE_CALL(cuCtxDestory(context));
}