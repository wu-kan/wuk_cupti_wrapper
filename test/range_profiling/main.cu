#include "cupti_wrapper.h"
#include "helper_cupti.h"
#include <cuda.h>

template <typename T> __global__ void stupid_kernel(T *x, size_t n) {
  for (size_t i = blockDim.x * (size_t)blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    x[i] += 1;
  }
}

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

int main(int argc, char *argv[]) {
  CUdevice cuDevice;
  CUcontext cuContext;
  DRIVER_API_CALL(cuInit(0));
  DRIVER_API_CALL(cuDeviceGet(&cuDevice, 0));
  DRIVER_API_CALL(cuDevicePrimaryCtxRetain(&cuContext, cuDevice));
  DRIVER_API_CALL(cuCtxPushCurrent(cuContext));

  // Set up the workload
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
    std::vector<std::string> metricNames = {"sm__cycles_elapsed.sum",
                                            "sm__cycles_active.sum"};
    do {
      wuk::CuProfiler p(metricNames);
      p.ProfileKernels("Workload", reset, run);
      std::string res = wuk::CuProfiler::res_to_json(p.MetricValues());
      std::fprintf(stdout, "%s", res.c_str());
    } while (0);
  } while (0);

  DRIVER_API_CALL(cuCtxPopCurrent(&cuContext));
  DRIVER_API_CALL(cuDevicePrimaryCtxRelease(cuDevice));
  return 0;
}
