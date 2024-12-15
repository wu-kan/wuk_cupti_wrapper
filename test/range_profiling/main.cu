/*
 *  Copyright 2024 NVIDIA Corporation. All rights reserved
 */

#include <atomic>
#include <chrono>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <thread>

#ifdef _WIN32
#define strdup _strdup
#endif

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

#include "range_profiling.h"

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

struct ParsedArgs {
  int deviceIndex = 0;
  std::string rangeMode = "auto";
  std::string replayMode = "user";
  uint64_t maxRange = 20;
  std::vector<const char *> metrics = {"sm__cycles_elapsed.sum",
                                       "sm__cycles_active.sum"};
};

ParsedArgs parseArgs(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  ParsedArgs args = parseArgs(argc, argv);
  DRIVER_API_CALL(cuInit(0));

  printf("Starting Range Profiling\n");

  // Get the current ctx for the device
  CUdevice cuDevice;
  DRIVER_API_CALL(cuDeviceGet(&cuDevice, args.deviceIndex));

  int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
  DRIVER_API_CALL(cuDeviceGetAttribute(
      &computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      cuDevice));
  DRIVER_API_CALL(cuDeviceGetAttribute(
      &computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      cuDevice));
  printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor,
         computeCapabilityMinor);

  if (computeCapabilityMajor < 7) {
    std::cerr << "Range Profiling is supported only on devices with compute "
                 "capability 7.0 and above"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  RangeProfilerConfig config;
  config.maxNumOfRanges = args.maxRange;
  config.minNestingLevel = 1;
  config.numOfNestingLevel = args.rangeMode == "user" ? 2 : 1;

  CuptiProfilerHostPtr pCuptiProfilerHost =
      std::make_shared<CuptiProfilerHost>();

  // Create a context
  CUcontext cuContext;
  DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));
  RangeProfilerTargetPtr pRangeProfilerTarget =
      std::make_shared<RangeProfilerTarget>(cuContext, config);

  // Get chip name
  std::string chipName;
  CUPTI_API_CALL(RangeProfilerTarget::GetChipName(cuDevice, chipName));

  // Get Counter availability image
  std::vector<uint8_t> counterAvailabilityImage;
  CUPTI_API_CALL(RangeProfilerTarget::GetCounterAvailabilityImage(
      cuContext, counterAvailabilityImage));

  // Create config image
  std::vector<uint8_t> configImage;
  pCuptiProfilerHost->SetUp(chipName, counterAvailabilityImage);
  CUPTI_API_CALL(
      pCuptiProfilerHost->CreateConfigImage(args.metrics, configImage));

  // Set up the workload
  do {
    StupidWordload<float> workload;
    workload.reset();

    // Enable Range profiler
    CUPTI_API_CALL(pRangeProfilerTarget->EnableRangeProfiler());

    // Create CounterData Image
    std::vector<uint8_t> counterDataImage;
    CUPTI_API_CALL(pRangeProfilerTarget->CreateCounterDataImage(
        args.metrics, counterDataImage));

    // Set range profiler configuration
    printf("Range Mode: %s\n", args.rangeMode.c_str());
    printf("Replay Mode: %s\n", args.replayMode.c_str());
    CUPTI_API_CALL(pRangeProfilerTarget->SetConfig(
        args.rangeMode == "auto" ? CUPTI_AutoRange : CUPTI_UserRange,
        args.replayMode == "kernel" ? CUPTI_KernelReplay : CUPTI_UserReplay,
        configImage, counterDataImage));

    do {
      // Start Range Profiling
      CUPTI_API_CALL(pRangeProfilerTarget->StartRangeProfiler());

      {
        // Push Range (Level 1)
        CUPTI_API_CALL(pRangeProfilerTarget->PushRange("Workload"));

        // Launch CUDA workload
        workload.run_async();
        workload.sync();

        {
          // Push Range (Level 2)
          CUPTI_API_CALL(pRangeProfilerTarget->PushRange("Nested workload"));

          workload.run_async();
          workload.sync();

          // Pop Range (Level 2)
          CUPTI_API_CALL(pRangeProfilerTarget->PopRange());
        }

        // Pop Range (Level 1)
        CUPTI_API_CALL(pRangeProfilerTarget->PopRange());
      }

      workload.run_async();
      workload.sync();

      // Stop Range Profiling
      CUPTI_API_CALL(pRangeProfilerTarget->StopRangeProfiler());
    } while (!pRangeProfilerTarget->IsAllPassSubmitted());

    // Get Profiler Data
    CUPTI_API_CALL(pRangeProfilerTarget->DecodeCounterData());

    // Evaluate the results
    size_t numRanges = 0;
    CUPTI_API_CALL(
        pCuptiProfilerHost->GetNumOfRanges(counterDataImage, numRanges));
    for (size_t rangeIndex = 0; rangeIndex < numRanges; ++rangeIndex) {
      CUPTI_API_CALL(pCuptiProfilerHost->EvaluateCounterData(
          rangeIndex, args.metrics, counterDataImage));
    }

    pCuptiProfilerHost->PrintProfilerRanges();

    // Clean up
    CUPTI_API_CALL(pRangeProfilerTarget->DisableRangeProfiler());
    pCuptiProfilerHost->TearDown();
  } while (0);

  DRIVER_API_CALL(cuCtxDestroy(cuContext));
  return 0;
}

void PrintHelp() {
  printf("Usage:\n");
  printf("  Range Profiling:\n");
  printf("    ./range_profiling [args]\n");
  printf("        --device/-d <deviceIndex> : Device index to run the range "
         "profiling\n");
  printf(
      "        --range/-r <auto/user> : Range profiling mode. auto: ranges are "
      "defined around each kernel user: user defined ranges (Push/Pop API)\n");
  printf("        --replay/-e <kernel/user> : Replay mode needed for "
         "multi-pass metrics. kernel: replay will be done by CUPTI internally "
         "user: replay done explicitly by user\n");
  printf("        --maxNumRanges/-n <maximum number of ranges stored in "
         "counterdata> : Maximum number of ranges stored in counterdata\n");
  printf("        --metrics/-m <metric1,metric2,...> : List of metrics to be "
         "collected\n");
}

ParsedArgs parseArgs(int argc, char *argv[]) {
  ParsedArgs args;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--device" || arg == "-d") {
      args.deviceIndex = std::stoi(argv[++i]);
    } else if (arg == "--range" || arg == "-r") {
      args.rangeMode = argv[++i];
    } else if (arg == "--replay" || arg == "-e") {
      args.replayMode = argv[++i];
    } else if (arg == "--maxNumRanges" || arg == "-n") {
      args.maxRange = std::stoull(argv[++i]);
    } else if (arg == "--metrics" || arg == "-m") {
      std::stringstream ss(argv[++i]);
      std::string metric;
      args.metrics.clear();
      while (std::getline(ss, metric, ',')) {
        args.metrics.push_back(strdup(metric.c_str()));
      }
    } else if (arg == "--help" || arg == "-h") {
      PrintHelp();
      exit(EXIT_SUCCESS);
    } else {
      fprintf(stderr, "Invalid argument: %s\n", arg.c_str());
      PrintHelp();
      exit(EXIT_FAILURE);
    }
  }
  return args;
}
