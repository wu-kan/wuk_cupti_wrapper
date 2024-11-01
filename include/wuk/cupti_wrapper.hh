#pragma once
#include <cuda.h>
#include <functional>
#include <string>
#include <vector>

namespace wuk {

// Per-device configuration, buffers, stream and device information, and device
// pointers.
struct Profiler {
  // Each device (or each context) needs its own CUPTI profiling config.
  CUcontext ctx;
  // Profiling data images.
  std::vector<uint8_t> counterDataImage;
  std::vector<uint8_t> counterDataPrefixImage;
  std::vector<uint8_t> counterDataScratchBufferImage;
  std::vector<uint8_t> configImage;

  // String name of target compute device, needed for NVPW calls.
  char const *pChipName;

  struct ProfilingConfig {

    // Maximum number of Ranges that may be encountered in this Session. (Nested
    // Ranges are multiplicative.)
    // Device 0 has max of 3 passes; other devices only run one pass in this
    // sample code
    // Maximum number of ranges that may be profiled in the current Session
    int maxNumRanges = 1;

    // Maximum number of kernel launches in any Pass in this Session.
    // Must be >= maxRangesPerPass.  Set this to the largest count of kernel
    // launches which may be encountered in any Pass in this Session.
    int maxLaunchesPerPass = 1;

    // Max length including NULL terminator of any range name.
    int maxRangeNameLength = 64;

    // Maximum number of Ranges in any Pass in this Session.
    // Max ranges that can be recorded in any Pass in this Session.
    int maxRangesPerPass = 1;

    // Minimum level to tag a Range within this session, must be >= 1.
    // Must be >= 1, minimum reported nest level for Ranges in this Session.
    int minNestingLevels = 1;

    // Maximum level for nested Ranges within this Session, must be >= 1.
    // Must be >= 1, max height of nested Ranges in this Session.
    int numNestingLevels = 1;

    ProfilingConfig();
  } config;

  Profiler(std::vector<std::string> const &MetricNames,
           const Profiler::ProfilingConfig &cfg = Profiler::ProfilingConfig());
  void ProfileKernels(char const *const RangeName,
                      const std::function<void()> &reset,
                      const std::function<void()> &kernel);
  std::string
  MetricValuesToJSON(const std::vector<std::string> &metricNames,
                     const uint8_t *pCounterAvailabilityImage = nullptr) const;
  static void init();
  static void deinit();
};

}; // namespace wuk