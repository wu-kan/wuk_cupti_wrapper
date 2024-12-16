#pragma once
#include <cupti_profiler_host.h>
#include <cupti_profiler_target.h>
#include <cupti_range_profiler.h>
#include <cupti_target.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace wuk {

struct ProfilerRange {
  size_t rangeIndex;
  std::string rangeName;
  std::unordered_map<std::string, double> metricValues;
};

struct RangeProfilerConfig {
  size_t maxNumOfRanges = 20;
  size_t numOfNestingLevel = 1;
  size_t minNestingLevel = 1;
  CUpti_ProfilerRange rangeMode = CUPTI_UserRange;        // CUPTI_AutoRange,
  CUpti_ProfilerReplayMode replayMode = CUPTI_UserReplay; // CUPTI_KernelReplay;
};

class CuptiProfilerHost;

class RangeProfilerTarget;

class CuProfiler {
private:
  std::shared_ptr<CuptiProfilerHost> pCuptiProfilerHost;
  std::shared_ptr<RangeProfilerTarget> pRangeProfilerTarget;
  std::vector<uint8_t> counterAvailabilityImage;
  std::vector<uint8_t> configImage;
  std::vector<uint8_t> counterDataImage;
  std::vector<const char *> metricsList;

public:
  CuProfiler(const std::vector<std::string> &metricsList,
             const RangeProfilerConfig &config = RangeProfilerConfig());
  ~CuProfiler();
  void ProfileKernels(const char *RangeName, const std::function<void()> &reset,
                      const std::function<void()> &run);
  std::vector<ProfilerRange> MetricValues();
  static std::string res_to_json(const std::vector<ProfilerRange> &res);
  static void init();
  static void deinit();
};

} // namespace wuk