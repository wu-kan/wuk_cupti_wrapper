#include "wuk/cupti_wrapper.hh"

// Make use of example code wrappers for NVPW calls
#include "Metric.h"
#include "Parser.h"
#include "ScopeExit.h"
#include "Utils.h"

// CUPTI headers
#include <cupti_profiler_target.h>
#include <cupti_target.h>

#include "helper_cupti.h"

#include <nvperf_cuda_host.h>
#include <nvperf_host.h>
#include <nvperf_target.h>

// CUDA headers
#include <cuda.h>
#include <driver_types.h>

// Standard STL headers
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#define EXIT_IF_NVPW_ERROR(retval, actual)                                     \
  do {                                                                         \
    NVPA_Status status = actual;                                               \
    if (NVPA_STATUS_SUCCESS != status) {                                       \
      std::fprintf(stderr, "FAILED: %s with error %s\n", #actual,              \
                   NV::Metric::Utils::GetNVPWResultString(status));            \
      std::exit(-1);                                                           \
    }                                                                          \
  } while (0)

namespace wuk {

CuProfiler::ProfilingConfig::ProfilingConfig() {}

// Call any needed initialization routines for host or target.
void CuProfiler::init() {
  // CUPTI Profiler API initialization.
  CUpti_Profiler_Initialize_Params profilerInitializeParams = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

  // NVPW required initialization.
  NVPW_InitializeHost_Params initializeHostParams = {
      NVPW_InitializeHost_Params_STRUCT_SIZE};
  NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));
}

void CuProfiler::deinit() {
  CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
      CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
}

CuProfiler::CuProfiler(std::vector<std::string> const &MetricNames,
                       const CuProfiler::ProfilingConfig &cfg)
    : config(cfg) {

  CuProfiler &deviceData = *this;

  DRIVER_API_CALL(cuCtxGetCurrent(&ctx));

  // Get size of counterAvailabilityImage - in first pass,
  // GetCounterAvailability return size needed for data.
  CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {
      CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};

  getCounterAvailabilityParams.ctx = ctx;
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

  // Allocate sized counterAvailabilityImage.
  std::vector<uint8_t> counterAvailabilityImage;
  counterAvailabilityImage.resize(
      getCounterAvailabilityParams.counterAvailabilityImageSize);

  // Initialize counterAvailabilityImage.
  getCounterAvailabilityParams.pCounterAvailabilityImage =
      counterAvailabilityImage.data();
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

  // Get chip name for the CUDA device.
  CUpti_Device_GetChipName_Params getChipNameParams = {
      CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  CUdevice device;
  DRIVER_API_CALL(cuCtxGetDevice(&device));
  getChipNameParams.deviceIndex = device;
  CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
  deviceData.pChipName = std::string(getChipNameParams.pChipName);

  // Fill in configImage - can be run on host or target.
  if (!NV::Metric::Config::GetConfigImage(deviceData.pChipName.c_str(),
                                          MetricNames, deviceData.configImage,
                                          counterAvailabilityImage.data())) {
    std::fprintf(stderr, "Failed to create configImage\n");
    std::exit(EXIT_FAILURE);
  }

  // Fill in counterDataPrefixImage - can be run on host or target.
  if (!NV::Metric::Config::GetCounterDataPrefixImage(
          deviceData.pChipName.c_str(), MetricNames,
          deviceData.counterDataPrefixImage, counterAvailabilityImage.data())) {
    std::fprintf(stderr, "Failed to create counterDataPrefixImage\n");
    std::exit(EXIT_FAILURE);
  }

  // Record counterDataPrefixImage info and other options for sizing the
  // counterDataImage.
  CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
  counterDataImageOptions.pCounterDataPrefix =
      deviceData.counterDataPrefixImage.data();
  counterDataImageOptions.counterDataPrefixSize =
      deviceData.counterDataPrefixImage.size();
  counterDataImageOptions.maxNumRanges = deviceData.config.maxNumRanges;
  counterDataImageOptions.maxNumRangeTreeNodes = deviceData.config.maxNumRanges;
  counterDataImageOptions.maxRangeNameLength =
      deviceData.config.maxRangeNameLength;

  // Calculate size of counterDataImage based on counterDataPrefixImage and
  // options.
  CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
      CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
  calculateSizeParams.pOptions = &counterDataImageOptions;
  calculateSizeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  CUPTI_API_CALL(
      cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));
  // Create counterDataImage
  deviceData.counterDataImage.resize(calculateSizeParams.counterDataImageSize);

  // Initialize counterDataImage.
  CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {
      CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  initializeParams.pOptions = &counterDataImageOptions;
  initializeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  initializeParams.counterDataImageSize = deviceData.counterDataImage.size();
  initializeParams.pCounterDataImage = deviceData.counterDataImage.data();
  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

  // Calculate scratchBuffer size based on counterDataImage size and
  // counterDataImage.
  CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
      scratchBufferSizeParams = {
          CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  scratchBufferSizeParams.counterDataImageSize =
      deviceData.counterDataImage.size();
  scratchBufferSizeParams.pCounterDataImage =
      deviceData.counterDataImage.data();
  CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(
      &scratchBufferSizeParams));
  // Create counterDataScratchBuffer.
  deviceData.counterDataScratchBufferImage.resize(
      scratchBufferSizeParams.counterDataScratchBufferSize);

  // Initialize counterDataScratchBuffer.
  CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
      initScratchBufferParams = {
          CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
  initScratchBufferParams.counterDataImageSize =
      deviceData.counterDataImage.size();
  initScratchBufferParams.pCounterDataImage =
      deviceData.counterDataImage.data();
  initScratchBufferParams.counterDataScratchBufferSize =
      deviceData.counterDataScratchBufferImage.size();
  ;
  initScratchBufferParams.pCounterDataScratchBuffer =
      deviceData.counterDataScratchBufferImage.data();
  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(
      &initScratchBufferParams));
}

// Wrapper which will launch numKernel kernel calls on a single device.
// The device streams vector is used to control which stream each call is made
// on. If 'serial' is non-zero, the device streams are ignored and instead the
// default stream is used.
void CuProfiler::ProfileKernels(char const *const RangeName,
                                const std::function<void()> &reset,
                                const std::function<void()> &kernel) {

  // Start a session.
  do {
    CUpti_Profiler_BeginSession_Params beginSessionParams = {
        CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
    beginSessionParams.counterDataImageSize = counterDataImage.size();
    beginSessionParams.pCounterDataImage = counterDataImage.data();
    beginSessionParams.counterDataScratchBufferSize =
        counterDataScratchBufferImage.size();
    beginSessionParams.pCounterDataScratchBuffer =
        counterDataScratchBufferImage.data();
    beginSessionParams.ctx = ctx;
    beginSessionParams.maxLaunchesPerPass = config.maxLaunchesPerPass;
    beginSessionParams.maxRangesPerPass = config.maxRangesPerPass;
    beginSessionParams.pPriv = NULL;
    // CUPTI_AutoRange or CUPTI_UserRange.
    beginSessionParams.range = CUPTI_UserRange;
    // CUPTI_KernelReplay, CUPTI_UserReplay, or CUPTI_ApplicationReplay.
    beginSessionParams.replayMode = CUPTI_UserReplay;
    CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));
  } while (0);

  do {
    CUpti_Profiler_SetConfig_Params setConfigParams = {
        CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
    setConfigParams.pConfig = configImage.data();
    setConfigParams.configSize = configImage.size();
    // Only set for Application Replay mode.
    setConfigParams.passIndex = 0;
    setConfigParams.minNestingLevel = config.minNestingLevels;
    setConfigParams.numNestingLevels = config.numNestingLevels;
    setConfigParams.targetNestingLevel = config.minNestingLevels;
    CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
  } while (0);

  int numPasses = 0;
  bool lastPass = false;
  // Perform multiple passes if needed to provide all configured metrics.
  // Note that in this mode, kernel input data is not restored to initial
  // values before each pass.
  do {
    reset();
    do {
      CUpti_Profiler_BeginPass_Params beginPassParams = {
          CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
      beginPassParams.ctx = ctx;
      CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
      numPasses++;
      CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
          CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
      enableProfilingParams.ctx = ctx;
      CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
    } while (0);
    // Then, time launching same amount of work in separate streams. (or
    // default stream if serial.) cuptiProfilerPushRange and PopRange will
    // serialize the kernel launches, so keep the calls outside the concurrent
    // stream launch loop.
    do {
      CUpti_Profiler_PushRange_Params pushRangeParams = {
          CUpti_Profiler_PushRange_Params_STRUCT_SIZE};
      pushRangeParams.ctx = ctx;
      pushRangeParams.pRangeName = RangeName;
      pushRangeParams.rangeNameLength = strlen(RangeName);
      CUPTI_API_CALL(cuptiProfilerPushRange(&pushRangeParams));
    } while (0);

    kernel();

    do {
      CUpti_Profiler_PopRange_Params popRangeParams = {
          CUpti_Profiler_PopRange_Params_STRUCT_SIZE};
      popRangeParams.ctx = ctx;
      CUPTI_API_CALL(cuptiProfilerPopRange(&popRangeParams));
    } while (0);
    do {
      CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
          CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
      disableProfilingParams.ctx = ctx;
      CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));
    } while (0);
    do {
      CUpti_Profiler_EndPass_Params endPassParams = {
          CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
      endPassParams.ctx = ctx;
      CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
      lastPass = endPassParams.allPassesSubmitted;
    } while (0);
  } while (lastPass == false);

  // Flush is required to ensure data is returned from device when running
  // User Replay mode.
  do {
    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
        CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
    flushCounterDataParams.ctx = ctx;
    CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
    if (flushCounterDataParams.numRangesDropped != 0 ||
        flushCounterDataParams.numTraceBytesDropped) {
      std::fprintf(stderr,
                   "WARNING: %d trace bytes dropped due to full "
                   "TraceBuffer\nWARNING: %d ranges dropped in pass\n",
                   (int)flushCounterDataParams.numTraceBytesDropped,
                   (int)flushCounterDataParams.numRangesDropped);
    }
  } while (0);

  do {
    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
        CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    unsetConfigParams.ctx = ctx;
    CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
  } while (0);
  do {
    CUpti_Profiler_EndSession_Params endSessionParams = {
        CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    endSessionParams.ctx = ctx;
    CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
  } while (0);
}

std::string
CuProfiler::MetricValuesToJSON(const std::vector<std::string> &metricNames,
                               const uint8_t *pCounterAvailabilityImage) const {
  std::string ret = "[", chipName = pChipName;
  if (!counterDataImage.size()) {
    std::fprintf(stderr, "Counter Data Image is empty!\n");
    std::exit(-1);
  }

  NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params
      calculateScratchBufferSizeParam = {
          NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  calculateScratchBufferSizeParam.pChipName = chipName.c_str();
  calculateScratchBufferSizeParam.pCounterAvailabilityImage =
      pCounterAvailabilityImage;
  EXIT_IF_NVPW_ERROR(false,
                     NVPW_CUDA_MetricsEvaluator_CalculateScratchBufferSize(
                         &calculateScratchBufferSizeParam));

  std::vector<uint8_t> scratchBuffer(
      calculateScratchBufferSizeParam.scratchBufferSize);
  NVPW_CUDA_MetricsEvaluator_Initialize_Params metricEvaluatorInitializeParams =
      {NVPW_CUDA_MetricsEvaluator_Initialize_Params_STRUCT_SIZE};
  metricEvaluatorInitializeParams.scratchBufferSize = scratchBuffer.size();
  metricEvaluatorInitializeParams.pScratchBuffer = scratchBuffer.data();
  metricEvaluatorInitializeParams.pChipName = chipName.c_str();
  metricEvaluatorInitializeParams.pCounterAvailabilityImage =
      pCounterAvailabilityImage;
  metricEvaluatorInitializeParams.pCounterDataImage = counterDataImage.data();
  metricEvaluatorInitializeParams.counterDataImageSize =
      counterDataImage.size();
  EXIT_IF_NVPW_ERROR(false, NVPW_CUDA_MetricsEvaluator_Initialize(
                                &metricEvaluatorInitializeParams));
  NVPW_MetricsEvaluator *metricEvaluator =
      metricEvaluatorInitializeParams.pMetricsEvaluator;

  NVPW_CounterData_GetNumRanges_Params getNumRangesParams = {
      NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE};
  getNumRangesParams.pCounterDataImage = counterDataImage.data();
  EXIT_IF_NVPW_ERROR(false, NVPW_CounterData_GetNumRanges(&getNumRangesParams));

  for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges;
       ++rangeIndex) {
    NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = {
        NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE};
    getRangeDescParams.pCounterDataImage = counterDataImage.data();
    getRangeDescParams.rangeIndex = rangeIndex;
    EXIT_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(
                                  &getRangeDescParams));
    std::vector<const char *> descriptionPtrs(
        getRangeDescParams.numDescriptions);
    getRangeDescParams.ppDescriptions = descriptionPtrs.data();
    EXIT_IF_NVPW_ERROR(false, NVPW_Profiler_CounterData_GetRangeDescriptions(
                                  &getRangeDescParams));

    std::string rangeName;
    for (size_t descriptionIndex = 0;
         descriptionIndex < getRangeDescParams.numDescriptions;
         ++descriptionIndex) {
      if (descriptionIndex) {
        rangeName += "/";
      }
      rangeName += descriptionPtrs[descriptionIndex];
    }

    do {
      NVPW_MetricsEvaluator_SetDeviceAttributes_Params setDeviceAttribParams = {
          NVPW_MetricsEvaluator_SetDeviceAttributes_Params_STRUCT_SIZE};
      setDeviceAttribParams.pMetricsEvaluator = metricEvaluator;
      setDeviceAttribParams.pCounterDataImage = counterDataImage.data();
      setDeviceAttribParams.counterDataImageSize = counterDataImage.size();
      EXIT_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_SetDeviceAttributes(
                                    &setDeviceAttribParams));
    } while (0);
    ret += "{\"Metrics\": {";
    for (std::string metricName : metricNames) {
      std::string reqName;
      bool isolated = true;
      bool keepInstances = true;
      NV::Metric::Parser::ParseMetricNameString(metricName, &reqName, &isolated,
                                                &keepInstances);
      NVPW_MetricEvalRequest metricEvalRequest;
      do {
        NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params
            convertMetricToEvalRequest = {
                NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest_Params_STRUCT_SIZE};
        convertMetricToEvalRequest.pMetricsEvaluator = metricEvaluator;
        convertMetricToEvalRequest.pMetricName = reqName.c_str();
        convertMetricToEvalRequest.pMetricEvalRequest = &metricEvalRequest;
        convertMetricToEvalRequest.metricEvalRequestStructSize =
            NVPW_MetricEvalRequest_STRUCT_SIZE;
        EXIT_IF_NVPW_ERROR(
            false, NVPW_MetricsEvaluator_ConvertMetricNameToMetricEvalRequest(
                       &convertMetricToEvalRequest));
      } while (0);
      double metricValue;
      NVPW_MetricsEvaluator_EvaluateToGpuValues_Params
          evaluateToGpuValuesParams = {
              NVPW_MetricsEvaluator_EvaluateToGpuValues_Params_STRUCT_SIZE};
      evaluateToGpuValuesParams.pMetricsEvaluator = metricEvaluator;
      evaluateToGpuValuesParams.pMetricEvalRequests = &metricEvalRequest;
      evaluateToGpuValuesParams.numMetricEvalRequests = 1;
      evaluateToGpuValuesParams.metricEvalRequestStructSize =
          NVPW_MetricEvalRequest_STRUCT_SIZE;
      evaluateToGpuValuesParams.metricEvalRequestStrideSize =
          sizeof(NVPW_MetricEvalRequest);
      evaluateToGpuValuesParams.pCounterDataImage = counterDataImage.data();
      evaluateToGpuValuesParams.counterDataImageSize = counterDataImage.size();
      evaluateToGpuValuesParams.rangeIndex = rangeIndex;
      evaluateToGpuValuesParams.isolated = true;
      evaluateToGpuValuesParams.pMetricValues = &metricValue;
      EXIT_IF_NVPW_ERROR(false, NVPW_MetricsEvaluator_EvaluateToGpuValues(
                                    &evaluateToGpuValuesParams));

      ret +=
          "\"" + metricName + "\"" + ": " + std::to_string(metricValue) + ",";
    }
    if (ret.back() == ',')
      ret.pop_back();
    ret += "}, \"RangeName\": \"" + rangeName + "\"" + "},";
  }
  if (ret.back() == ',')
    ret.pop_back();
  ret += "]";

  NVPW_MetricsEvaluator_Destroy_Params metricEvaluatorDestroyParams = {
      NVPW_MetricsEvaluator_Destroy_Params_STRUCT_SIZE};
  metricEvaluatorDestroyParams.pMetricsEvaluator = metricEvaluator;
  EXIT_IF_NVPW_ERROR(
      false, NVPW_MetricsEvaluator_Destroy(&metricEvaluatorDestroyParams));
  return ret;
}
} // namespace wuk
