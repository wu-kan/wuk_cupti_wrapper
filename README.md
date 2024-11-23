# wuk_cupti_wrapper

This library provides an simple API to use [CUPTI (the CUDA Profiling Tools Interface)](https://docs.nvidia.com/cupti/index.html). The code is mainly modified from the cupti sample [concurrent_profiling](https://github.com/mmcloughlin/cuptisamples/blob/main/samples/concurrent_profiling/concurrent_profiling.cu).

## Example Usage

For more detials, please refer to [concurrent_profiling sample](./test/concurrent_profiling/main.cu) and the [header](./include/wuk/cupti_wrapper.hh).

```cpp
auto reset = [&] {
  kernel0.reset();
  kernel1.reset();
};
auto run = [&] {
  kernel0.run();
  kernel1.run();
};
wuk::CuProfiler::init();
do {
  // https://docs.nvidia.com/cupti/main/main.html#metrics-mapping-table
  std::vector<std::string> metricNames{"sm__cycles_elapsed.sum",
                                       "sm__cycles_active.sum"};
  wuk::CuProfiler p(metricNames);
  p.ProfileKernels("range_name", reset, run);
  std::string res =
      wuk::CuProfiler::res_to_json(p.MetricValues(metricNames));
  std::fprintf(stdout, "%s", to_json(res).c_str());
} while (0);
wuk::CuProfiler::deinit();
```

The output is a json.

```json
[{"Metrics": {"sm__cycles_elapsed.sum": 1257716664.000000,"sm__cycles_active.sum": 907696.000000}, "RangeName": "range_name"}]
```

## Build from Source

```shell
rm -rf $HOME/build
cmake -B $HOME/build .
cmake --build $HOME/build
cmake --build $HOME/build -t test
cat $HOME/build/Testing/Temporary/LastTest.log
```

## See also

- [cupti_profiler](https://github.com/srvm/cupti_profiler)

