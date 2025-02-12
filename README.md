# wuk_cupti_wrapper

This library provides an simple API to use [CUPTI (the CUDA Profiling Tools Interface)](https://docs.nvidia.com/cupti/index.html). The code is mainly modified from the cupti sample [range_profiling](https://github.com/mmcloughlin/cuptisamples/tree/main/samples/range_profiling).

## Example Usage

For more detials, please refer to [concurrent_profiling sample](./test/concurrent_profiling/main.cu) and the [header](./include/wuk/cupti_wrapper.hh).

```cpp
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
```

The output is a json.

```json
[{"Metrics": {"sm__cycles_active.sum": 1196714.000000,"sm__cycles_elapsed.sum": 1360945416.000000},"RangeName": "range_name","RangeIndex": 0}]
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

