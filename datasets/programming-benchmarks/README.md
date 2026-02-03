# Programming Language Benchmarks Dataset

A synthetic dataset of **2,200+ programming language benchmark results** spanning 16 languages and 10 computational tasks, collected over 2020--2025.

## Dataset Overview

This dataset captures realistic performance characteristics of popular programming languages across diverse benchmark tasks. It accounts for:

- **Language-specific strengths**: Rust/C++ for raw speed, Go for networking, Julia for numerical computing, Python for conciseness
- **Garbage collection overhead**: Languages with GC show higher memory usage
- **Compiler/runtime improvements over time**: Languages like Julia and Python (3.11+) show significant year-over-year gains
- **Multi-core scaling**: Diminishing returns with parallelism, varying by task type
- **Lines of code trade-offs**: Concise languages (Python, Ruby) vs. verbose ones (Java, C++)

## Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `language` | string | Programming language name (16 languages) |
| `benchmark_name` | string | Name of the benchmark task (10 tasks) |
| `execution_time_ms` | float | Wall-clock execution time in milliseconds |
| `memory_usage_mb` | float | Peak memory usage in megabytes |
| `lines_of_code` | int | Lines of code in the benchmark implementation |
| `cpu_cores` | int | Number of CPU cores used (1, 2, 4, 8, or 16) |
| `year` | int | Year the benchmark was run (2020--2025) |
| `paradigm` | string | Primary paradigm: `functional`, `oop`, `multi-paradigm`, `procedural` |
| `typing` | string | Type system: `static`, `dynamic`, `gradual` |
| `gc` | string | Garbage collection: `yes`, `no`, `optional` |
| `popularity_index` | float | TIOBE-like popularity score (0--100), varies by year |

## Languages Included

| Language | Paradigm | Typing | GC | Approx. Popularity (2024) |
|----------|----------|--------|----|--------------------------|
| Python | multi-paradigm | dynamic | yes | 28.0 |
| JavaScript | multi-paradigm | dynamic | yes | 16.0 |
| Java | oop | static | yes | 15.0 |
| C++ | multi-paradigm | static | no | 10.5 |
| TypeScript | multi-paradigm | gradual | yes | 8.5 |
| C# | oop | static | yes | 7.0 |
| Go | multi-paradigm | static | yes | 4.5 |
| Swift | multi-paradigm | static | optional | 3.5 |
| Rust | multi-paradigm | static | no | 3.0 |
| Kotlin | multi-paradigm | static | yes | 2.8 |
| Ruby | oop | dynamic | yes | 1.8 |
| R | multi-paradigm | dynamic | yes | 1.5 |
| Scala | multi-paradigm | static | yes | 1.2 |
| Julia | multi-paradigm | dynamic | yes | 0.8 |
| Elixir | functional | dynamic | yes | 0.6 |
| Haskell | functional | static | yes | 0.5 |

## Benchmark Tasks

| Benchmark | Type | Description |
|-----------|------|-------------|
| `fibonacci` | CPU | Recursive/iterative Fibonacci computation |
| `matrix_multiply` | CPU | Dense matrix multiplication |
| `sort_large_array` | CPU | Sorting a large array of random integers |
| `http_server` | IO | HTTP server request handling throughput |
| `json_parse` | IO | Parsing large JSON documents |
| `regex_match` | CPU | Regular expression matching over text |
| `file_io` | IO | Sequential file read/write operations |
| `binary_trees` | CPU/Mem | Allocation-heavy binary tree operations |
| `mandelbrot` | CPU | Mandelbrot set computation |
| `spectral_norm` | CPU | Spectral norm calculation |

## Key Statistics

- **Total records**: 2,200
- **Languages**: 16
- **Benchmarks**: 10
- **Year range**: 2020--2025
- **CPU core configurations**: 1, 2, 4, 8, 16

## Use Cases

1. **Language comparison**: Compare execution speed, memory efficiency, and code conciseness across languages for specific tasks.
2. **Performance prediction**: Build regression models to predict execution time from language, task, and hardware features.
3. **Trade-off analysis**: Explore the speed vs. memory vs. code-size Pareto frontier.
4. **Trend analysis**: Track how language performance and popularity evolve over time.
5. **Language recommendation**: Suggest optimal languages for different use cases based on benchmark characteristics.
6. **Software engineering research**: Study relationships between type systems, GC, paradigms, and performance.

## How It Was Generated

The dataset was created using `create_dataset.py` with:

- **Language-specific speed/memory/LOC multipliers** calibrated against real-world benchmark data
- **Benchmark-language interaction effects** (e.g., Go's networking advantage, Julia's numerical computing strength)
- **Year-over-year improvement factors** reflecting compiler and runtime advances
- **Multi-core scaling models** with diminishing returns varying by task parallelizability
- **Gaussian noise** (5--15% variance) to simulate real measurement variability

All data is synthetic and should not be used as definitive language benchmarks.

## Sample Rows

| language | benchmark_name | execution_time_ms | memory_usage_mb | lines_of_code | cpu_cores | year |
|----------|---------------|-------------------|-----------------|---------------|-----------|------|
| Rust | fibonacci | 48.3 | 2.1 | 28 | 1 | 2024 |
| Python | matrix_multiply | 1240.5 | 312.4 | 24 | 4 | 2023 |
| Go | http_server | 72.1 | 68.5 | 62 | 8 | 2025 |
| Julia | mandelbrot | 198.7 | 28.3 | 38 | 2 | 2022 |

## License

This dataset is released under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** license.

## Citation

If you use this dataset in your work, please cite:

```
@dataset{scaturchio2025benchmarks,
  title={Programming Language Benchmarks Dataset},
  author={Scaturchio, Lorenzo},
  year={2025},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/lorenzoscaturchio/programming-language-benchmarks}
}
```
