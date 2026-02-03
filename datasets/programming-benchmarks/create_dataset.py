"""
Generate a synthetic dataset of 2000+ programming language benchmark results.

Simulates realistic performance characteristics across 16 languages and 10
benchmark tasks, accounting for language-specific strengths, typing systems,
garbage collection overhead, and historical performance improvements.
"""

import csv
import random
import math

random.seed(123)

NUM_RECORDS = 2200

LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Rust", "Go", "Java", "C++",
    "C#", "Ruby", "Swift", "Kotlin", "Scala", "R", "Julia", "Haskell", "Elixir",
]

BENCHMARKS = [
    "fibonacci", "matrix_multiply", "sort_large_array", "http_server",
    "json_parse", "regex_match", "file_io", "binary_trees",
    "mandelbrot", "spectral_norm",
]

PARADIGMS = {
    "Python": "multi-paradigm", "JavaScript": "multi-paradigm",
    "TypeScript": "multi-paradigm", "Rust": "multi-paradigm",
    "Go": "multi-paradigm", "Java": "oop", "C++": "multi-paradigm",
    "C#": "oop", "Ruby": "oop", "Swift": "multi-paradigm",
    "Kotlin": "multi-paradigm", "Scala": "multi-paradigm",
    "R": "multi-paradigm", "Julia": "multi-paradigm",
    "Haskell": "functional", "Elixir": "functional",
}

TYPING = {
    "Python": "dynamic", "JavaScript": "dynamic", "TypeScript": "gradual",
    "Rust": "static", "Go": "static", "Java": "static", "C++": "static",
    "C#": "static", "Ruby": "dynamic", "Swift": "static",
    "Kotlin": "static", "Scala": "static", "R": "dynamic",
    "Julia": "dynamic", "Haskell": "static", "Elixir": "dynamic",
}

GC = {
    "Python": "yes", "JavaScript": "yes", "TypeScript": "yes",
    "Rust": "no", "Go": "yes", "Java": "yes", "C++": "no",
    "C#": "yes", "Ruby": "yes", "Swift": "optional",
    "Kotlin": "yes", "Scala": "yes", "R": "yes",
    "Julia": "yes", "Haskell": "yes", "Elixir": "yes",
}

# TIOBE-like popularity index (0-100) - approximate 2024 values
POPULARITY_BASE = {
    "Python": 28.0, "JavaScript": 16.0, "TypeScript": 8.5, "Rust": 3.0,
    "Go": 4.5, "Java": 15.0, "C++": 10.5, "C#": 7.0,
    "Ruby": 1.8, "Swift": 3.5, "Kotlin": 2.8, "Scala": 1.2,
    "R": 1.5, "Julia": 0.8, "Haskell": 0.5, "Elixir": 0.6,
}

# Popularity trend by year (multipliers)
POPULARITY_TREND = {
    "Python":     {2020: 0.80, 2021: 0.87, 2022: 0.93, 2023: 0.97, 2024: 1.0, 2025: 1.02},
    "JavaScript": {2020: 1.05, 2021: 1.03, 2022: 1.02, 2023: 1.01, 2024: 1.0, 2025: 0.98},
    "TypeScript":  {2020: 0.55, 2021: 0.65, 2022: 0.78, 2023: 0.90, 2024: 1.0, 2025: 1.08},
    "Rust":       {2020: 0.40, 2021: 0.55, 2022: 0.70, 2023: 0.85, 2024: 1.0, 2025: 1.15},
    "Go":         {2020: 0.70, 2021: 0.78, 2022: 0.85, 2023: 0.93, 2024: 1.0, 2025: 1.03},
    "Java":       {2020: 1.10, 2021: 1.07, 2022: 1.04, 2023: 1.02, 2024: 1.0, 2025: 0.97},
    "C++":        {2020: 0.95, 2021: 0.96, 2022: 0.98, 2023: 0.99, 2024: 1.0, 2025: 1.01},
    "C#":         {2020: 0.90, 2021: 0.93, 2022: 0.96, 2023: 0.98, 2024: 1.0, 2025: 1.01},
    "Ruby":       {2020: 1.30, 2021: 1.20, 2022: 1.10, 2023: 1.05, 2024: 1.0, 2025: 0.95},
    "Swift":      {2020: 0.80, 2021: 0.85, 2022: 0.90, 2023: 0.95, 2024: 1.0, 2025: 1.02},
    "Kotlin":     {2020: 0.60, 2021: 0.72, 2022: 0.83, 2023: 0.92, 2024: 1.0, 2025: 1.05},
    "Scala":      {2020: 1.15, 2021: 1.10, 2022: 1.05, 2023: 1.02, 2024: 1.0, 2025: 0.97},
    "R":          {2020: 1.20, 2021: 1.12, 2022: 1.06, 2023: 1.02, 2024: 1.0, 2025: 0.96},
    "Julia":      {2020: 0.50, 2021: 0.62, 2022: 0.75, 2023: 0.88, 2024: 1.0, 2025: 1.10},
    "Haskell":    {2020: 1.10, 2021: 1.05, 2022: 1.02, 2023: 1.01, 2024: 1.0, 2025: 0.98},
    "Elixir":     {2020: 0.60, 2021: 0.70, 2022: 0.80, 2023: 0.90, 2024: 1.0, 2025: 1.08},
}

# --- Relative performance characteristics ---
# Base execution time multiplier (1.0 = C++ baseline, higher = slower)
LANGUAGE_SPEED = {
    "C++": 1.0, "Rust": 1.05, "C#": 2.5, "Java": 2.8, "Go": 2.2,
    "Swift": 1.8, "Kotlin": 3.0, "Scala": 3.5, "Julia": 1.6,
    "Haskell": 3.0, "JavaScript": 5.0, "TypeScript": 5.2,
    "Elixir": 8.0, "Python": 35.0, "Ruby": 30.0, "R": 40.0,
}

# Memory usage multiplier (1.0 = C++ baseline)
LANGUAGE_MEMORY = {
    "C++": 1.0, "Rust": 1.1, "Go": 2.5, "C#": 4.0, "Java": 5.0,
    "Swift": 2.0, "Kotlin": 5.5, "Scala": 6.0, "Julia": 3.0,
    "Haskell": 3.5, "JavaScript": 4.5, "TypeScript": 4.8,
    "Elixir": 5.0, "Python": 6.0, "Ruby": 7.0, "R": 8.0,
}

# Lines of code multiplier (1.0 = Python baseline - fewest lines)
LANGUAGE_LOC = {
    "Python": 1.0, "Ruby": 1.1, "Elixir": 1.2, "Julia": 1.1,
    "Haskell": 1.05, "Scala": 1.15, "R": 1.0, "Kotlin": 1.3,
    "JavaScript": 1.4, "TypeScript": 1.5, "Swift": 1.4, "Go": 1.6,
    "C#": 1.7, "Java": 2.0, "Rust": 1.8, "C++": 2.2,
}

# Benchmark-specific base times (ms) and characteristic properties
BENCHMARK_BASE = {
    "fibonacci":        {"base_ms": 50,    "base_mem_mb": 2,   "base_loc": 15,  "cpu_intensive": True,  "io_intensive": False},
    "matrix_multiply":  {"base_ms": 200,   "base_mem_mb": 50,  "base_loc": 25,  "cpu_intensive": True,  "io_intensive": False},
    "sort_large_array": {"base_ms": 150,   "base_mem_mb": 100, "base_loc": 20,  "cpu_intensive": True,  "io_intensive": False},
    "http_server":      {"base_ms": 500,   "base_mem_mb": 30,  "base_loc": 40,  "cpu_intensive": False, "io_intensive": True},
    "json_parse":       {"base_ms": 80,    "base_mem_mb": 20,  "base_loc": 18,  "cpu_intensive": False, "io_intensive": True},
    "regex_match":      {"base_ms": 60,    "base_mem_mb": 5,   "base_loc": 12,  "cpu_intensive": True,  "io_intensive": False},
    "file_io":          {"base_ms": 300,   "base_mem_mb": 15,  "base_loc": 22,  "cpu_intensive": False, "io_intensive": True},
    "binary_trees":     {"base_ms": 400,   "base_mem_mb": 200, "base_loc": 30,  "cpu_intensive": True,  "io_intensive": False},
    "mandelbrot":       {"base_ms": 350,   "base_mem_mb": 10,  "base_loc": 35,  "cpu_intensive": True,  "io_intensive": False},
    "spectral_norm":    {"base_ms": 250,   "base_mem_mb": 8,   "base_loc": 28,  "cpu_intensive": True,  "io_intensive": False},
}

# Special benchmark-language interactions
# Some languages are particularly good/bad at specific tasks
BENCHMARK_LANGUAGE_MODIFIERS = {
    "http_server": {
        "Go": 0.3,       # Go excels at HTTP/networking
        "Elixir": 0.4,   # Elixir/BEAM great at concurrent IO
        "JavaScript": 0.5,  # Node.js async
        "Java": 0.6,
        "Rust": 0.5,     # Rust async also fast
        "Python": 2.0,   # Python GIL hurts here
    },
    "json_parse": {
        "JavaScript": 0.4,  # Native JSON support
        "Go": 0.6,
        "Rust": 0.5,     # serde is fast
        "Python": 1.5,
    },
    "matrix_multiply": {
        "Julia": 0.4,    # Julia excels at numerical computing
        "R": 0.8,        # R with BLAS is decent
        "Python": 0.6,   # NumPy (C backend)
        "Haskell": 1.5,
    },
    "regex_match": {
        "Rust": 0.5,     # Rust regex crate is very fast
        "Go": 0.7,       # RE2 engine
        "Python": 1.2,
        "Ruby": 1.3,
    },
    "file_io": {
        "Go": 0.5,
        "Rust": 0.5,
        "Elixir": 0.6,
        "Java": 0.7,
    },
    "fibonacci": {
        "Haskell": 0.6,  # Lazy eval helps
        "Julia": 0.5,
    },
    "binary_trees": {
        "Java": 0.7,     # JVM GC handles this well
        "Go": 0.6,
    },
}

# Year-over-year performance improvement (compiler/runtime improvements)
YEAR_IMPROVEMENT = {
    "Rust":  {2020: 1.15, 2021: 1.10, 2022: 1.06, 2023: 1.03, 2024: 1.01, 2025: 1.00},
    "Go":    {2020: 1.20, 2021: 1.14, 2022: 1.08, 2023: 1.04, 2024: 1.01, 2025: 1.00},
    "Julia": {2020: 1.40, 2021: 1.25, 2022: 1.15, 2023: 1.08, 2024: 1.03, 2025: 1.00},
    "Java":  {2020: 1.12, 2021: 1.08, 2022: 1.05, 2023: 1.03, 2024: 1.01, 2025: 1.00},
    "C++":   {2020: 1.05, 2021: 1.04, 2022: 1.03, 2023: 1.02, 2024: 1.01, 2025: 1.00},
    "Python":{2020: 1.00, 2021: 1.00, 2022: 0.98, 2023: 0.95, 2024: 0.90, 2025: 0.85},  # Python getting faster (3.11+, JIT)
}
# Default: slight improvement for unlisted languages
DEFAULT_YEAR_IMPROVEMENT = {2020: 1.10, 2021: 1.07, 2022: 1.05, 2023: 1.03, 2024: 1.01, 2025: 1.00}


def compute_execution_time(language, benchmark, year, cpu_cores):
    """Compute realistic execution time for a benchmark run."""
    bench = BENCHMARK_BASE[benchmark]
    base_ms = bench["base_ms"]

    # Language speed factor
    speed = LANGUAGE_SPEED[language]

    # IO-intensive benchmarks narrow the gap (most time in OS/IO, not language)
    if bench["io_intensive"]:
        speed = 1.0 + (speed - 1.0) * 0.3

    # Benchmark-specific modifiers
    bm_mods = BENCHMARK_LANGUAGE_MODIFIERS.get(benchmark, {})
    modifier = bm_mods.get(language, 1.0)

    # Year improvement
    year_imp = YEAR_IMPROVEMENT.get(language, DEFAULT_YEAR_IMPROVEMENT)
    yr_factor = year_imp.get(year, 1.0)

    # CPU cores effect (diminishing returns)
    core_factor = 1.0
    if cpu_cores > 1:
        parallelism_benefit = {
            "matrix_multiply": 0.7, "sort_large_array": 0.5,
            "http_server": 0.8, "mandelbrot": 0.9,
            "spectral_norm": 0.6, "binary_trees": 0.4,
        }
        benefit = parallelism_benefit.get(benchmark, 0.1)
        core_factor = 1.0 / (1.0 + benefit * math.log2(cpu_cores))

    time_ms = base_ms * speed * modifier * yr_factor * core_factor

    # Add noise (5-15% variance)
    noise = random.gauss(1.0, 0.08)
    noise = max(0.7, min(1.3, noise))
    time_ms *= noise

    return round(max(0.1, time_ms), 2)


def compute_memory(language, benchmark, cpu_cores):
    """Compute realistic memory usage."""
    bench = BENCHMARK_BASE[benchmark]
    base_mem = bench["base_mem_mb"]

    mem_factor = LANGUAGE_MEMORY[language]

    # Multi-core increases memory
    core_mem_factor = 1.0 + 0.1 * (cpu_cores - 1)

    mem = base_mem * mem_factor * core_mem_factor

    # Noise
    noise = random.gauss(1.0, 0.1)
    noise = max(0.6, min(1.5, noise))
    mem *= noise

    return round(max(0.5, mem), 2)


def compute_loc(language, benchmark):
    """Compute lines of code."""
    base = BENCHMARK_BASE[benchmark]["base_loc"]
    loc_factor = LANGUAGE_LOC[language]

    loc = base * loc_factor

    # Some variation per implementation
    noise = random.gauss(1.0, 0.12)
    noise = max(0.7, min(1.4, noise))
    loc *= noise

    return max(5, round(loc))


def generate_record():
    """Generate a single benchmark record."""
    language = random.choice(LANGUAGES)
    benchmark = random.choice(BENCHMARKS)
    year = random.choices(
        list(range(2020, 2026)),
        weights=[0.10, 0.12, 0.16, 0.20, 0.22, 0.20],
        k=1,
    )[0]
    cpu_cores = random.choice([1, 1, 1, 2, 2, 4, 4, 8, 16])

    execution_time = compute_execution_time(language, benchmark, year, cpu_cores)
    memory_usage = compute_memory(language, benchmark, cpu_cores)
    loc = compute_loc(language, benchmark)

    paradigm = PARADIGMS[language]
    typing = TYPING[language]
    gc = GC[language]

    # Popularity index with year trend
    pop_base = POPULARITY_BASE[language]
    pop_trend = POPULARITY_TREND[language].get(year, 1.0)
    popularity_index = round(pop_base * pop_trend, 2)

    return {
        "language": language,
        "benchmark_name": benchmark,
        "execution_time_ms": execution_time,
        "memory_usage_mb": memory_usage,
        "lines_of_code": loc,
        "cpu_cores": cpu_cores,
        "year": year,
        "paradigm": paradigm,
        "typing": typing,
        "gc": gc,
        "popularity_index": popularity_index,
    }


def main():
    print("Generating Programming Language Benchmarks dataset...")

    records = []
    # Ensure coverage: at least one record per language-benchmark-year combo
    for language in LANGUAGES:
        for benchmark in BENCHMARKS:
            for year in range(2020, 2026):
                cpu_cores = random.choice([1, 2, 4])
                rec = generate_record()
                rec["language"] = language
                rec["benchmark_name"] = benchmark
                rec["year"] = year
                rec["cpu_cores"] = cpu_cores
                # Recompute derived values
                rec["execution_time_ms"] = compute_execution_time(language, benchmark, year, cpu_cores)
                rec["memory_usage_mb"] = compute_memory(language, benchmark, cpu_cores)
                rec["lines_of_code"] = compute_loc(language, benchmark)
                rec["paradigm"] = PARADIGMS[language]
                rec["typing"] = TYPING[language]
                rec["gc"] = GC[language]
                pop_base = POPULARITY_BASE[language]
                pop_trend = POPULARITY_TREND[language].get(year, 1.0)
                rec["popularity_index"] = round(pop_base * pop_trend, 2)
                records.append(rec)

    # Fill remaining with random records (varied cpu_cores, additional runs)
    while len(records) < NUM_RECORDS:
        records.append(generate_record())

    random.shuffle(records)

    fieldnames = [
        "language", "benchmark_name", "execution_time_ms", "memory_usage_mb",
        "lines_of_code", "cpu_cores", "year", "paradigm", "typing", "gc",
        "popularity_index",
    ]

    output_path = "/Users/gr8monk3ys/code/ml-portfolio/kaggle/datasets/programming-benchmarks/language_benchmarks.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated {len(records)} records -> {output_path}")

    # Print basic stats
    from collections import Counter
    lang_counts = Counter(r["language"] for r in records)
    print("\nRecords per language:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")

    bench_counts = Counter(r["benchmark_name"] for r in records)
    print("\nRecords per benchmark:")
    for b, c in sorted(bench_counts.items(), key=lambda x: -x[1]):
        print(f"  {b}: {c}")

    # Show performance leaders
    print("\nMedian execution times (fibonacci, 1 core, 2025):")
    for lang in sorted(LANGUAGES):
        times = [r["execution_time_ms"] for r in records
                 if r["language"] == lang and r["benchmark_name"] == "fibonacci"
                 and r["cpu_cores"] == 1 and r["year"] == 2025]
        if times:
            times.sort()
            median = times[len(times) // 2]
            print(f"  {lang}: {median:.1f} ms")


if __name__ == "__main__":
    main()
