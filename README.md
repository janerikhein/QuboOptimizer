QuboOptimizer is a heuristic for the quadratic unconstrained binary optimization
(**QUBO**) problem written in Rust.
Optimization is done in three main steps:

1. Preprocessing
2. Running a greedy start heuristic
3. Doing tabu search

We also provide some experiments to benchmark/test this implementation.  
This is part of a project for the Scientific Computing (2023/2024) course at
TU Berlin.

**Authors:**  
* Jan-Erik Hein  
* Lukas Mehl  
* Paul Meinhold

## Benchmark instances
Some of the used instances can be found in the `instances/` directory.
Due to size limitations, the bigger ones have to be attained
[here](https://github.com/rliang/qubo-benchmark-instances).

A comprehensive list of all used instances can be found in
[./metadata.json](metadata.json).

## Usage
Make sure all instances are locally available.  
Run `cargo run --release EXPERIMENT_NUM` inside the project directory and the
corresponging experiment will be executed:
```Rust
match experiment_num {
    1 => {
        experiments::analyze_preproc();
    }
    2 => {
        experiments::analyze_start_heuristics();
    }
    3 => {
        experiments::tune_tabu_params();
    }
    4 => {
        experiments::tune_dsf();
    }
    5 => {
        experiments::tune_tr();
    }
    6 => {
        experiments::analyze_tabu_search();
    }
    _ => {
        println!("No valid experiment_num");
    }
}
```

## Experiment logs
For each experiment previously run by us, we provide the logs as
`EXPERIMENT_NUM.out`.

## Source files
**Rust**
```
src
├── qubo.rs             // Implement QuboInstance struct with upper-triang. matrix
├── preprocess.rs       // Implement five preprocessing rules
├── start_heuristics.rs // Implement greedy start heuristic for QuboInstance
├── tabu_search.rs      // Implement tabu search for QuboInstance
├── experiments.rs      // Implement experiments
└── main.rs             // Run experiments via command line
```
**Python**
```
qbsolv
└── test.py             // Use Qbsolv to compare with out heuristic
```
Read [Qbsolv](https://github.com/dwavesystems/qbsolv) for more info.
