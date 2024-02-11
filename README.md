QuboOptimizer is a heuristic for the quadratic unconstrained binary optimization
(**QUBO**) problem written in Rust.
Optimization is done in three main steps:

1. Preprocessing
2. Running a greedy start heuristic
3. Doing tabu search

We also provide some experiments to benchmark/test this implementation.

This is part of a project for the Scientific Computing course of TU Berlin.

**Authors:**  
* Jan-Erik Hein  
* Lukas Mehl  
* Paul Meinhold

## Source files
```
src
├── qubo.rs             // Implement QuboInstance struct with upper-triang. matrix
├── preprocess.rs       // Implement five preprocessing rules
├── start_heuristics.rs // Implement greedy start heuristic for QuboInstance
├── tabu_search.rs      // Implement tabu search for QuboInstance
├── experiments.rs      // Implement experiments
└── main.rs             // Run experiments via command line
```

[Literature](https://pads.ccc.de/QUwrTGlwvn)
