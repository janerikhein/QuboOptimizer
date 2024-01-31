/// QUBO experiments functions

use std::io::stdin;
use crate::qubo::*;
use crate::preprocess;
use crate::start_heuristics::StartHeuristic;
use crate::tabu_search;

/// Read a file path from user input
fn read_path_from_user_input() -> String {
    let mut buffer = String::new();
    println!("Enter file path to qubo instance:");
    stdin().read_line(&mut buffer).expect("Failed to read file path");
    buffer
}

/// Example experiment to illustrate 
pub fn example() {
    /// 0) Read file_path if experiment uses an instance file
    let file_path = read_path_from_user_input();
    /// 1) Read or create instance
    let qubo = QuboInstance::from_file(&file_path);
    /// 2) Do preprocessing
    let qubo = preprocess::shrink(qubo);
    /// 3) Do start heursitic
    let start_heuristic = StartHeuristic::GreedyFromHint(0.5);
    let start_solution = start_heuristic.get_solution(&qubo);
    /// 4) Do tabu search heursitic
    let good_solution = tabu_search::tabu_search(&qubo, &start_solution);
}

///TODO: greedy_multiple_steps, 0.4-0.6, 0.2-0.8, 0.0-1.0
pub fn test_greedy_from_vec() {
    todo!();
}

/// Another experiment
pub fn foo() {
    todo!();
}

/// Yet another experiment
pub fn bar() {
    todo!();
}
