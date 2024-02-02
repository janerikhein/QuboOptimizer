/// QUBO experiment functions

use std::io::stdin;
use crate::qubo::*;
use crate::preprocess;
use crate::start_heuristics::StartHeuristic;
use crate::tabu_search;
use ndarray_stats::QuantileExt;

const INST_DIR: &str = "instances/";

/// Read a file path from user input
fn read_path_from_user_input() -> String {
    let mut buffer = String::new();
    println!("Enter file path to qubo instance:");
    stdin().read_line(&mut buffer).expect("Failed to read file path");
    buffer
}

/// Example experiment to illustrate 
pub fn example() {
    // 0) Read file_path if experiment uses an instance file
    let file_path = read_path_from_user_input();
    // 1) Read or create instance
    let qubo = QuboInstance::from_file(&file_path);
    // 2) Do preprocessing
    let qubo = preprocess::shrink(qubo);
    // 3) Do start heuristic
    let start_heuristic = StartHeuristic::GreedyFromHint(0.5);
    let start_solution = start_heuristic.get_solution(&qubo);
    // 4) Do tabu search heuristic
    let good_solution = tabu_search::tabu_search(&qubo, &start_solution);
}

pub fn test_start_heuristics() {
    fn create_vecs(qubo: &QuboInstance) -> (Vector, Vector) {
        let n = qubo.size();
        let mut  p = vec![0; n];  // number of positives
        let mut  z = vec![0; n];  // number of nonzeros
        let mut sp = vec![0.; n]; // sum of positives
        let mut sn = vec![0.; n]; // sum of negatives
        for i in 0..n {
            for j in i..n {
                // Run through row i
                let entry_ij = qubo.get_entry_at(i, j);
                if entry_ij > 0. {
                    p[i] += 1;
                    z[i] += 1;
                    sp[i] += entry_ij;
                }
                else if entry_ij < 0. {
                    z[i] += 1;
                    sn[i] += entry_ij;
                }
            }
        }
        let mut a = Vector::from_vec(vec![0.; n]);
        let mut b = Vector::from_vec(vec![0.; n]);
        for i in 0..n {
            if   z[i] != 0 { a[i] = (p[i] as f64)/(z[i] as f64); }
            else { a[i] = 1.; }
            if sp[i] - sn[i] != 0. { b[i] = sp[i]/(sp[i] - sn[i]); }
            else { b[i] = 1.; }
        }
        (a, b)
    }
    //TODO: greedy_multiple_steps?
    let instances = ["p3000.1", "p3000.4", "p6000.1", "p6000.3"];
    println!("Compare start heuristics for instances {instances:?}");
    for i in instances {
        println!("--- Starting for {i} ---");
        let filename = INST_DIR.to_owned() + i;
        let qubo = QuboInstance::from_file(&filename);
        let n = qubo.size();
        let (a, b) = create_vecs(&qubo);
        let heuristics = [
            StartHeuristic::Random(42),
            StartHeuristic::GreedyFromVec(Vector::from_vec(vec![0.5; n])),
            StartHeuristic::GreedyFromVec(a),
            StartHeuristic::GreedyFromVec(b),
        ];
        let mut obj_vals = Vector::from_vec(vec![0.; heuristics.len()]);
        for k in 0..heuristics.len() {
            let h = &heuristics[k];
            let mut avg_time = std::time::Duration::new(0, 0);
            let mut sol = BinaryVector::from_vec(vec![false; n]);
            for _ in 0..10 {
                let now = std::time::Instant::now();
                sol = h.get_solution(&qubo);
                avg_time += now.elapsed();
            }
            obj_vals[k] = qubo.compute_objective(sol);
            avg_time /= 10;
            println!(
                "#{k} {h:?}: {}, took {avg_time:.2?} on 10 run avg.",
                obj_vals[k]);
        }
        let best = obj_vals.argmin().unwrap();
        println!(
            "--- Done. #Best={best}: {:?}, {} ---",
            heuristics[best],
            obj_vals[best]);
    }
}

/// Another experiment
pub fn foo() {
    todo!();
}

/// Yet another experiment
pub fn bar() {
    todo!();
}
