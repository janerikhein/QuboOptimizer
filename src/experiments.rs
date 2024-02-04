/// QUBO experiment functions

use std::io::stdin;
use crate::qubo::*;
use crate::preprocess;
use crate::start_heuristics::StartHeuristic;
use crate::tabu_search::*;
use ndarray_stats::QuantileExt;
use serde_json;
use std::fs::File;
use std::io::Read;

const INST_DIR: &str = "instances/";
const METADATA: &str = "metadata.json";

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
    //let good_solution = tabu_search::tabu_search(&qubo, &start_solution);
}


pub fn test_tabu_search_params() {
    let mut file = File::open(METADATA).unwrap();
    let mut buffer = String::new();
    file.read_to_string(&mut buffer).unwrap();
    let data: serde_json::Value =
        serde_json::from_str(&buffer).expect("JSON was not well-formatted");
    println!("{}", data["bqp50.1"]["best"]);
    // let params = ModelParameters {
    //     tenure_ratio: f64,
    //     diversification_base_factor: f64,
    //     diversification_scaling_factor: f64,
    //     improvement_threshold: usize,
    //     blocking_move_number: usize,
    //     activation_function: ActivationFunction,
    // }
    let instances = ["bqp50.1",];
    // Pseudocode:
    // let x = start_vec
    // for i in instances {
    //     let qubo = QuboInstance::from_file(i);
    //     for _ in params {
    //         let obj_vals = (...);
    //         for _ in param_vals {
    //             let solution = tabu_search(qubo, x)
    //             obj_val[...] = qubo.compute_objective(x);
    //         }
    //         let best = obj_vals.argmin();
    //         println!("{param}, {}", param_vals[best]);
    //     }
    // }
}

pub fn test_start_heuristics() {
    let instances = [
        "bqp50.1",
        "bqp50.2",
        "bqp50.3",
        "bqp100.1",
        "bqp100.2",
        "bqp100.3",
        "bqp250.1",
        "bqp250.2",
        "bqp250.3",
        "bqp500.1",
        "bqp500.2",
        "bqp500.3",
        "bqp1000.1",
        "bqp1000.2",
        "bqp1000.3",
        "bqp2500.1",
        "bqp2500.2",
        "bqp2500.3",
        "p3000.1",
        "p3000.4",
        "p6000.1",
        "p6000.3",
    ];
    println!("Compare start heuristics for instances {instances:?}");
    let mut goodness = Vector::from_vec(vec![0.; 4]);
    for i in instances {
        println!("--- Starting for {i} ---");
        let filename = INST_DIR.to_owned() + i;
        let qubo = QuboInstance::from_file(&filename);
        let n = qubo.size();
        let (a, b) = create_hint_vecs(&qubo);
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
            goodness[k] += obj_vals[k];
            avg_time /= 10;
            println!(
                "#{k} {h:?}: {}, took {avg_time:.2?} on 10 run avg.",
                obj_vals[k]
            );
        }
        let best = obj_vals.argmin().unwrap();
        println!(
            "--- Done. #{best} is best: {:?}, {} ---",
            heuristics[best],
            obj_vals[best]
        );
    }
    let best = goodness.argmin().unwrap();
    println!(
        "#{best} is best overall with goodness {} vs.{}",
        goodness[best],
        goodness,
    );
}

/// Another experiment
pub fn foo() {
    todo!();
}

/// Yet another experiment
pub fn bar() {
    todo!();
}

/// Helper function for start heuristic testing
fn create_hint_vecs(qubo: &QuboInstance) -> (Vector, Vector) {
    let n = qubo.size();
    let mut negs = vec![0; n];  // number of negatives
    let mut nzrs = vec![0; n];  // number of nonzeros
    let mut sneg = vec![0.; n]; // sum of negatives
    let mut spos = vec![0.; n]; // sum of positives
    // x_i
    for i in 0..n {
        // Row i
        for j in i..n {
            let entry_ij = qubo.get_entry_at(i, j);
            if entry_ij > 0. {
                nzrs[i] += 1;
                spos[i] += entry_ij;
            }
            else if entry_ij < 0. {
                nzrs[i] += 1;
                negs[i] += 1;
                sneg[i] += entry_ij;
            }
        }
        // Col i
        for j in 0..i {
            let entry_ji = qubo.get_entry_at(j, i);
            if entry_ji > 0. {
                nzrs[i] += 1;
                spos[i] += entry_ji;
            }
            else if entry_ji < 0. {
                nzrs[i] += 1;
                negs[i] += 1;
                sneg[i] += entry_ji;
            }
        }
    }
    let mut a = Vector::from_vec(vec![0.; n]);
    let mut b = Vector::from_vec(vec![0.; n]);
    for i in 0..n {
        if   nzrs[i] != 0 {
            a[i] = (negs[i] as f64)/(nzrs[i] as f64);
        }
        else {
            a[i] = 1.;
        }
        if sneg[i] - spos[i] != 0. {
            b[i] = sneg[i]/(sneg[i] - spos[i]);
        }
        else {
            b[i] = 1.;
        }
    }
    (a, b)
}
