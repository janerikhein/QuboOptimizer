/// QUBO experiment functions

use std::io::stdin;
use crate::qubo::*;
use crate::{preprocess, tabu_search};
use crate::start_heuristics::StartHeuristic;
use crate::tabu_search::*;
use ndarray_stats::QuantileExt;
use serde_json;
use std::fs::File;
use std::io::Read;

const INST_DIR: &str = "instances/";
const METADATA: &str = "metadata.json";

fn filepath_from_name(filename: &str) -> &str {
    &(INST_DIR.to_owned() + filename)
}

fn get_literature_obj(filename: &str) -> f64 {
    let mut file = File::open(METADATA).unwrap();
    let mut buffer = String::new();
    file.read_to_string(&mut buffer).unwrap();
    let data: serde_json::Value =
        serde_json::from_str(&buffer).expect("JSON was not well-formatted");
    data["bqp50.1"]["best"].as_f64().unwrap()
}

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
    //let file_path = read_path_from_user_input();
    let file_path= "instances/bqp100.9";
    // 1) Read or create instance
    let qubo = QuboInstance::from_file(&file_path);
    // 2) Do preprocessing
    let qubo = preprocess::shrink(qubo);
    // 3) Do start heuristic
    let start_heuristic = StartHeuristic::GreedyFromHint(0.5);
    let start_solution = start_heuristic.get_solution(&qubo);
    // 4) Do tabu search heuristic
    let good_solution = tabu_search::tabu_search_with_defaults(&qubo, &start_solution, 2);
}

fn compute_best_start_solution(qubo: &QuboInstance) -> BinaryVector {
    let obj_vals = Vector::from_vec(vec![0.; 3]);
    let sols = Vec::<BinaryVector>::new();
    let (a, b, c) = create_hint_vecs(&qubo);
    let heuristics = [
        StartHeuristic::GreedyFromVec(a),
        StartHeuristic::GreedyFromVec(b),
        StartHeuristic::GreedyFromVec(c),
    ];
    for i in 0..heuristics.len() {
        sols[i] = heuristics[i].get_solution(&qubo);
        obj_vals[i] = qubo.compute_objective(sols[i]);
    }
    let best = obj_vals.argmin().unwrap();
    sols[best]
}

// qubo_instance: &QuboInstance,
// tenure_ratio: f64,
// diversification_base_factor: f64,
// diversification_scaling_factor: f64,
// activation_function: ActivationFunction,
// improvement_threshold: usize,
// blocking_move_number: usize,
// time_limit_seconds: usize,
pub fn tune_tabu_params() {
    let instances = ["bqp50.1",];
    for i in instances {
        let qubo = QuboInstance::from_file(filepath_from_name(i));
        let start_solution =
            StartHeuristic::GreedyFromHint(0.5).get_solution(&qubo);
        let params_mat = Matrix::from_shape_vec(
            (6, 4),
            vec![
                1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1.,
                1., 1., 1., 1., 1.,
            ]
        ).unwrap();
        for i in 0..params_mat.ncols() {
            todo!();
            let start_solution = compute_best_start_solution(&qubo);
            //let params = SearchParameters::new(...);
            //let solution =
            //    tabu_search::tabu_search(&qubo, &start_solution, &params);
        }
    }
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
    let mut total_goodness = Vector::from_vec(vec![0.; 4]);
    for i in instances {
        println!("--- Starting for {i} ---");
        let literature_obj = get_literature_obj(i);
        let qubo = QuboInstance::from_file(filepath_from_name(i));
        let n = qubo.size();
        let (a, b, c) = create_hint_vecs(&qubo);
        let heuristics = [
            StartHeuristic::Random(42),
            StartHeuristic::GreedyFromVec(a),
            StartHeuristic::GreedyFromVec(b),
            StartHeuristic::GreedyFromVec(c),
        ];
        let mut goodness = Vector::from_vec(vec![0.; heuristics.len()]);
        let mut obj_vals = Vector::from_vec(vec![0.; heuristics.len()]);
        for k in 0..heuristics.len() {
            let mut sol = BinaryVector::from_vec(vec![false; n]);
            let now = std::time::Instant::now();
            for _ in 0..10 {
                sol = heuristics[k].get_solution(&qubo);
            }
            let mut avg_time = now.elapsed()/10;
            obj_vals[k] = qubo.compute_objective(sol);
            goodness[k] = obj_vals[k]/literature_obj;
            total_goodness[k] += goodness[k];
            avg_time /= 10;
            println!(
                "#{k} {:?}: {} {}, took {avg_time:.2?} on 10 run avg.",
                heuristics[k],
                obj_vals[k],
                goodness[k],
            );
        }
        let best = obj_vals.argmin().unwrap();
        println!(
            "--- Done. #{best} is best: {:?}, {} ---",
            heuristics[best],
            obj_vals[best]
        );
    }
    let best = total_goodness.argmin().unwrap();
    println!(
        "#{best} is best overall with goodness {} vs.{}",
        total_goodness[best],
        total_goodness,
    );
}

/// Helper function for start heuristic testing
fn create_hint_vecs(qubo: &QuboInstance) -> (Vector, Vector, Vector) {
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
    (Vector::from_vec(vec![0.5; n]), a, b)
}
