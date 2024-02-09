/// QUBO experiment functions

use std::io::stdin;
use crate::qubo::*;
use crate::{preprocess, tabu_search};
use crate::start_heuristics::StartHeuristic;
use crate::tabu_search::*;
use ndarray::{Array1, Array5};
use ndarray_stats::QuantileExt;
use serde_json;
use std::fs::File;
use std::io::Read;

const INST_DIR: &str = "instances/";
const METADATA: &str = "metadata.json";

fn filepath_from_name(filename: &str) -> String {
    INST_DIR.to_owned() + filename
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
    let qubo = QuboInstance::from_file(file_path);
    // 2) Do preprocessing
    let qubo = preprocess::shrink(qubo);
    // 3) Do start heuristic
    let start_heuristic = StartHeuristic::GreedyFromHint(0.5);
    let start_solution = start_heuristic.get_solution(&qubo);
    // 4) Do tabu search heuristic
    tabu_search::tabu_search_with_defaults(&qubo, &start_solution, 2);
}

fn compute_best_start_solution(qubo: &QuboInstance) -> BinaryVector {
    let mut obj_vals = Vector::from_vec(vec![0.; 3]);
    let mut sols = Vec::<BinaryVector>::new();
    let (a, b, c) = create_hint_vecs(qubo);
    let heuristics = [
        StartHeuristic::GreedyFromVec(a),
        StartHeuristic::GreedyFromVec(b),
        StartHeuristic::GreedyFromVec(c),
    ];
    for i in 0..heuristics.len() {
        sols.push(heuristics[i].get_solution(qubo));
        obj_vals[i] = qubo.compute_objective(&sols[i]);
    }
    let best = obj_vals.argmin().unwrap();
    sols[best].clone()
}

/**
// The QUBO instance
qubo_instance: &QuboInstance,
// ratio for setting tabu tenures relative to problem size
//TODO: test values: (0.1, 0.25, 0.5)
tenure_ratio: f64,

// ratio for length of diversification phase relative to problem size
//TODO: test values: (0.1, 0.5, 1.0)
diversification_length_scale: f64,

// base factor for diversification penalties
//TODO: test values: (0.1, 0.25, 0.5)
diversification_base_factor: f64,

// scaling factor for increasing diversification intensity after unsuccessful phases
//TODO: not too impactful, can fix to 1.5
diversification_scaling_factor: f64,

// activation function for diversification penalties
//TODO: ignore this, fix to ActivationFunction::CONSTANT
activation_function: ActivationFunction,

// improvement threshold relative to problem size
//TODO: test values: (1.0, 5.0, 10.0),
improvement_threshold_scale: f64,

// blocking move number relative to problem size
//TODO: test values: (0.0, 0.05)
blocking_move_number_scale: f64,

// time limit for tabu search
// TODO: set something reasonable here, small instances <500 should terminate in less than 30s
time_limit_seconds: usize,

// seed value
// TODO: fix some seed value, i.e 42
seed: usize,
*/
fn params_from
(q: &QuboInstance, tr: f64, dls: f64, dbf: f64, its: f64, bmns: f64, tl: usize)
-> SearchParameters {
    let af = ActivationFunction::Constant;
    SearchParameters::new(q, tr, dls, dbf, 1.5, af, its, bmns, tl, 42)
}

pub fn tune_tabu_params() {
    // Parameters to tune
    let tr = vec![0.1, 0.25,  0.5,];
    let dls = vec![0.1, 0.5,   1.0,];
    let dbf = vec![0.1, 0.25,  0.5,];
    let its = vec![1.0, 5.0,  10.0,];
    let bmns = vec![0.0, 0.05];
    // Counters
    let mut tr_count = Array1::from_vec(vec![0; 3]);
    let mut dls_count = Array1::from_vec(vec![0; 3]);
    let mut dbf_count = Array1::from_vec(vec![0; 3]);
    let mut its_count = Array1::from_vec(vec![0; 3]);
    let mut bmns_count = Array1::from_vec(vec![0; 2]);
    let instances = [
        "G1",
        "G2",
        "bqp50.1",
        "bqp50.2",
        "bqp100.1",
        "bqp100.2",
        "bqp250.1",
        "bqp500.1",
    ];
    for i in instances {
        let literature_obj = get_literature_obj(i);
        let qubo = QuboInstance::from_file(&filepath_from_name(i));
        let n = qubo.size();
        let time_limit_secs = std::cmp::max(5, n.ilog10() as usize);
        println!("Using time limit of {time_limit_secs}s");
        let start_solution =
            StartHeuristic::GreedyFromHint(0.5).get_solution(&qubo);
        let mut obj_vals = Array5::from_elem((3, 3, 3, 3, 2), 0.);
        let start_solution = compute_best_start_solution(&qubo);
        // Iterate over all possible combinations
        for i in 0..tr.len() {
            for j in 0..dls.len() {
                for k in 0..dbf.len() {
                    for l in 0..its.len() {
                        for m in 0..bmns.len() {
                            //println!("{i},{j},{k},{l},{m}");
                            let params = params_from(
                                &qubo,
                                tr[i],
                                dls[j],
                                dbf[k],
                                its[l],
                                bmns[m],
                                time_limit_secs,
                            );
                            let solution = tabu_search::tabu_search(
                                &qubo,
                                &start_solution,
                                5,
                                params
                            );
                            obj_vals[[i, j, k, l, m]] =
                                qubo.compute_objective(&solution);
                        }
                    }
                }
            }
        }
        // Get best params
        let best = obj_vals.argmin().unwrap();
        tr_count[best.0] += 1;
        dls_count[best.1] += 1;
        dbf_count[best.2] += 1;
        its_count[best.3] += 1;
        bmns_count[best.4] += 1;
        println!(
            "best choice for {i}: {} {} {} {} {}",
            tr[best.0],
            dls[best.1],
            dbf[best.2],
            its[best.3],
            bmns[best.4],
        );
    }
    let i = tr_count.argmax().unwrap();
    let j = dls_count.argmax().unwrap();
    let k = dbf_count.argmax().unwrap();
    let l = its_count.argmax().unwrap();
    let m = bmns_count.argmax().unwrap();
    println!(
        "best choice overall: {} {} {} {} {}",
        tr[i],
        dls[j],
        dbf[k],
        its[l],
        bmns[m],
    );
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
        let qubo = QuboInstance::from_file(&filepath_from_name(i));
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
            let avg_time = now.elapsed()/10;
            obj_vals[k] = qubo.compute_objective(&sol);
            goodness[k] = obj_vals[k]/literature_obj;
            total_goodness[k] += goodness[k];
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
