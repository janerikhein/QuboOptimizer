/// QUBO experiment functions

use crate::qubo::*;
use crate::preprocess::shrink;
use crate::start_heuristics::StartHeuristic;
use crate::tabu_search::*;
use ndarray::{Array1, Array4};
use ndarray_stats::QuantileExt;
use serde_json;
use std::fs::File;
use std::io::Read;

const INST_DIR: &str = "instances/";
const METADATA: &str = "metadata.json";
const LOG_LEVEL: usize = 2;

/// Helper function to get filepath from instance name
fn filepath_from_name(filename: &str) -> String {
    INST_DIR.to_owned() + filename
}

/// Helper function to get best objective value from literature
fn get_literature_obj(instance: &str) -> f64 {
    let mut file = File::open(METADATA).unwrap();
    let mut buffer = String::new();
    file.read_to_string(&mut buffer).unwrap();
    let data: serde_json::Value =
        serde_json::from_str(&buffer).expect("JSON was not well-formatted");
    data[instance]["best"].as_f64().unwrap()
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

/// Helper function to compute the best start solution from all possible ones
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

/// Helper function to create SearchParameters with SOME defaults
fn params_from(
    q: &QuboInstance,
    tr: f64,
    dls: f64,
    dbf: f64,
    dsf: f64,
    its: f64,
    bmns: f64,
    tl: usize
) -> SearchParameters {
    let af = ActivationFunction::Constant;
    //                 qubo ...  ...  ...  ... ...  ...   ... ... seed
    SearchParameters::new(q, tr, dls, dbf, dsf, af, its, bmns, tl, 42)
}

/// Analyze preprocessing using all instances
pub fn analyze_preproc() {
    println!("Run preprocessing analysis (omitting ineffective from table)");
    let mut file = File::open(METADATA).unwrap();
    let mut buffer = String::new();
    file.read_to_string(&mut buffer).unwrap();
    let data: serde_json::Value =
        serde_json::from_str(&buffer).expect("JSON was not well-formatted");
    println!(" name      &   size & density% & shrink% &   [ms] \\\\");
    let mut effective = 0;
    let mut ineffective = 0;
    for (name, val) in data.as_object().unwrap() {
        let qubo = QuboInstance::from_file(&filepath_from_name(name));
        let m = qubo.size();
        let now = std::time::Instant::now();
        let qubo = shrink(qubo);
        let elapsed = now.elapsed().as_millis();
        let n = qubo.size();
        if m == n { ineffective += 1; continue; }
        effective += 1;
        let shrink = 100.*((m - n) as f64)/(m as f64);
        let dens = val["density"].as_f64().unwrap();
        println!(
            "{name:10} & {m:6} &   {dens:6.2} &  {shrink:6.2} & {elapsed:>6?} \
            \\\\"
        );
    }
    let eff_prc = 100.*(effective as f64)/((effective + ineffective) as f64);
    println!("Percentage of effective preprocess runs: {eff_prc:.2}%");
}

/// Analyze start heuristics
pub fn analyze_start_heuristics() {
    let instances = [
        "bqp500.1",
        "bqp500.2",
        "bqp500.3",
        "bqp500.4",
        "bqp1000.1",
        "bqp1000.2",
        "bqp1000.3",
        "bqp1000.4",
        "bqp2500.1",
        "bqp2500.2",
        "G1",
        "G2",
        "G3",
        "G4",
        "G22",
        "G23",
        "G55",
        "p3000.1",
        "p3000.2",
        "p3000.3",
        "p3000.4",
        "p7000.2",
        "p7000.3",
    ];
    println!("Run start heuristics analysis");
    println!(
        "{:10} & {:6} & {:10} & {:10} & {:10} & {:10} & {:10} & {:10} \
        & {:10} & {:6} & {:5} & {:5} \\\\",
        "name", "size", "x0",   "x1",   "x2",   "r0",   "r1",   "r2",
        "lit",  "gap%",  "greedy[ms]", "random[ms]",
    );
    let mut total_goodness = Vector::from_vec(vec![0.; 6]);
    for inst in instances {
        let qubo = QuboInstance::from_file(&filepath_from_name(inst));
        let n = qubo.size();
        let (a, b, c) = create_hint_vecs(&qubo);
        let heuristics = [
            StartHeuristic::GreedyFromVec(a.clone()),
            StartHeuristic::GreedyFromVec(b.clone()),
            StartHeuristic::GreedyFromVec(c.clone()),
            StartHeuristic::RandomFromVec(a, 100, 42),
            StartHeuristic::RandomFromVec(b, 100, 42),
            StartHeuristic::RandomFromVec(c, 100, 42),
        ];
        let best_lit = get_literature_obj(inst);
        let mut goodness = Vector::from_vec(vec![0.; heuristics.len()]);
        let mut obj_vals = Vector::from_vec(vec![0.; heuristics.len()]);
        let mut avg_ms_greedy = 0;
        let mut avg_ms_random = 0;
        for k in 0..heuristics.len() {
            let now = std::time::Instant::now();
            let sol = heuristics[k].get_solution(&qubo);
            if k < 3 {
                avg_ms_greedy += now.elapsed().as_millis();
            }
            else {
                avg_ms_random += now.elapsed().as_millis();
            }
            obj_vals[k] = qubo.compute_objective(&sol);
            goodness[k] = 100.*obj_vals[k]/best_lit;
            total_goodness[k] += goodness[k];
        }
        avg_ms_greedy /= 3;
        avg_ms_random /= 3;
        let gap = 100. - goodness.max().unwrap();
        let x0 = obj_vals[0];
        let x1 = obj_vals[1];
        let x2 = obj_vals[2];
        let r0 = obj_vals[3];
        let r1 = obj_vals[4];
        let r2 = obj_vals[5];
        println!(
            "{:10} & {:6} & {:10} & {:10} & {:10} & {:10} & {:10} & {:10} \
            & {:10} & {:6.2} & {:5} & {:5} \\\\",
            inst,    n,     x0,     x1,     x2,     r0,     r1,     r2,
            best_lit,  gap,  avg_ms_greedy, avg_ms_random,
        );
    }
    let best = total_goodness.argmax().unwrap();
    let avg = total_goodness[best]/(instances.len() as f64);
    println!("x{best} is best with avg. goodness of {:.2}", avg);
}

/// Tune variable tabu search parameters except tenure ratio
pub fn tune_tabu_params() {
    let instances = [
        "bqp1000.1",
        "bqp1000.2",
        "bqp1000.3",
        "bqp1000.4",
        "bqp1000.5",
        "bqp1000.6",
    ];
    println!("Run tabu parameter tuning");
    // Parameters to tune
    let dls  = [0.05,  0.2,  0.5];   // diversification length scale
    let dbf  = [0.1,   0.25, 0.5,];  // diversification base factor
    let its  = [1.0,   5.0, 10.0,];  // improvements threshold scale
    let bmns = [0.005, 0.01];        // blocking move number scale
    // Set constant tenure ratio and dsf
    let tr = 0.01;
    let dsf = 1.5;
    let mut total_goodness = Array4::from_elem((3, 3, 3, 2), 0.);
    // Use small instances here
    for inst in instances {
        println!("START FOR {inst}");
        let best_lit = get_literature_obj(inst);
        let qubo = QuboInstance::from_file(&filepath_from_name(inst));
        let m = qubo.size();
        let qubo = shrink(qubo);
        let n = qubo.size();
        println!("Size shrunk from {m} to {n}");
        let time_limit_secs = 5;
        let start_solution = compute_best_start_solution(&qubo);
        //let mut obj_vals = Array4::from_elem((3, 3, 3, 2), 0.);
        let mut goodness = Array4::from_elem((3, 3, 3, 2), 0.);
        // Iterate over all possible combinations
        for (i, dls) in dls.iter().enumerate() {
            for (j, dbf) in dbf.iter().enumerate() {
                for (k, its) in its.iter().enumerate() {
                    for (l, bmns) in bmns.iter().enumerate() {
                        let params = params_from(
                            &qubo,
                            tr,
                            *dls,
                            *dbf,
                            dsf,
                            *its,
                            *bmns,
                            time_limit_secs,
                        );
                        let solution =
                            tabu_search(&qubo, &start_solution, LOG_LEVEL, params);
                        let indices = [i, j, k, l];
                        //obj_vals[indices] = qubo.compute_objective(&solution);
                        let obj = qubo.compute_objective(&solution);
                        goodness[indices] = 100.*obj/best_lit;
                        total_goodness[indices] += goodness[indices];
                    }
                }
            }
        }
        println!("Computed goodness values for {inst}:\n{goodness:?}");
    }
    let best = total_goodness.argmax().unwrap();
    let avg = total_goodness[best]/(instances.len() as f64);
    println!(
        "best overall choice: {}, {}, {}, {} with avg. goodness {avg}",
        dls[best.0],
        dbf[best.1],
        its[best.2],
        bmns[best.3],
    );
}

/// Tune dsf only
pub fn tune_dsf() {
    let instances = [
        "bqp1000.5",
        "bqp1000.6",
        "bqp1000.7",
        "bqp1000.8",
        "bqp1000.9",
    ];
    println!("Run dsf tuning");
    let dsf = [1.1, 1.25,  1.5];
    let mut total_goodness = Array1::from_elem(dsf.len(), 0.);
    let time_limit_secs = 30;
    // Use constant tenure ratio
    let tr = 0.01;
    // Use best params given by param_tuning
    let dls = 0.05;
    let dbf = 0.5;
    let its = 1.;
    let bmns = 0.005;
    for inst in instances {
        let best_lit = get_literature_obj(inst);
        let qubo = QuboInstance::from_file(&filepath_from_name(inst));
        let m = qubo.size();
        let qubo = shrink(qubo);
        let n = qubo.size();
        println!("Size shrunk from {m} to {n}");
        let start_solution = compute_best_start_solution(&qubo);
        let mut goodness = Array1::from_elem(dsf.len(), 0.);
        for i in 0..dsf.len() {
            // Use best params given by param_tuning
            let params = params_from(
                &qubo, tr, dls, dbf, dsf[i], its, bmns, time_limit_secs
            );
            let solution =
                tabu_search(&qubo, &start_solution, LOG_LEVEL, params);
            let obj = qubo.compute_objective(&solution);
            goodness[i] = 100.*obj/best_lit;
            total_goodness[i] += goodness[i];
        }
    }
    let best_i = total_goodness.argmax().unwrap();
    let dsf = dsf[best_i];
    let avg = total_goodness[best_i]/(instances.len() as f64);
    println!("\nbest dsf={dsf} with {avg}");
}

/// Tune tenure ratio only
pub fn tune_tr() {
    // Use bigger instances here
    let instances = [
        "bqp1000.5",
        "bqp1000.6",
        "bqp1000.7",
        "bqp1000.8",
        "bqp1000.9",
    ];
    println!("Run tenure ratio tuning");
    let tr = [0.0, 0.05,  0.2,  0.5, 1.0];
    let mut total_goodness = Array1::from_elem(tr.len(), 0.);
    let time_limit_secs = 30;
    // Use best dsf and params given by previous param tunings
    let dls = 0.05;
    let dbf = 0.5;
    let dsf = 1.1;
    let its = 1.;
    let bmns = 0.005;
    for inst in instances {
        let best_lit = get_literature_obj(inst);
        let qubo = QuboInstance::from_file(&filepath_from_name(inst));
        let m = qubo.size();
        let qubo = shrink(qubo);
        let n = qubo.size();
        println!("Size shrunk from {m} to {n}");
        let start_solution = compute_best_start_solution(&qubo);
        let mut goodness = Array1::from_elem(tr.len(), 0.);
        for i in 0..tr.len() {
            // Use best params given by param_tuning
            let params = params_from(
                &qubo, tr[i], dls, dbf, dsf, its, bmns, time_limit_secs
            );
            let solution
                = tabu_search(&qubo, &start_solution, LOG_LEVEL, params);
            let obj = qubo.compute_objective(&solution);
            goodness[i] = 100.*obj/best_lit;
            total_goodness[i] += goodness[i];
        }
    }
    let best_i = total_goodness.argmax().unwrap();
    let tr = tr[best_i];
    let avg = total_goodness[best_i]/(instances.len() as f64);
    println!("\nbest tr={tr} with {avg}");
}

/// Analyze tabu search
pub fn analyze_tabu_search() {
    let instances = [
        "bqp500.1",
        "bqp500.2",
        "bqp500.3",
        "bqp500.4",
        "bqp1000.1",
        "bqp1000.2",
        "bqp1000.3",
        "bqp1000.4",
        "bqp2500.1",
        "bqp2500.2",
        "G1",
        "G2",
        "G3",
        "G4",
        "G22",
        "G23",
        "G55",
        "p3000.1",
        "p3000.2",
        "p3000.3",
        "p3000.4",
        "p7000.2",
        "p7000.3",
    ];
    println!("Run tabu search analysis");
    // Use best params given by tune_tabu_params(), tune_tr() and tune_dsf()
    let tr = 0.05;
    let dls = 0.05;
    let dbf = 0.5;
    let dsf = 1.1;
    let its = 1.;
    let bmns = 0.005;
    let time_limit_secs = 3600; // 1h
    let mut goodness = Array1::from_elem(instances.len(), 0.);
    let mut obj_vals = Array1::from_elem(instances.len(), 0.);
    let mut times = Array1::from_elem(instances.len(), 0);
    for (i, inst) in instances.iter().enumerate() {
        let best_lit = get_literature_obj(inst);
        let qubo = QuboInstance::from_file(&filepath_from_name(inst));
        let m = qubo.size();
        let qubo = shrink(qubo);
        let n = qubo.size();
        println!("Size shrunk from {m} to {n}");
        let start_solution = compute_best_start_solution(&qubo);
        // Use best params and tenure_ratio
        let params =
            params_from(&qubo, tr, dls, dbf, dsf, its, bmns, time_limit_secs);
        let now = std::time::Instant::now();
        let solution = tabu_search(&qubo, &start_solution, LOG_LEVEL, params);
        times[i] = now.elapsed().as_millis();
        obj_vals[i] = qubo.compute_objective(&solution);
        goodness[i] = 100.*obj_vals[i]/best_lit;
    }
    // Table printing
    println!(
        "{:10} & {:6} & {:10} & {:10} & {:6} & {:5} \\\\",
        "name", "size", "obj",  "lit",  "gap%",  "[ms]",
    );
    for (i, inst) in instances.iter().enumerate() {
        let best_lit = get_literature_obj(inst);
        let qubo = QuboInstance::from_file(&filepath_from_name(inst));
        let m = qubo.size();
        let g = goodness[i];
        let obj = obj_vals[i];
        let gap = 100. - g;
        let elapsed = times[i];
        println!(
            "{:10} & {:6} & {:10} & {:10} & {:6.2} & {:5} \\\\",
            inst,    m,     obj,   best_lit,  gap,  elapsed,
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bqp500_2() {
        let inst = "bqp500.2";
        let qubo = QuboInstance::from_file(&filepath_from_name(inst));
        let qubo = shrink(qubo);
        let params = SearchParameters::new(
            &qubo,
            0.05,
            0.05,
            0.5,
            1.1,
            ActivationFunction::Constant,
            1.,
            0.005,
            3600,
            42,
        );
        let best_lit = get_literature_obj(inst);
        let start_solution = compute_best_start_solution(&qubo);
        let solution = tabu_search(&qubo, &start_solution, LOG_LEVEL, params);
        let obj = qubo.compute_objective(&solution);
        let goodness = 100.*obj/best_lit;
        assert!(goodness > 100.0);
        let known_solution = BinaryVector::from_vec(vec![
            false,
            true, true, true, true, true, true, false, true, true, true, true,
            false, true, true, false, false, true, false, true, true, false,
            false, true, true, false, true, false, false, true, true, false,
            true, true, true, true, false, false, true, true, false, true,
            false, false, true, false, true, false, true, false, false, true,
            true, false, false, true, true, true, false, true, true, true,
            false, false, true, true, false, true, false, true, true, true,
            true, true, true, true, false, false, false, true, true, true, true,
            true, true, true, true, true, false, true, true, false, true, true,
            true, true, false, true, true, true, true, true, true, true, true,
            true, true, false, true, false, true, true, false, false, false,
            true, false, false, true, true, false, true, true, true, false,
            true, true, false, true, true, true, false, true, true, false, true,
            true, true, false, true, false, true, true, false, true, true, true,
            true, false, true, true, false, true, false, true, true, false,
            false, false, true, true, true, true, true, true, false, false,
            true, false, true, false, true, true, true, false, true, true, true,
            false, false, true, true, false, false, true, true, true, false,
            true, true, true, true, true, true, false, true, true, true, true,
            false, true, true, false, true, false, true, false, true, true,
            true, true, true, true, true, true, false, false, true, true, true,
            true, false, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, false, true, true, true, false, false,
            false, true, false, true, true, true, true, false, false, true,
            true, true, true, true, true, false, true, true, true, true, true,
            true, true, true, true, false, true, false, false, true, true, true,
            true, false, true, false, false, false, true, true, true, false,
            false, true, true, true, true, false, false, true, true, true, true,
            true, true, false, true, true, false, true, true, true, true, true,
            false, false, true, false, true, false, true, false, true, true,
            true, true, true, true, true, true, false, true, true, true, false,
            true, false, false, false, false, false, true, true, true, false,
            true, true, true, true, false, false, true, true, false, true,
            false, false, true, false, true, true, false, true, false, true,
            true, true, false, true, false, true, true, true, true, false,
            false, false, true, false, true, true, true, true, false, true,
            true, true, true, true, false, false, true, false, false, true,
            true, true, true, true, true, true, false, false, false, true,
            false, false, false, true, false, false, true, false, true, true,
            false, false, false, true, true, true, false, false, false, true,
            true, false, true, true, true, false, true, false, false, true,
            true, true, true, false, false, true, true, true, true, true, false,
            true, false, false, false, false, true, true, true, false, false,
            true, true, false, true, true, false, false, true, true, true, true,
            true, true, false, true, true, true, false, false, false, false,
            true, false, true, false, true, true, true, false, true, false,
            false, true, false, true, false, true, false, true, true, true,
            true, true, true, true, true, false, true, false, true, true, false,
        ]);
        let known_obj = qubo.compute_objective(&known_solution);
        assert!(known_obj == obj);
    }
}
