/// Run QUBO experiments
/// Lukas Mehl, Paul Meinhold, Jan-Erik Hein
/// 12.01.24 Scientific Computing 23/24
/// Literature: https://pads.ccc.de/QUwrTGlwvn

pub mod qubo;
pub mod preprocess;
pub mod start_heuristics;
pub mod tabu_search;
pub mod experiments;
use std::env;

fn main() {
    // We use the standard arguments to define the experiment
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        panic!("usage: {} experiment_num", &args[0]);
    }
    let experiment_num: u8 = args[1].parse().expect("No valid experiment_num");
    // Match experiment_num with experiment function
    match experiment_num {
        1 => { experiments::example(); },
        2 => { experiments::test_start_heuristics(); },
        3 => { experiments::test_tabu_search_params(); },
        _ => { panic!("No valid experiment_num"); }
    }
}
