/// Run QUBO experiments
/// Lukas Mehl, Paul Meinhold, Jan-Erik Hein
/// 12.01.24 Scientific Computing 23/24
/// Literature: https://pads.ccc.de/QUwrTGlwvn

pub mod qubo;
pub mod preprocess;
pub mod start_heuristics;
pub mod tabu_search;
pub mod experiments;

fn main() {
    // We use the standard arguments to define the experiment
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        panic!("usage: {} experiment_num", &args[0]);
    }
    let experiment_num: u8 = args[1].parse().expect("No valid experiment_num");
    // Match experiment_num with experiment function
    match experiment_num {
        1 => { experiments::analyze_preproc(); },
        2 => { experiments::analyze_start_heuristics(); },
        3 => { experiments::tune_tabu_params(); },
        4 => { experiments::tune_tr(); },
        5 => { experiments::analyze_tabu_search(); },
        _ => { panic!("No valid experiment_num"); }
    }
}
