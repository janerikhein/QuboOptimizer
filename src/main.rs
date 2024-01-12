/// Run QUBO heuristics tests 
/// Paul Meinhold, Jan-Erik
/// 12.01.24 Scientific Computing 23/24

mod qubo;
use qubo::{BinaryVector, Vector, Matrix, QuboInstance, QuboHeuristics};

fn main() {
    let matrix = Matrix::from_diag(&Vector::from_vec(vec![-3.0, 2.0, -2.0]));
    let correct_solution = BinaryVector::from_vec(vec![true, false, true]);
    let qubo = QuboInstance::new(matrix, 0.0);
    let heur = QuboHeuristics::new(qubo);

    //let solution = StartHeuristic::GreedyRounding(0.5).get_solution(&qubo);
    //assert_eq!(solution, correct_solution);
}
