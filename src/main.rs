use ndarray::{Array1, Array2};


mod qubo {
    use ndarray::{Array1, Array2};
    use crate::qubo_start_heuristic::StartHeuristic;

    pub struct QuboInstance {
        // TODO: Add relevant fields. Note: make all fields private, i.e without pub, as changing fields will require updating others
        // upper triangular Matrix Q
        Q : Array2<f64>,
        // objective sense
        max : bool,
        // boolean solution vector
        solution : Array1<bool>
    }

    // TODO: Discuss whther should allow for generic types instead of only f64.
    impl QuboInstance {
        pub fn new(Q: Array2<f64>, maximize: bool, solution: Array1<bool>) -> QuboInstance {
            todo!()
        }

        // number of variables
        pub fn nvars(&self) -> u32 {
            // TODO: return size of Q, i.e. number of rows or columns
            todo!()
        }


    }
}


mod qubo_start_heuristic {
    use crate::qubo::QuboInstance;

    // TODO: add other start heuristics to compare with here
    pub enum StartHeuristic {
        // Select a uniform (p=0.5) random 0-1-vector as start solution
        // TODO: to be implemented
        Random,
        // Greedy rounding of fractional constant start solution [x,x,x...] where 0<x<1 is a parameter
        // TODO: to be implemented
        GreedyRounding(f32),
    }

    impl StartHeuristic {
        pub fn get_solution(&self, qubo: QuboInstance) {
            match self {
                StartHeuristic::Random => {StartHeuristic::get_solution_rand(qubo)}
                StartHeuristic::GreedyRounding(hint) => {StartHeuristic::get_solution_greedy_rounding(qubo, hint)}
            }
        }

        fn get_solution_rand(qubo: QuboInstance) {
            todo!()
        }

        fn get_solution_greedy_rounding(qubo: QuboInstance, hint: &f32) {
            todo!()
        }
    }

}


fn main() {
    println!("Hello, world!");
}
