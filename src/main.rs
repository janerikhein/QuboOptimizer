mod qubo {
    use ndarray::{Array1, Array2};

    pub struct QuboInstance {
        // upper triangular Matrix Q
        q: Array2<f64>,
        // boolean solution vector
        solution: Option<Array1<bool>>,
        // objective value
        obj_val : Option<f64>,
        // objective changes when flipping variables
        flip_values : Option<Array1<f64>>,
    }

    // TODO: Discuss whther should allow for generic types instead of only f64.
    impl QuboInstance {
        pub fn new(q: Array2<f64>, solution: Option<Array1<bool>>) -> QuboInstance {
            let mut inst = QuboInstance {q, solution: None, obj_val: None, flip_values: None};
            if solution.is_some() {
                inst.set_solution(solution.unwrap());
            }
            inst
        }

        // TODO: test this
        // Set a new solution to the instance
        pub fn set_solution(&mut self, solution: Array1<bool>) {
            assert_eq!(solution.len(), self.nvars());
            if self.solution.is_none() {
                // Initilize to zero vector
                self.solution = Some(Array1::zeros(self.nvars()));
                // For all zero solution, flip values are given by values of Q on diagonal
                self.flip_values = Some(self.q.diag().to_owned());
                // objective for zero vector is zero
                self.obj_val = Some(0.0);
            }
            let cur_sol = self.solution.unwrap();
            for (index, (&val1, &val2)) in cur_sol.iter().zip(solution.iter()).enumerate() {
                if val1 != val2 {
                    self.flip(index as u32);
                }
            }
        }

        // number of variables
        pub fn nvars(&self) -> u32 {
            // TODO: return size of Q, i.e. number of rows or columns
            todo!()
        }

        pub fn get_obj(&self) -> f64 {
            self.obj_val.expect("No solution set for QuboInstance.")
        }

        // get objective difference of applying a flip to a given xi
        pub fn get_obj_delta_on_flip(&self, flip_index: u32) -> f64 {
            assert!(flip_index < self.nvars());
            self.flip_values.expect("No solution set for QuboInstance.")[flip_index]
        }

        // Apply a variable flip for a given xi
        pub fn flip(&mut self, flip_index: u32) {
            // TODO: modify flip values accordingly
            // TODO: update objective value
            //

            todo!()
        }

        pub fn is_active(&self, i: u32, j: u32) {
            self.solution[i] & self.solution[j]
        }

    }
}


mod qubo_start_heuristic {
    use ndarray::Array1;
    use crate::qubo::QuboInstance;

    // TODO: add other start heuristics to compare with here
    pub enum StartHeuristic {
        // Select a uniform (p=0.5) random 0-1-vector as start solution
        // TODO: to be implemented
        Random,
        // Greedy rounding of fractional constant start solution [x,x,x...] where 0<x<1 is a parameter
        // TODO: to be implemented, parmeter is just placeholder
        GreedyRounding(f32),
    }

    impl StartHeuristic {
        pub fn get_solution(&self, qubo: &QuboInstance) -> Array1<bool> {
            match self {
                StartHeuristic::Random => {StartHeuristic::get_solution_rand(qubo)}
                //TODO: add (parameterized) heuristics here
                StartHeuristic::GreedyRounding(plch) => {StartHeuristic::get_solution_greedy_rounding(qubo, plch)}
            }
        }

        pub fn set_solution(&self, qubo: &mut QuboInstance) {
            let solution = self.get_solution(qubo);
            qubo.set_solution(solution)
        }

        fn get_solution_rand(qubo: &QuboInstance) -> Array1<bool> {
            todo!()
        }

        fn get_solution_greedy_rounding(qubo: &QuboInstance, hint: &f32) -> Array1<bool> {
            todo!()
            // let n = qubo.nvars();
            // let mut x = vec![0.5; n];
            // let mut min = func(&triang, &x);
            // for i in 0..n {
            //     x[i] = 0.0;
            //     let tmp0 = func(&triang, &x);
            //     x[i] = 1.0;
            //     let tmp1 = func(&triang, &x);
            //     if tmp0 < tmp1 {
            //         min = tmp0;
            //         x[i] = 0.0;
            //     }
            //     else {
            //         min = tmp1;
            //     }
            // }
            // min
        }
    }

}

mod tabu_search {

}


fn main() {

}
