mod qubo {
    use ndarray::{Array1, Array2};

    pub struct QuboInstance {
        // upper triangular Matrix Q
        q: Array2<f64>,
        // boolean solution vector
        solution: Option<Array1<bool>>,
        // objective value, to be minimized
        obj_val : Option<f64>,
        // objective changes to obj_val when flipping variables
        flip_values : Option<Array1<f64>>,
    }

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

        // TODO: test this
        pub fn get_matrix(&self) -> Array2 {
            self.q
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
        GreedyRounding(f64),
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

        /// This is O(n^2) TODO: Test
        fn get_solution_greedy_rounding(qubo: &QuboInstance, hint: &f64) -> Array1<bool> {
            let n = qubo.nvars();
            let q = qubo.get_matrix();
            let mut hint_vec = Array1::from_elem((n), hint);
            //let mut diff = Array1::from_elem((n), 0.0);
            let mut obj_val = 0.0;
            // Calculate the obj_val for the entire hint vector in O(n^2)
            for i in 0..n {
                for j in 0..n {
                    obj_val = q[i,j]*hint*hint;
                }
            }
            // TODO: randomize array access via random permutation
            let mut solution = Array1::from_elem((n), false);
            for k in 0..n {
                // subtract old "sum cross"
                for i in 0..n {
                    obj_val -= q[i,k]*hint_vec[i]*hint_vec[k];
                }
                for j in 0..n {
                    if j == k { continue; } // already subtracted for [k,k]
                    obj_val -= q[k,j]*hint_vec[j]*hint_vec[k];
                }
                // add new "sum cross" for hint_vec[k] = 0, which is 0
                // add new "sum cross" for hint_vec[k] = 1
                let mut obj_val_true = obj_val;
                for i in 0..n {
                    obj_val_true += q[i,k]*hint_vec[i]; // * 1
                }
                for j in 0..n {
                    if j == k { continue; } // already added for [k,k]
                    obj_val_true += q[k,j]*hint_vec[j]; // * 1
                }
                // Set solution at [k]
                if obj_val_true < obj_val {
                    solution[k] = true;
                    obj_val = obj_val_true;
                }
            }
            solution
        }
    }

}

mod tabu_search {

}


fn main() {

}
