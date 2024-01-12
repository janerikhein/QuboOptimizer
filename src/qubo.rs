/// QUBO (Quadratic Binary Optimization) problem instance and heuristics structs
/// Scientific Computing 23/24

/// Literature: https://pads.ccc.de/QUwrTGlwvn

use ndarray::{Array1, Array2};
use ndarray_stats::QuantileExt

/// Optional type definitions
type Float = f64;
pub type Vector = Array1<Float>;
pub type Matrix = Array2<Float>;
pub type BinaryVector = Array1<bool>;

/// Computes the "sum cross" of x^T@Q@x for index k efficiently: O(n)
fn compute_sum_cross(triang: &Matrix, x: &BinaryVector, k: usize) -> Float {
    let mut y = Vector::zeros(x.len());
    for i in 0..x.len() {
        y[i] = x[i] as u8 as Float;
    }
    compute_sum_cross_float(&triang, &y, k)
}

/// Computes the "sum cross" of x^T@Q@x for index k efficiently: O(n)
fn compute_sum_cross_float(triang: &Matrix, x: &Vector, k: usize) -> Float {
    assert!(k <= triang.nrows());
    let xk = x[k];
    let mut sum_cross = 0.0;
    for i in 0..k {
        let xi = x[i]; // false => 0, true => 1
        sum_cross += triang[[i, k]]*xi*xk;
        sum_cross += triang[[k, i]]*xi*xk;
    }
    sum_cross + triang[[k, k]]*xk*xk
}

pub struct QuboInstance {
    // Upper triangular square matrix
    triang: Matrix,
    // Baseline objective value that cannot be further optimized
    baseline: Float,
}
impl QuboInstance {
    /// Default initilize
    pub fn new(triang: Matrix, baseline: Float) -> Self {
        Self { triang, baseline }
    }

    /// Initialize from matrix file
    pub fn from_file(file_path: &str) -> Self {
        todo!();
    }

    /// Returns the objective value for a given BinaryVector
    pub fn get_objective(&self, x: BinaryVector) -> Float {
        todo!();
    }

    /// Returns matrix size, i.e. number of rows or columns
    pub fn size(&self) -> usize {
        self.triang.nrows()
    }

    /// Returns the matrix
    pub fn get_matrix(&self) -> Matrix {
        self.triang
    }

    /// Preprocessing rules 1-5, maybe just as standalone-functions?
    pub fn shrink(&mut self) {
        todo!();
    }
}

pub struct QuboHeuristics {
    qubo: QuboInstance,
    flip_values: Vector, //or obj_on_flip: Vector,
    solution: BinaryVector, // best current boolean solution vector
}
impl QuboHeuristics {
    /// Default initilize
    pub fn new(qubo: QuboInstance) -> Self {
        let n = qubo.size();
        let flip_values = Vector::from_vec(vec![0.0; n]);
        let solution = BinaryVector::from_vec(vec![false; n]);
        Self { qubo, flip_values, solution }
    }

    /// Start heuristic 1
    pub fn greedy_rounding(&mut self, hint: Float) {
        let n = self.qubo.size();
        let triang = qubo.get_matrix();
        let mut hint_vec = Vector::from_vec(vec![hint; n]);
        let mut obj_val = 0.0;
        let mut obj_on_round_up = Vector::from_vec(vec![0.0; n]);
        let mut obj_on_round_down = Vector::from_vec(vec![0.0; n]);
        for k in 0..n {
            obj += compute_sum_cross_float(triang, hint_vec, k);
            hint_vec[k] = 1.0;
            obj_on_round_up += compute_sum_cross_float(triang, hint_vec, k);
            hint_vec[k] = 0.0;
            obj_on_round_down += compute_sum_cross_float(triang, hint_vec, k);
        }
        // TODO: Find best rounding with obj_on_round_*, then apply to hint_vec
        // repeat until each hint_vec[i] is rounded.
        for _ in 0..n {
            todo!();
        }
        let argmin_up = obj_on_round_up.argmin().unwrap();
        let argmin_down = obj_on_round_down.argmin().unwrap();
        todo!();
    }

    /// Do basic tabu search
    pub fn tabu_search(&mut self) {
        todo!();
    }

    /// Do an optimized tabu search
    pub fn staged_tabu_search(&mut self) {
        todo!();
    }

    /// Set a solution and update flip values
    fn set_solution(&mut self, solution: BinaryVector) {
        todo!();
    }

    /// get objective difference of applying a flip to a given xi
    fn get_obj_delta_on_flip(&self, flip_index: usize) -> f64 {
        todo!();
    }

    /// Apply a variable flip for a given xi
    fn flip(&mut self, flip_index: usize) {
        todo!();
    }

    /// ?
    fn is_active(&self, i: usize, j: usize) {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use crate::qubo::*;

    #[test]
    fn test_sum_cross() {
        let x = BinaryVector::from_vec(vec![true, false, true]);
        let matrix = Matrix::from_shape_vec((3,3),
            vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0,]).unwrap();
        let mut obj_val = 0.0;
        for i in 0..matrix.nrows() {
            obj_val += compute_sum_cross(&matrix, &x, i);
        }
        assert_eq!(10.0, obj_val);
    }

    #[test]
    fn test_greedy_rounding() {
        todo!();
    }
}
