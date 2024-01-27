/// QUBO instance struct and useful types

use ndarray::{Array1, Array2};

/// Useful type definitions
pub type Float = f64;
pub type Vector = Array1<Float>;
pub type Matrix = Array2<Float>;
pub type BinaryVector = Array1<bool>;
pub type IntegerVector = Array1<u32>;
pub type IntegerMatrix = Array2<u32>;

pub struct QuboInstance {
    // Upper triangular square matrix
    mat: Matrix,
    // Baseline objective value that cannot be further optimized
    baseline: Float,
}
impl QuboInstance {
    /// Default initilize
    pub fn new(mat: Matrix, baseline: Float) -> Self {
        // TODO: Check for square and triangular
        Self { mat, baseline }
    }

    /// Random matrix initilize
    pub fn new_rand(n: usize, density: Float) -> Self {
        todo!();
        //Self { mat, baseline }
    }

    /// Initialize from problem instance file
    pub fn from_file(file_path: &str) -> Self {
        todo!();
    }

    /// Computes the objective value for a given BinaryVector, inefficient?
    pub fn compute_objective(&self, x: BinaryVector) -> Float {
        todo!();
    }

    /// Returns matrix size, i.e. number of rows or columns
    pub fn size(&self) -> usize {
        self.mat.nrows()
    }

    /// Returns the matrix
    pub fn get_matrix(&self) -> &Matrix {
        &self.mat
    }

    pub fn get_entry_at(&self, i: usize, j: usize) -> Float {
        assert!(j >= i);
        self.mat[[i,j]]
    }
}
