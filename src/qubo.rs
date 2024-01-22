/// QUBO instance struct and useful types

use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use rand_pcg::Pcg32;

/// Useful type definitions
pub type Float = f64;
pub type Vector = Array1<Float>;
pub type Matrix = Array2<Float>;
pub type BinaryVector = Array1<bool>;

pub struct QuboInstance {
    // Upper triangular square matrix
    mat: Matrix,
    // Baseline objective value that cannot be further optimized
    baseline: Float,
}
impl QuboInstance {
    /// Default initilize
    pub fn new(mat: Matrix, baseline: Float) -> Self {
        let n = mat.nrows();
        for i in 0..n {
            for j in 0..i {
                if mat[[i, j]] != 0.0 { panic!("Matrix not upper triangular"); }
            }
        }
        Self { mat, baseline }
    }

    /// Instance with matrix having uniformly random entries in [-10, 10]
    pub fn new_rand(n: usize, density: Float) -> Self {
        let mut rng = Pcg32::seed_from_u64(42);
        let mut mat = Matrix::random_using(
            (n, n), Uniform::new(-10.0, 10.0), &mut rng);
        for i in 0..n {
            for j in 0..i {
                mat[[i, j]] = 0.0;
            }
        }
        Self { mat, baseline: 0.0 }
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
