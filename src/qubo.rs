/// QUBO instance struct and useful types

use std::io::{BufRead,BufReader};
use std::fs::File;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use rand_pcg::Pcg32;

/// Useful type definitions
pub type Vector = Array1<f64>;
pub type Matrix = Array2<f64>;
pub type BinaryVector = Array1<bool>;

pub struct QuboInstance {
    // Upper triangular square matrix
    mat: Matrix,
    // Baseline objective value that cannot be further optimized
    baseline: f64,
}
impl QuboInstance {
    /// Default initilize
    pub fn new(mat: Matrix, baseline: f64) -> Self {
        let n = mat.nrows();
        for i in 0..n {
            for j in 0..i {
                if mat[[i, j]] != 0.0 { panic!("Matrix not upper triangular"); }
            }
        }
        Self { mat, baseline }
    }

    /// Instance with matrix having uniformly random entries in [-10, 10]
    pub fn new_rand(n: usize, density: f64) -> Self {
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
        let f = File::open(file_path).expect("Couldn't open file");
        let mut reader = BufReader::new(f);
        let mut buffer = String::new();
        // Read number vertices and edges
        let _ = reader.read_line(&mut buffer).expect("Error reading line");
        let header: Vec<&str> = buffer.split_whitespace().collect();
        let n = header[0].parse().unwrap();
        let mut mat = Matrix::from_shape_vec((n, n), vec![0.0; n*n]).unwrap();
        // Read each entry
        for _ in 0..n*n {
            buffer.clear();
            let result = reader.read_line(&mut buffer);
            match result {
                Ok(0)  => { break; },
                Ok(_)  => { },
                Err(_) => { panic!("Error reading file"); }
            }
            let entry: Vec<&str> = buffer.split_whitespace().collect();
            let row = entry[0].parse().unwrap();
            let col = entry[1].parse().unwrap();
            // Transpose due to file containing lower-triang
            mat[[col, row]] = entry[2].parse().unwrap()
        }
        Self { mat, baseline: 0.0 }
    }

    /// Computes the objective value for a given BinaryVector
    pub fn compute_objective(&self, x: BinaryVector) -> f64 {
        let n = self.mat.nrows();
        let mut obj_val = 0.0;
        for i in 0..n {
            for j in i..n {
                obj_val += self.get_entry_at(i, j)
                    *(x[i] as u8 as f64)
                    *(x[j] as u8 as f64);
            }
        }
        obj_val + self.baseline
    }

    /// Returns matrix size, i.e. number of rows or columns
    pub fn size(&self) -> usize {
        self.mat.nrows()
    }

    /// Returns the matrix
    pub fn get_matrix(&self) -> &Matrix {
        &self.mat
    }

    pub fn get_entry_at(&self, i: usize, j: usize) -> f64 {
        if j < i {
            panic!("Don't access lower (0) entries of upper triangular\
                matrix"); }
        self.mat[[i,j]]
    }
}
