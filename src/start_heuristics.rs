/// QUBO start heuristic enum

use crate::qubo::*;

/// Computes the "sum cross" of x^T@Q@x for index k efficiently: O(n)
fn compute_sum_cross(mat: &Matrix, x: &BinaryVector, k: usize) -> Float {
    let mut y = Vector::zeros(x.len());
    for i in 0..x.len() {
        y[i] = x[i] as u8 as Float;
    }
    compute_sum_cross_float(mat, &y, k)
}

/// Computes the "sum cross" of x^T@Q@x for index k efficiently: O(n)
fn compute_sum_cross_float(mat: &Matrix, x: &Vector, k: usize) -> Float {
    assert!(k <= mat.nrows());
    let xk = x[k];
    let mut sum_cross = 0.0;
    for i in 0..k {
        let xi = x[i]; // false => 0, true => 1
        sum_cross += mat[[i, k]]*xi*xk;
        sum_cross += mat[[k, i]]*xi*xk;
    }
    sum_cross + mat[[k, k]]*xk*xk
}

/// Cast Vector (Float) to BinaryVector (bool)
fn vecf_to_vecb(x: Vector) -> BinaryVector {
    let n = x.len();
    let mut y = BinaryVector::from_vec(vec![false; n]);
    for i in 0..n {
        y[i] = x[i] as u8 != 0;
    }
    y
}

pub enum StartHeuristic {
    Random(),
    GreedyRounding(Float),
}
impl StartHeuristic {
    pub fn get_solution(&self, qubo: &QuboInstance) -> BinaryVector {
        match self {
            StartHeuristic::Random()
                => { StartHeuristic::rand(qubo) },
            StartHeuristic::GreedyRounding(hint)
                => { StartHeuristic::round_greedy(qubo, hint) },
        }
    }

    /// Return a random start solution
    fn rand(qubo: &QuboInstance) -> BinaryVector {
        todo!()
    }

    /// Find best rounding at some index until all indices are rounded
    fn round_greedy(qubo: &QuboInstance, hint: &Float) -> BinaryVector {
        let n = qubo.size();
        let mat = qubo.get_matrix();
        let mut hint_vec = Vector::from_vec(vec![*hint; n]);
        // Changes on round of entry:
        let mut on_round_up = Vector::from_vec(vec![0.0; n]);
        let mut on_round_down = Vector::from_vec(vec![0.0; n]);
        // Change minima and indices
        let mut best_up = Float::MAX;
        let mut best_down = Float::MAX;
        let mut i_up = 0;
        let mut i_down = 0;
        // Find best up/down rounding (with biggest downward obj change)
        for _ in 0..n {
            // Compute changes to obj_val on 
            for k in 0..n {
                // compute change on round up of hint_vec[k]
                hint_vec[k] = 1.0;
                on_round_up[k] = compute_sum_cross_float(mat, &hint_vec, k);
                if on_round_up[k] < best_up {
                    best_up = on_round_up[k];
                    i_up = k;
                }
                // compute change on round down of hint_vec[k]
                hint_vec[k] = 0.0;
                on_round_down[k] = compute_sum_cross_float(mat, &hint_vec, k);
                if on_round_down[k] < best_down {
                    best_down = on_round_down[k];
                    i_down = k;
                }
                // Reset hint_vec[k]
                hint_vec[k] = *hint;
            }
            // Apply best rounding
            if best_up < best_down {
                // Round up
                hint_vec[i_up] = 1.0;
            }
            else {
                // Round down
                hint_vec[i_down] = 0.0;
            }
        }
        // Cast hint_vec to BinaryVector
        vecf_to_vecb(hint_vec)
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
        let x = BinaryVector::from_vec(vec![true, false, true]);
        todo!();
    }
}
