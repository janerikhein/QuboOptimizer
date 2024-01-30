/// QUBO start heuristic enum

use crate::qubo::*;
use ndarray_stats::QuantileExt;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use rand_pcg::Pcg32;

/// Computes the "sum cross" of x^T@Q@x for index k efficiently: O(n)
fn compute_sum_cross_float(mat: &Matrix, x: &Vector, k: usize) -> f64 {
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

/// Cast Vector (f64) to BinaryVector (bool)
fn vecf_from_vecb(x: &Vector) -> BinaryVector {
    let n = x.len();
    let mut y = BinaryVector::from_vec(vec![false; n]);
    for i in 0..n {
        y[i] = x[i] as u8 != 0;
    }
    y
}

pub enum StartHeuristic {
    Random(u64),
    GreedyFromHint(f64),
    GreedyFromVec(Vector),
}
impl StartHeuristic {
    pub fn get_solution(&self, qubo: &QuboInstance) -> BinaryVector {
        match self {
            StartHeuristic::Random(seed)
                => { StartHeuristic::rand(qubo, seed) },
            StartHeuristic::GreedyFromHint(hint)
                => { StartHeuristic::greedy_from_hint(qubo, hint) },
            StartHeuristic::GreedyFromVec(hint_vec)
                => { StartHeuristic::greedy_from_vec(qubo, hint_vec) },
        }
    }

    /// Return a random start solution
    fn rand(qubo: &QuboInstance, seed: &u64) -> BinaryVector {
        let n = qubo.size();
        let mut rng = Pcg32::seed_from_u64(*seed);
        let mut solution = Vector::random_using(
            n, Uniform::new(0.0, 1.0), &mut rng);
        for i in 0..solution.len() {
            solution[i] = solution[i].round();
        }
        vecf_from_vecb(&solution)
    }

    /// Find best rounding at some index until all indices are rounded
    fn greedy_from_vec(qubo: &QuboInstance, hint_vec: &Vector) -> BinaryVector {
        // Make mutable copy
        let mut hint_vec = hint_vec.clone();
        let n = qubo.size();
        let mat = qubo.get_matrix();
        //let mut hint_vec = Vector::from_vec(vec![*hint; n]);
        // Changes on round of entry:
        let mut dx_on_round =
            Matrix::from_shape_vec((2, n), vec![0.0; 2*n]).unwrap();
        // Compute initial dx
        for i in 0..n {
            // Round down
            let hint_i = hint_vec[i];
            hint_vec[i] = 0.0;
            dx_on_round[[0, i]] = compute_sum_cross_float(mat, &hint_vec, i);
            // Round up
            hint_vec[i] = 1.0;
            dx_on_round[[1, i]] = compute_sum_cross_float(mat, &hint_vec, i);
            // Undo rounding
            hint_vec[i] = hint_i;
        }
        for _ in 0..n {
            // Find next best k for up/down rounding (for biggest downward dx)
            let (r, k) = dx_on_round.argmin().unwrap();
            // Round to 0 or 1
            hint_vec[k] = r as f64;
            // Update dx at k to MAX so it is not found as minimum again
            dx_on_round[[0, k]] = f64::MAX;
            dx_on_round[[1, k]] = f64::MAX;
            let hint_k = hint_vec[k];
            // Update dx at unvisited columns to respect the rounding at k
            for i in 0..n {
                let hint_i = hint_vec[i];
                if hint_i == f64::MAX { continue; }
                dx_on_round[[0, i]] -= (mat[[i,k]] + mat[[k,i]])
                                    *(hint_i)*(hint_k);
                dx_on_round[[1, i]] -= (mat[[i,k]] + mat[[k,i]])
                                    *(hint_i)*(hint_k);
                dx_on_round[[1, i]] += mat[[i,k]] + mat[[k,i]];
            }
        }
        // Cast hint_vec to BinaryVector
        vecf_from_vecb(&hint_vec)
    }
    
    /// Do greedy rounding with "hint" for all starting entries
    fn greedy_from_hint(qubo: &QuboInstance, hint: &f64) -> BinaryVector {
        let hint_vec = Vector::from_vec(vec![*hint; qubo.size()]);
        Self::greedy_from_vec(qubo, &hint_vec)
    }
}

#[cfg(test)]
mod tests {
    use crate::qubo::*;
    use crate::start_heuristics::*;

    fn compute_sum_cross(mat: &Matrix, x: &BinaryVector, k: usize) -> f64 {
        let mut y = Vector::zeros(x.len());
        for i in 0..x.len() {
            y[i] = x[i] as u8 as f64;
        }
        compute_sum_cross_float(mat, &y, k)
    }

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
        let matrix = Matrix::from_shape_vec((6, 6),
            vec![-5.0,  2.0, -10.0, -9.0,  3.0, -3.0,
                  0.0,  3.0,   8.0,  8.0,  6.0,  0.0,
                  0.0,  0.0,   0.0, -9.0,  7.0,  6.0,
                  0.0,  0.0,   0.0,  5.0,  3.0,  8.0,
                  0.0,  0.0,   0.0,  0.0, -5.0,  0.0,
                  0.0,  0.0,   0.0,  0.0,  0.0,  0.0,]).unwrap();
        let qubo = QuboInstance::new(matrix, 0.0);
        let heur = StartHeuristic::GreedyFromHint(0.5);
        let solution = heur.get_solution(&qubo);
        let x = BinaryVector::from_vec(
            vec![true, false, true, true, false, false]);
        assert_eq!(solution, x);
    }
}
