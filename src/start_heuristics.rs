/// QUBO start heuristic enum

use crate::qubo::*;
use ndarray_stats::QuantileExt;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use rand_pcg::Pcg32;

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
    Random(u64),
    GreedyRounding(Float),
}
impl StartHeuristic {
    pub fn get_solution(&self, qubo: &QuboInstance) -> BinaryVector {
        match self {
            StartHeuristic::Random(seed)
                => { StartHeuristic::rand(qubo, seed) },
            StartHeuristic::GreedyRounding(hint)
                => { StartHeuristic::round_greedy(qubo, hint) },
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
        vecf_to_vecb(solution)
    }

    /// Find best rounding at some index until all indices are rounded
    fn round_greedy(qubo: &QuboInstance, hint: &Float) -> BinaryVector {
        let n = qubo.size();
        let mat = qubo.get_matrix();
        let mut hint_vec = Vector::from_vec(vec![*hint; n]);
        // Changes on round of entry:
        let mut dx_on_round =
            Matrix::from_shape_vec((2, n), vec![0.0; 2*n]).unwrap();
        // Compute initial dx
        for i in 0..n {
            // Round down
            hint_vec[i] = 0.0;
            dx_on_round[[0, i]] = compute_sum_cross_float(mat, &hint_vec, i);
            // Round up
            hint_vec[i] = 1.0;
            dx_on_round[[1, i]] = compute_sum_cross_float(mat, &hint_vec, i);
            // Undo rounding
            hint_vec[i] = *hint;
        }
        let mut unvisited: Vec<usize> = (0..n).collect();
        for _ in 0..n {
            // Find next best k for up/down rounding (for biggest downward dx)
            let (r, k) = dx_on_round.argmin().unwrap();
            // Round
            hint_vec[k] = r as Float;
            // Update dx at k to MAX so it is not found as minimum again
            dx_on_round[[0, k]] = Float::MAX;
            dx_on_round[[1, k]] = Float::MAX;
            unvisited[k] = usize::MAX;
            // Update dx at unvisited columns to respect the rounding at k
            for i in &unvisited {
                let i = *i;
                if i == usize::MAX { continue; }
                dx_on_round[[0, i]] -= (mat[[i,k]] + mat[[k,i]])*(*hint)*(*hint);
                dx_on_round[[1, i]] -= (mat[[i,k]] + mat[[k,i]])*(*hint)*(*hint);
                dx_on_round[[1, i]] += mat[[i,k]] + mat[[k,i]];
            }
        }
        // Cast hint_vec to BinaryVector
        vecf_to_vecb(hint_vec)
    }
}

#[cfg(test)]
mod tests {
    use crate::qubo::*;
    use crate::start_heuristics::*;

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
        let heur = StartHeuristic::GreedyRounding(0.5);
        let solution = heur.get_solution(&qubo);
        let x = BinaryVector::from_vec(
            vec![true, false, true, true, false, false]);
        assert_eq!(solution, x);
    }
}
