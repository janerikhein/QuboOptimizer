/// QUBO start heuristic enum

use crate::qubo::*;
use ndarray_stats::QuantileExt;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use rand_pcg::Pcg32;
use rand::prelude::{SliceRandom, StdRng};

const AT_DN: usize = 0;
const AT_UP: usize = 1;

/// Computes the "sum cross" of x^T@Q@x for index k efficiently: O(n)
fn compute_sum_cross(mat: &Matrix, x: &Vector, k: usize) -> f64 {
    assert!(k < mat.nrows());
    let xk = x[k];
    let mut sum_cross = 0.0;
    for i in 0..k {
        let xi = x[i];
        assert!(i < k);
        sum_cross += mat[[i, k]]*xi*xk;
    }
    for i in k+1..mat.nrows() {
        let xi = x[i];
        assert!(k < i);
        sum_cross += mat[[k, i]]*xi*xk;
    }
    sum_cross + mat[[k, k]]*xk*xk
}

/// Cast Vector (f64) to BinaryVector (bool)
fn vecb_from_vecf(x: &Vector) -> BinaryVector {
    let n = x.len();
    let mut y = BinaryVector::from_vec(vec![false; n]);
    for i in 0..n {
        y[i] = x[i].round() != 0.0;
    }
    y
}

pub enum StartHeuristic {
    Random(u64),
    GreedyFromHint(f64),
    GreedyFromVec(Vector),
    RandomFromVec(Vector, usize, u64),
    GreedyInSteps(),
}
impl StartHeuristic {
    pub fn get_solution(&self, qubo: &QuboInstance) -> BinaryVector {
        match self {
            StartHeuristic::Random(seed) => {
                StartHeuristic::rand(qubo, seed)
            },
            StartHeuristic::GreedyFromHint(hint) => {
                StartHeuristic::greedy_from_hint(qubo, *hint)
            },
            StartHeuristic::GreedyFromVec(hint_vec) => {
                StartHeuristic::greedy_from_vec(qubo, hint_vec)
            },
            StartHeuristic::RandomFromVec(hint_vec, n_tries, seed) => {
                StartHeuristic::random_order_rounding(
                    qubo, hint_vec, *n_tries, *seed
                )
            },
            StartHeuristic::GreedyInSteps() => {
                StartHeuristic::greedy_in_steps(qubo)
            },
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
        vecb_from_vecf(&solution)
    }
    
    /// Do greedy rounding with "hint" for all starting entries
    fn greedy_from_hint(qubo: &QuboInstance, hint: f64) -> BinaryVector {
        let hints = Vector::from_vec(vec![hint; qubo.size()]);
        vecb_from_vecf(&Self::greedy(qubo, &hints, 0.0, 1.0))
    }

    /// Do greedy rounding with hint vector
    fn greedy_from_vec(qubo: &QuboInstance, hints: &Vector) -> BinaryVector {
        vecb_from_vecf(&Self::greedy(qubo, hints, 0.0, 1.0))
    }

    /// Do greedy rounding five times with ever-evolving floors/ceilings
    fn greedy_in_steps(qubo: &QuboInstance) -> BinaryVector {
        let hints = &Vector::from_vec(vec![0.5; qubo.size()]);
        let hints = &Self::greedy(qubo, hints, 0.4, 0.6);
        let hints = &Self::greedy(qubo, hints, 0.3, 0.7);
        let hints = &Self::greedy(qubo, hints, 0.2, 0.8);
        let hints = &Self::greedy(qubo, hints, 0.1, 0.9);
        let hints = &Self::greedy(qubo, hints, 0.0, 1.0);
        vecb_from_vecf(hints)
    }

    /// Find best rounding toward floor or ceil for each hint entry greedily
    fn greedy(
        qubo:  &QuboInstance,
        hints: &Vector,
        floor: f64,
        ceil:  f64
    ) -> Vector {
        assert!(0.0 <= floor && floor < ceil && ceil <= 1.0);
        // Make mutable copy
        //let mut hints = hints.clone();
        let mut solution = hints.clone();
        let n = qubo.size();
        let mat = qubo.get_matrix();
        // Changes on round of entry:
        let mut dx_on_round =
            Matrix::from_shape_vec((2, n), vec![0.0; 2*n]).unwrap();
        // Compute initial dx
        for i in 0..n {
            let tmp = solution[i];
            let sum_cross_of_old_solution = compute_sum_cross(mat, &solution, i);
            // Round down
            solution[i] = floor;
            dx_on_round[[AT_DN, i]]
                = compute_sum_cross(mat, &solution, i) - sum_cross_of_old_solution;
            // Round up
            solution[i] = ceil;
            dx_on_round[[AT_UP, i]]
                = compute_sum_cross(mat, &solution, i) - sum_cross_of_old_solution;
            // Undo rounding
            solution[i] = tmp;
        }
        for _ in 0..n {
            // Find next best k for up/down rounding (for biggest downward dx)
            let (row, k) = dx_on_round.argmin().unwrap();
            let round = match row {
                AT_DN => floor,
                _     => ceil,
            };
            // Update dx at k to MAX so it is not found as minimum again
            dx_on_round[[AT_DN, k]] = f64::MAX;
            dx_on_round[[AT_UP, k]] = f64::MAX;
            // Update dx at unvisited columns to respect the rounding at k
            for i in 0..n {
                if solution[i] == f64::MAX { continue; }
                let matsum = mat[[i, k]] + mat[[k, i]];
                dx_on_round[[AT_DN, i]] += matsum*(floor-solution[i])*(round-solution[k]);
                dx_on_round[[AT_UP, i]] += matsum*(ceil-solution[i])*(round-solution[k]);
            }
            // Actually round at k
            solution[k] = round;
        }
        solution
    }

    fn random_order_rounding (
        qubo:  &QuboInstance,
        hints: &Vector,
        number_of_tries : usize,
        seed: u64,
    ) -> BinaryVector {
        let mat = qubo.get_matrix();
        let mut base_contribution = Vector::zeros(qubo.size());
        let mut rng = StdRng::seed_from_u64(seed);
        // intialize contributions aka "sum crosses"
        for i in 0..qubo.size() {
            base_contribution[i] = compute_sum_cross(mat, hints, i);
        }
        let base_objective = base_contribution.sum();
        let mut best_solution = None;
        let mut best_objective = f64::MAX;
        for _ in 0..number_of_tries {
            let mut contribution = base_contribution.clone();
            let mut solution = hints.clone();
            let mut solution_objective = base_objective;
            let mut is_rounded = BinaryVector::from_elem(qubo.size(), false);
            let mut rounding_order: Vec<usize> = (0..qubo.size()).collect();
            rounding_order.shuffle(&mut rng);
            for index in rounding_order {
                is_rounded[index] = true;
                let delta_up = (1.0 - solution[index]) * contribution[index];
                let delta_down = -solution[index] * contribution[index];
                for k in 0..qubo.size() {
                    if is_rounded[k] { continue }
                    let matrix_entry = if index < k { mat[[index, k]] } else { mat[[k, index]] };
                    if delta_up <= 0.0 {
                        assert!(delta_down >= 0.0);
                        contribution[k] += (1.0 - solution[index]) * matrix_entry;
                        solution_objective += delta_up;
                    } else {
                        assert!(delta_up >= 0.0);
                        contribution[k] -= solution[index] * matrix_entry;
                        solution_objective += delta_up;
                    }
                }
                solution[index] = if delta_up <= 0.0 {1.0} else {0.0};
            }
            if solution_objective < best_objective {
                best_objective = solution_objective;
                best_solution = Some(solution);
            }
        }
        vecb_from_vecf(&best_solution.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::qubo::*;
    use crate::start_heuristics::*;

    fn compute_sum_cross_bin(mat: &Matrix, x: &BinaryVector, k: usize) -> f64 {
        let mut y = Vector::zeros(x.len());
        for i in 0..x.len() {
            y[i] = x[i] as u8 as f64;
        }
        compute_sum_cross(mat, &y, k)
    }

    #[test]
    fn test_sum_cross() {
        let x = BinaryVector::from_vec(vec![true, false, true]);
        let matrix = Matrix::from_shape_vec((3,3),
            vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0,]).unwrap();
        let sum_cross_values = vec![5.0, 0.0, 10.0];
        let mut obj_val = 0.0;
        for i in 0..matrix.nrows() {
            let sum_cross = compute_sum_cross_bin(&matrix, &x, i);
            obj_val += sum_cross;
            assert_eq!(sum_cross, sum_cross_values[i]);
        }
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
