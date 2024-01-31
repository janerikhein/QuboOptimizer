use std::mem::take;
use std::time::Instant;
use ndarray::{Array2, s, Zip};

use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::num_traits::Pow;
use ndarray_rand::rand_distr::Uniform;
use rand_pcg::Pcg32;

/// QUBO tabu search functions

use crate::qubo::*;

fn vecb_to_vecf(x: &BinaryVector) -> Vector {
    let n = x.len();
    let mut y = Vector::from_vec(vec![f64::MAX; n]);
    for i in 0..n {
        y[i] = if x[i] == true { 1. } else { 0. };
    }
    y
}

/// NOTE:
/// These functions only serve as usage hints in the greater project context, as
/// they could be used by the experiment functions.

/// Do (simple) tabu search
pub fn tabu_search(qubo: &QuboInstance, start_solution: &BinaryVector)
                   -> BinaryVector {
    todo!();
}

/// Do staged tabu search
pub fn staged_tabu_search(qubo: &QuboInstance, start_solution: &BinaryVector)
                          -> BinaryVector {
    todo!();
}


struct ModelParameters {
    tenure_ratio: f64,
    diversification_base_factor: f64,
    diversification_scaling_factor: f64,
    improvement_threshold: usize,
    blocking_move_number: usize,
    activation_function: ActivationFunction,
}

enum ActivationFunction {
    // constant diversification intensity used
    Constant,
    // linearly descending intensity
    Linear,
}

struct SearchParameters {
    time_limit_seconds: f64,
    improvement_threshold: usize,
    diversification_length: usize,
    diversification_activation_function: ActivationFunction,
    tabu_tenure: usize,
    diversification_scaling_factor: f64,
}

struct QuboEvaluator {
    matrix: Matrix,
    size: usize,
    curr_solution: BinaryVector,
    objective_deltas_on_flip: Vector,
    objective_of_curr_solution: f64
}


impl QuboEvaluator {

    fn new(matrix: Matrix, solution: BinaryVector) -> QuboEvaluator {
        assert!(matrix.is_square());
        assert_eq!(matrix.nrows(), solution.len());
        let size = matrix.nrows();
        let mut new = QuboEvaluator{matrix: matrix,
            size: size,
            curr_solution: Default::default(),
            objective_deltas_on_flip: Default::default(),
            objective_of_curr_solution: f64::MIN };
        //one could also integrate set_solution into this function?
        new.set_solution(solution);
        new
    }

    pub fn get_matrix_entry_at(&self, i: usize, j: usize) -> f64 {
        assert!(j >= i);
        self.matrix[[i,j]]
    }

    // objective delta when flipping a given index
    fn get_objective_delta_on_flip(&self, flip_index: usize) -> f64 {
        assert!(flip_index < self.size);
        self.objective_deltas_on_flip[flip_index]
    }

    // current objective
    fn get_objective_of_curr_solution(&self) -> f64 {
        self.objective_of_curr_solution
    }

    fn flip(&mut self, flip_index: usize) {
        self.objective_of_curr_solution += self.objective_deltas_on_flip[flip_index];
        self.objective_deltas_on_flip[flip_index] = -self.objective_deltas_on_flip[flip_index];
        for idx in 0..self.size {
            let sigma = if self.curr_solution[idx] == self.curr_solution[flip_index] { 1. } else { -1. };
            if idx < flip_index {
                self.objective_deltas_on_flip[idx] += sigma * self.get_matrix_entry_at(idx, flip_index);
            }
            if idx > flip_index {
                self.objective_deltas_on_flip[idx] += sigma * self.get_matrix_entry_at(flip_index, idx);
            }
        }
        self.curr_solution[flip_index] = !self.curr_solution[flip_index];
    }

    //sets curr_solution,  computes corresponding objective_deltas_on_flip and objective_of_curr_solution
    fn set_solution(&mut self, solution: BinaryVector) {
        assert_eq!(solution.len(), self.size);
        self.curr_solution = solution;

        let solution_f = vecb_to_vecf(&self.curr_solution);
        self.objective_of_curr_solution = solution_f.dot(&self.matrix.dot(&solution_f));

        self.objective_deltas_on_flip = Vector::from_vec(vec![f64::MIN; self.size]);
        for i in 0..self.size {
            let sigma = if self.curr_solution[i] == false { 1. } else { -1. };
            self.objective_deltas_on_flip[i] = sigma * self.get_matrix_entry_at(i, i);
            for j in 0..self.size {
                if j == i {continue}
                let x_j = (self.curr_solution[j] as usize) as f64;
                let matrix_entry = if j < i {self.get_matrix_entry_at(j, i)}
                                        else {self.get_matrix_entry_at(i, j)};
                self.objective_deltas_on_flip[i] += sigma * matrix_entry * x_j;
            }
        }
    }
}


enum PhaseType {
    Search,
    Diversification,
}

struct TabuSearchState {
    search_parameters: SearchParameters,
    search_start_time: Instant,
    // TODO: maybe move this to a seperate SearchStatistics struct
    phase_type: PhaseType,
    phase_it: usize,
    last_improved: usize,
    qubo: QuboEvaluator,
    tabu_list: IntegerVector,
    best_objective: f64,
    frequency_matrix: IntegerMatrix,
    diversification_intensity: f64,
    explored_moves: BinaryVector,
    successive_unsuccesful_phases: usize,
    last_diversification_initial_steps: BinaryVector,
}


impl SearchParameters {
    fn new(model_parameters: ModelParameters, qubo_instance: QuboInstance) -> SearchParameters {
        todo!()
        // Initialize search_paramters based on model_parameters and given instance, i.e. tabu_tenure <- tabu_ratio * model_size
    }
}

impl QuboEvaluator {
    fn new() {
        todo!()
    }

    fn set_solution(&mut self, solution: BinaryVector) {
        todo!()
    }
}

impl TabuSearchState {
    fn new(qubo_instance: QuboInstance, model_parameters: ModelParameters, start_solution: BinaryVector) {}

    // Main function that performs a Tabu Search Iteration TODO: return codes: local_improvement, global_improvement,
    fn get_next(&mut self) {
        if self.check_phase_transition() {
            self.perform_phase_transition()
        }
        let swap_index = self.get_next_move();
        self.perform_move(swap_index);
        // termination criteria
        if self.check_termination() {
            self.finalize_search()
        }
        self.perform_move(swap_index.unwrap()[0]);
    }

    // Check if search should be terminated
    fn check_termination(&self) -> bool {
        match self.phase_type {
            PhaseType::Search => {
                self.search_start_time.elapsed().as_secs_f64() >= self.search_parameters.time_limit_seconds
            },
            PhaseType::Diversification => {
                self.search_start_time.elapsed().as_secs_f64() >= self.search_parameters.time_limit_seconds ||
                    {
                        // No eligible move found as all moves are tabu
                        todo!()
                    }
            }
        }
    }

    // Stop search and retrieve best solution found
    fn finalize_search(&self) {

        todo!()
        // retrieve best solution and return
        // do final logging with overall statistics
        // TODO: probably add a different struct for storing statistics throughout the search, i.e. overall iterations
        // TODO: number of phases performed, number of successful/unsuccesful phases, objective value of solution
    }

    // Check if phase transition should be initiated (search phase <-> diversification phase)
    fn check_phase_transition(&self) -> bool {
        match self.phase_type {
            PhaseType::Search => {
                self.phase_it - self.last_improved == self.search_parameters.improvement_threshold
            },
            PhaseType::Diversification => {
                self.phase_it == self.search_parameters.diversification_length || todo!()
                // TODO: transition if new best solution found
            }
        }
    }

    // Perform a phase transition (search phase <-> diversification phase)
    fn perform_phase_transition(&mut self) {
        match self.phase_type {
            PhaseType::Search => {
                // Search phase was successful, i.e. one or more global improvements found
                if self.last_improved != 0 {
                    self.successive_unsuccesful_phases = 0;
                    self.diversification_intensity = self.get_base_diversification_intensity();
                    self.explored_moves.mapv_inplace(|_| false);
                }
                // Search phase was unsuccessful, i.e. no global improvement found
                if self.last_improved == 0 {
                    self.successive_unsuccesful_phases += 1;
                    self.diversification_intensity = self.search_parameters.diversification_scaling_factor.pow(
                        self.successive_unsuccesful_phases
                    ) * self.get_base_diversification_intensity();
                    Zip::from(&mut self.explored_moves).and(&self.last_diversification_initial_steps).for_each(
                        |a, &b| *a = *a || b
                    );
                }
                // if successful -> reset diversification intensity, reset additional tabu moves to empty vec
                // TODO: if unsuccessful -> scale diversification intensity, add previous first steps of diversification to tabu list
                // TODO      -> needs caching of first tabu moves probably in tabu search state
                // TODO: restore best solution together with its tabu_list
                // TODO: reset phase_it to 0 and update tabu_list list (substract phase_id from all)
                // TODO: mark all additional tabu moves tabu
            }
            PhaseType::Diversification => {
                //TODO: reset phase_it to 0 and update tabu_list (as prev)
                //TODO: simply change PhaseType
                todo!()
            }
        }
    }

    // Select next variable swap, if a eligble swap exists (non-tabu or tabu with aspiration)
    fn get_next_move(&self) -> Option<(usize, f64)> {
        let mut best_mv: Option<usize> = None;
        let mut best_obj_delta = f64::MAX;
        for flip_index in 0..self.qubo.size {
            let is_tabu = self.tabu_list[flip_index] >= self.phase_it;
            let orig_obj_delta = self.qubo.get_objective_delta_on_flip(flip_index);
            let additional_penalty = match self.phase_type {
                PhaseType::Search => 0.0,
                PhaseType::Diversification => self.get_diversification_penalty(flip_index)
            };
            // is best move found so far wrt. objective of the phase
            let is_best_local = orig_obj_delta + additional_penalty < best_obj_delta;
            // leads to best solution found throughout the search wrt. original objective
            let is_best_global = self.qubo.get_objective_of_curr_solution() + orig_obj_delta < self.best_objective;

            if let (true, _, true) = (is_tabu, is_best_local, is_best_global) {

                // aspiration reached
                best_mv.take().replace(flip_index);
                best_obj_delta = orig_obj_delta + additional_penalty;
                if self.phase_type == PhaseType::Diversification {
                    break;
                }
            } else if let (false, true, _) = (is_tabu, is_best_local, is_best_global) {

                // non-tabu and best move found so far
                best_mv.take().replace(flip_index);
                best_obj_delta = orig_obj_delta + additional_penalty;
            };
        }
        match best_mv {
            Some(flip_index) => Some((flip_index, best_obj_delta)),
            _ => None
        }
    }

    // Perform a given variable swap and update memory structures accordingly
    fn perform_move(&mut self, flip_index: usize) {
        self.qubo.flip(flip_index);
        self.tabu_list[flip_index] = self.phase_it + self.search_parameters.tabu_tenure;

        todo!()
        // update frequency measure -> get activated/dactivated entries -> add 1 if entry in Q is non-zero
    }


    fn get_diversification_penalty(&self, flip_index: usize) -> f64 {
        assert_eq!(self.phase_type, PhaseType::Diversification);
        let mut frequency_penalty_sum = 0;
        for i in 0..self.qubo.size {
            if self.qubo.curr_solution[i] {
                if self.qubo.matrix[(flip_index, i)] != 0.0 {
                    frequency_penalty_sum += self.frequency_matrix[(flip_index, i)]
                };
                if self.qubo.matrix[(i, flip_index)] != 0.0 {
                    frequency_penalty_sum += self.frequency_matrix[(i, flip_index)]
                };
            }
        }
        let scale_factor = match self.search_parameters.diversification_activation_function {
            ActivationFunction::Constant => { self.diversification_intensity }
            ActivationFunction::Linear => {
                let l = self.search_parameters.diversification_length as f64;
                ((l - self.phase_it as f64) / l) * self.diversification_intensity
            }
        };

        scale_factor * frequency_penalty_sum
    }
    fn get_base_diversification_intensity(&self) -> f64 {
        todo!()
    }
}


//tests flip values for a "current" solution that is the zero vector
#[test]
fn test_qubo_evaluator_rand() {
    let seed = 0;
    let size = 10;
    let (min, max) = (-10., 10.);
    let mut rng = Pcg32::seed_from_u64(seed);
    //todo: mir fällt hier auf, dass wir ja gar nicht so richtig ausnutzen, dass wir ne obere Dreiecksmatrix haben oder
    let mut matrix = Matrix::random_using(
        (size, size), Uniform::new(min, max), &mut rng);
    //set entries to zero to make the matrix upper triangular
    for row_idx in 0..size {
        for column_idx in 0..row_idx {
            matrix[[row_idx, column_idx]] = 0.;
        }
    }
    let zero_vector = BinaryVector::from_vec(vec![false; size]);
    let evaluator = QuboEvaluator::new(matrix.clone(), zero_vector);
    assert_eq!(evaluator.get_objective_of_curr_solution(), 0.);
    for i in 0..size {
        assert_eq!(evaluator.get_objective_delta_on_flip(i), matrix[[i, i]]);
    }
}

#[test]
fn test_qubo_evaluator_small_example() {
    let mut matrix = Matrix::from_shape_vec((3, 3),
                                            vec![1., -2.,  -3.,
                                                 0., 4., 5.,
                                                 0., 0., -6.]).unwrap();
    //set entries to zero to make the matrix upper triangular
    for row_idx in 0..3 {
        for column_idx in 0..row_idx {
            matrix[[row_idx, column_idx]] = 0.;
        }
    }
    let solution = BinaryVector::from_vec(vec![true, false, true]);
    let evaluator = QuboEvaluator::new(matrix.clone(), solution);
    assert_eq!(evaluator.get_objective_of_curr_solution(), -8.);
    assert_eq!(evaluator.get_objective_delta_on_flip(0), 2.);
    assert_eq!(evaluator.get_objective_delta_on_flip(1), 7.);
    assert_eq!(evaluator.get_objective_delta_on_flip(2), 9.);
}

