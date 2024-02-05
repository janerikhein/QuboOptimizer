use std::cmp::min;
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
pub fn tabu_search(qubo: &QuboInstance, start_solution: &BinaryVector) -> BinaryVector {
    // define how verbose log is: -1 no log, 0 only global improvements,
    let log_level = 0;

    let search_parameters = SearchParameters::default(qubo);
    let mut search_state = TabuSearchState::new(qubo, search_parameters, start_solution);

    if log_level
    while let Some(return_code) = search_state.get_next() {
        match return_code {
            MoveReturnCode::GlobalImprovement => {
                if log_level >= 2 {

                }
            },
            MoveReturnCode::LocalImprovement => {
                if log_level >= 1 {

                }
            },
            MoveReturnCode::NonImprovement => {
                if log_level >= 0 {
                    println!()
                }
            },
        }
    }
    search_state.best_solution
}



struct ModelParameters {
    tenure_ratio: f64,
    diversification_base_factor: f64,
    diversification_scaling_factor: f64,
    improvement_threshold: usize,
    blocking_move_number: usize,
    activation_function: ActivationFunction,
    time_limit_seconds: usize,
    diversification_length: usize
}

enum ActivationFunction {
    // constant diversification intensity used
    Constant,
    // linearly descending intensity
    Linear,
}

struct SearchParameters {
    tabu_tenure: usize,
    diversification_length: usize,
    diversification_base_factor: f64,
    diversification_scaling_factor: f64,
    diversification_activation_function: ActivationFunction,
    improvement_threshold: usize,
    blocking_move_number: usize,
    activation_function: ActivationFunction,
    time_limit_seconds: usize,
}

impl SearchParameters {
    fn new(
        qubo_instance: &QuboInstance,
        tenure_ratio: f64,
        diversification_base_factor: f64,
        diversification_scaling_factor: f64,
        activation_function: ActivationFunction,
        improvement_threshold: usize,
        blocking_move_number: usize,
        time_limit_seconds: usize,
    ) -> SearchParameters {

        let tabu_tenure = tenure_ratio * qubo_instance.size();
        if tabu_tenure < 5 {
            println!("Warning: Given tabu_ratio leads to small tabu_tenure of {tabu_tenure}.\
             Minimum tabu_tenure set instead");
            let tabu_tenure = 5;
        }
        SearchParameters {
            tabu_tenure,
            diversification_length: tabu_tenure,
            diversification_base_factor,
            diversification_scaling_factor,
            diversification_activation_function: ActivationFunction::Constant,
            improvement_threshold,
            blocking_move_number,
            activation_function,
            time_limit_seconds,
        }
    }

    fn default(qubo_instance: &QuboInstance) -> SearchParameters {
        SearchParameters::new(
            qubo_instance,
            0.01,
            0.1,
            1.5,
            ActivationFunction::Constant,
            200,
            5,
            100,
        )
    }
}

struct QuboEvaluator {
    matrix: Matrix,
    size: usize,
    curr_solution: BinaryVector,
    objective_deltas_on_flip: Vector,
    objective_of_curr_solution: f64,
    matrix_norm: f64,
}


impl QuboEvaluator {

    fn new(matrix: Matrix, initial_solution: BinaryVector) -> QuboEvaluator {
        assert!(matrix.is_square());
        assert_eq!(matrix.nrows(), initial_solution.len());
        let matrix_norm = matrix.iter().map(|&val| val.abs()).sum();
        let size = matrix.nrows();
        let mut new_inst = QuboEvaluator{
            matrix,
            size,
            curr_solution: BinaryVector::zeros(size),
            objective_deltas_on_flip: matrix.into_diag(),
            objective_of_curr_solution: 0.0,
            matrix_norm,

        };
        new_inst.set_solution(&initial_solution);
        new_inst
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

    fn set_solution(&mut self, solution: &BinaryVector) {
        assert_eq!(solution.len(), self.size);

        for i in 0..self.size {
            if solution[i] != self.curr_solution[i] {
                self.flip(i)
            }
        }
    }
}


enum PhaseType {
    Search,
    Diversification,
}

enum MoveReturnCode {
    LocalImprovement,
    GlobalImprovement,
    NonImprovement,
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
    successive_unsuccessful_phases: usize,
    last_diversification_initial_steps: BinaryVector,
    last_improved_initial_steps: BinaryVector,
    best_solution: BinaryVector,
    //TODO: substract phase_it before setting
    last_improved_tabu_list: IntegerVector,
    frequency_matrix_norm: f64,
}

impl TabuSearchState {
    fn new(
        qubo_instance: &QuboInstance,
        search_parameters: SearchParameters,
        start_solution: &BinaryVector,
    ) -> TabuSearchState {
        todo!()

    }

    // Main function that performs a Tabu Search Iteration TODO: return codes: local_improvement, global_improvement,
    fn get_next(&mut self) -> Option<MoveReturnCode> {
        if self.check_phase_transition() {
            self.perform_phase_transition()
        }
        let swap = self.get_next_move();
        let secs_elapsed = self.search_start_time.elapsed().as_secs_f64();
        // termination criteria
        if swap.is_none() || secs_elapsed >= self.search_parameters.time_limit_seconds as f64 {
            self.finalize_search();
            None
        }
        else {
            Some(self.perform_move(swap.unwrap()[0]))
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
                self.phase_it == self.search_parameters.diversification_length ||
                    (self.last_improved >= self.phase_it - 1 && self.last_improved != 0)
            }
        }
    }

    // Perform a phase transition (search phase <-> diversification phase)
    fn perform_phase_transition(&mut self) {
        match self.phase_type {
            PhaseType::Search => {
                // Search phase was successful, i.e. one or more global improvements found
                if self.last_improved != 0 {
                    self.successive_unsuccessful_phases = 0;
                    self.diversification_intensity = self.get_base_diversification_intensity();
                    // reset explored moves and add initial steps performed after last global improvement
                    self.explored_moves.assign(&self.last_improved_initial_steps)
                }
                // Search phase was unsuccessful, i.e. no global improvement found
                if self.last_improved == 0 {
                    self.successive_unsuccessful_phases += 1;
                    self.diversification_intensity = self.search_parameters.diversification_scaling_factor.pow(
                        self.successive_unsuccessful_phases
                    ) * self.get_base_diversification_intensity();
                    // add initial steps of last diversification phase to explored moves
                    Zip::from(&mut self.explored_moves).and(&self.last_diversification_initial_steps).for_each(
                        |a, &b| *a = *a || b
                    );
                }
                // restore best solution and tabu list
                self.qubo.set_solution(&self.best_solution);
                self.tabu_list.assign(&self.last_improved_tabu_list);
                self.phase_it = 0;

                // mark explored moves tabu
                for flip_index in 0..self.qubo.size {
                    if self.explored_moves[flip_index] {
                        self.tabu_list[flip_index] = self.phase_it + self.search_parameters.tabu_tenure;
                    }
                }
            },
            PhaseType::Diversification => {
                self.tabu_list.iter_mut().for_each(|x| *x = (*x-self.phase_it).max(0));
                self.phase_it = 0;
            }
        }
    }

    // Select next variable swap, if a eligble swap exists (non-tabu or tabu with aspiration)
    fn get_next_move(&self) -> Option<(usize, MoveReturnCode)> {
        let mut best_mv: Option<usize> = None;
        let mut best_obj_delta = f64::MAX;
        let mut global_improvement_found = false;
        let mut local_improvement_found = false;
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

            match (is_tabu, is_best_local, is_best_global) {
                (true, _, true) => { // aspiration reached
                    global_improvement_found = true;
                    best_mv.take().replace(flip_index);
                    best_obj_delta = orig_obj_delta + additional_penalty;
                    if self.phase_type == PhaseType::Diversification {
                        break;
                    }
                }
                (false, true, _) => { // best non-tabu move so far
                    if orig_obj_delta < -0.0 {
                        local_improvement_found = true;
                    }
                    best_mv.take().replace(flip_index);
                    best_obj_delta = orig_obj_delta + additional_penalty;
                }
                _ => {}
            };
        }

        match best_mv {
            Some(flip_index) => {
                if global_improvement_found {
                    Some((flip_index, MoveReturnCode::GlobalImprovement))
                }
                else if local_improvement_found  {
                    Some((flip_index, MoveReturnCode::LocalImprovement))
                }
                else {
                    Some((flip_index, MoveReturnCode::NonImprovement))
                }
            },
            _ => None
        }
    }

    // Perform a given variable swap and update memory structures accordingly
    fn perform_move(&mut self, flip_index: usize) -> MoveReturnCode {
        self.qubo.flip(flip_index);
        self.tabu_list[flip_index] = self.phase_it + self.search_parameters.tabu_tenure;

        // Update frequency matrix
        for index in 0..self.qubo.size {
            if !self.qubo.curr_solution[index] {
                continue
            }
            if index <= flip_index && self.qubo.matrix[[index,flip_index]] != 0.0 {
                self.frequency_matrix[[index, flip_index]] += 1
            }
            if index > flip_index && self.qubo.matrix[[index,flip_index]] != 0.0 {
                self.frequency_matrix[[flip_index, index]] += 1
            }
            self.frequency_matrix_norm += 1.0
        }
        todo!()
        // compute type of move and updte everthing accordingly

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
        self.search_parameters.diversification_base_factor * self.qubo.matrix_norm / self.frequency_matrix_norm
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

