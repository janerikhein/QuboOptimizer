use std::fmt::Debug;
use std::time::Instant;
use ndarray::Zip;

use ndarray_rand::rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// QUBO tabu search functions

use crate::qubo::*;

pub fn tabu_search_with_defaults(qubo: &QuboInstance, start_solution: &BinaryVector, log_level:usize) {
    let search_parameters = SearchParameters::default(qubo);
    tabu_search(qubo, start_solution, log_level, search_parameters);
}



/// Do tabu search
pub fn tabu_search(qubo: &QuboInstance, start_solution: &BinaryVector, log_level:usize, search_parameters: SearchParameters) -> BinaryVector {
    // define how verbose log is: 3 no log, 0 complete log
    //let log_level = 2;

    //let search_parameters = SearchParameters::default(qubo);
    let mut search_state = TabuSearchState::new(qubo, search_parameters, start_solution);
    let mut current_phase_type = "Search";
    let mut phase_nr = 1;

    if log_level <= 2 {
        println!("Time(ms) | Phase nr. | It | Phase It | Phase Type | objective | best obj. | move type")
    }
    while let Some(return_code) = search_state.get_next() {

        let phase_type = match search_state.phase_type {
            PhaseType::Search => "Search",
            PhaseType::Diversification => "Diversification"
        };
        if phase_type != current_phase_type {
            phase_nr += 1;
            current_phase_type = phase_type;
        }
        let base_str = format!("{} | {} | {} | {} | {} | {} | {}",
                               search_state.search_start_time.elapsed().as_millis(),
                               phase_nr,
                               search_state.it,
                               search_state.phase_it,
                               phase_type,
                               search_state.qubo.objective_of_curr_solution,
                               search_state.best_objective);

        match return_code {
            MoveReturnCode::GlobalImprovement => {
                println!("{base_str} | Global Improvement.")
            },
            MoveReturnCode::LocalImprovement => {
                if log_level <= 1 {
                    println!("{base_str} | Local Improvement.")
                }
            },
            MoveReturnCode::NonImprovement => {
                if log_level == 0 {
                    println!("{base_str} | Non-Improvement.")
                }
            },
        }
    }
    let duration_secs = search_state.search_start_time.elapsed().as_secs_f64();
    let best_obj = search_state.best_objective;
    print!("\n Optimization finished in {duration_secs}s. \n Objective of best solution found: {best_obj}");
    // TODO: write optimal solution to file
    search_state.best_solution
}


#[derive(Debug,Clone)]
pub enum ActivationFunction {
    // constant diversification intensity used
    Constant,
    // linearly descending intensity
    Linear,
}


#[derive(Debug)]
pub struct SearchParameters {
    tabu_tenure: usize,
    diversification_length: usize,
    diversification_base_factor: f64,
    diversification_scaling_factor: f64,
    diversification_activation_function: ActivationFunction,
    improvement_threshold: usize,
    blocking_move_number: usize,
    time_limit_seconds: usize,
    seed: usize,
}

const MIN_TABU_TENURE: usize = 5;

impl SearchParameters {
    pub fn new(
        // The QUBO instance
        qubo_instance: &QuboInstance,
        // ratio for setting tabu tenures relative to problem size
        //TODO: test values: (0.1, 0.25, 0.5)
        tenure_ratio: f64,
        // ratio for length of diversification phase relative to problem size
        //TODO: test values: (0.1, 0.5, 1.0)
        diversification_length_scale: f64,
        // base factor for diversification penalties
        //TODO: test values: (0.1, 0.25, 0.5)
        diversification_base_factor: f64,
        // scaling factor for increasing diversification intensity after unsuccessful phases
        //TODO: not too impactful, can fix to 1.5
        diversification_scaling_factor: f64,
        // activation function for diversification penalties
        //TODO: ignore this, fix to ActivationFunction::CONSTANT
        activation_function: ActivationFunction,
        // improvement threshold relative to problem size
        //TODO: test values: (1.0, 5.0, 10.0),
        improvement_threshold_scale: f64,
        // blocking move number relative to problem size
        //TODO: test values: (0.0, 0.05)
        blocking_move_number_scale: f64,
        // time limit for tabu search
        // TODO: set something reasonable here, small instances <500 should terminate in less than 30s
        time_limit_seconds: usize,
        // seed value
        // TODO: fix some seed value, i.e 42
        seed: usize,
    ) -> SearchParameters {
        let tabu_tenure = tenure_ratio * qubo_instance.size() as f64;
        let mut tabu_tenure = tabu_tenure.round() as usize;
        if tabu_tenure < MIN_TABU_TENURE {
            println!("Warning: Given tenure ratio leads to small tabu tenure of {tabu_tenure}.\
             Minimum tabu_tenure {MIN_TABU_TENURE} set instead");
            tabu_tenure = MIN_TABU_TENURE;
        }
        let improvement_threshold = (improvement_threshold_scale * qubo_instance.size() as f64).round() as usize;
        let blocking_move_number= (blocking_move_number_scale * qubo_instance.size() as f64).round() as usize;
        let diversification_length = (diversification_length_scale * qubo_instance.size() as f64).round() as usize;

        SearchParameters {
            tabu_tenure,
            diversification_length,
            diversification_base_factor,
            diversification_scaling_factor,
            diversification_activation_function: activation_function,
            improvement_threshold,
            blocking_move_number,
            time_limit_seconds,
            seed
        }
    }

    fn default(qubo_instance: &QuboInstance) -> SearchParameters {
        SearchParameters::new(
            qubo_instance,
            0.05,
            0.05,
            0.75,
            1.25,
            ActivationFunction::Constant,
            2.0,
            0.01,
            1800,
            42
        )
    }
}

struct QuboEvaluator {
    matrix: Matrix,
    frequency_matrix : IntegerMatrix,
    size: usize,
    curr_solution: BinaryVector,
    objective_deltas_on_flip: Vector,
    active_freq_sum: IntegerVector,
    partial_freq_sum: IntegerVector,
    inactive_freq_sum: IntegerVector,
    objective_of_curr_solution: f64,
    matrix_norm: f64,
    frequency_norm : f64,
}


impl QuboEvaluator {

    fn new(matrix: &Matrix, initial_solution: BinaryVector) -> QuboEvaluator {
        assert!(matrix.is_square());
        assert_eq!(matrix.nrows(), initial_solution.len());
        let matrix_norm = matrix.iter().map(|&val| val.abs()).sum();
        let size = matrix.nrows();
        let mut new_inst = QuboEvaluator{
            matrix: matrix.clone(),
            frequency_matrix: IntegerMatrix::zeros((size, size)),
            size,
            curr_solution: BinaryVector::from_elem(size, false),
            objective_deltas_on_flip: matrix.diag().into_owned(),
            active_freq_sum: IntegerVector::zeros(size),
            partial_freq_sum: IntegerVector::zeros(size),
            inactive_freq_sum: IntegerVector::zeros(size),
            objective_of_curr_solution: 0.0,
            matrix_norm,
            frequency_norm: 0.0,
        };
        new_inst.set_solution(&initial_solution);
        if new_inst.frequency_norm == 0.0 {
            new_inst.frequency_norm = 1.0
        }
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

    fn get_frequency_penalty_on_flip(&self, flip_index: usize) -> usize {
        if self.curr_solution[flip_index] {
            self.active_freq_sum[flip_index]
        } else {
            self.partial_freq_sum[flip_index]
        }
    }

    // current objective
    fn get_objective_of_curr_solution(&self) -> f64 {
        self.objective_of_curr_solution
    }

    fn register_activation_deactivation(&mut self, i: usize, j:usize) {
        self.frequency_matrix[[i,j]] += 1;
        self.frequency_norm += 1.0;
        match (self.curr_solution[i], self.curr_solution[j]) {
            (true, true) => {
                self.active_freq_sum[i] += 1;
                if i!=j {
                    self.active_freq_sum[j] += 1;
                }
            }
            (false, true) => {
                self.partial_freq_sum[i] += 1;
                self.partial_freq_sum[j] += 1;
            }
            (true, false) => {
                self.partial_freq_sum[i] += 1;
                self.partial_freq_sum[j] += 1;
            }
            (false, false) => {
                if i != j {
                    panic!("no activation/deactivation of non-diagonal element should be possible here.")
                }
                self.partial_freq_sum[i] += 1
            }
        }
    }

    fn flip(&mut self, flip_index: usize) {
        self.objective_of_curr_solution += self.objective_deltas_on_flip[flip_index];
        let is_activation = !self.curr_solution[flip_index];
        self.register_activation_deactivation(flip_index, flip_index);
        self.objective_deltas_on_flip[flip_index] = -self.objective_deltas_on_flip[flip_index];
        for idx in 0..self.size {
            let obj_sigma = if self.curr_solution[idx] == self.curr_solution[flip_index] { 1. } else { -1. };
            if idx < flip_index {
                if self.curr_solution[idx] {
                    self.register_activation_deactivation(idx, flip_index)
                }
                self.objective_deltas_on_flip[idx] += obj_sigma * self.get_matrix_entry_at(idx, flip_index);
            }
            if idx > flip_index {
                if self.curr_solution[idx] {
                    self.register_activation_deactivation(flip_index, idx);
                }
                self.objective_deltas_on_flip[idx] += obj_sigma * self.get_matrix_entry_at(flip_index, idx);
            }
        }
        // updating frequency flip values
        for idx in 0..self.size {
            if idx == flip_index {
                if is_activation {
                    self.active_freq_sum[flip_index] = self.partial_freq_sum[flip_index];
                    self.partial_freq_sum[flip_index] = self.inactive_freq_sum[flip_index];
                    self.inactive_freq_sum[flip_index] = 0;
                } else {
                    self.inactive_freq_sum[flip_index] = self.partial_freq_sum[flip_index];
                    self.partial_freq_sum[flip_index] = self.active_freq_sum[flip_index];
                    self.active_freq_sum[flip_index] = 0;
                }
                continue
            }
            let f_idx = if idx <= flip_index {(idx, flip_index)} else {(flip_index, idx)};
            let freq_matrix_entry = self.frequency_matrix[[f_idx.0, f_idx.1]];
            let freq_sigma = if is_activation {1} else {-1};
            if self.curr_solution[idx] {
                self.active_freq_sum[idx] = (self.active_freq_sum[idx] as isize + freq_sigma * freq_matrix_entry as isize) as usize;
                self.partial_freq_sum[idx] = (self.partial_freq_sum[idx] as isize - freq_sigma * freq_matrix_entry as isize) as usize;
            } else {
                self.partial_freq_sum[idx] = (self.partial_freq_sum[idx] as isize + freq_sigma * freq_matrix_entry as isize) as usize;
                self.inactive_freq_sum[idx] = (self.inactive_freq_sum[idx] as isize - freq_sigma * freq_matrix_entry as isize) as usize;
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


#[derive(Debug, PartialEq)]
enum PhaseType {
    Search,
    Diversification,
}


#[derive(Debug, PartialEq)]
enum MoveReturnCode {
    LocalImprovement,
    GlobalImprovement,
    NonImprovement,
}

struct TabuSearchState {
    search_parameters: SearchParameters,
    search_start_time: Instant,
    phase_type: PhaseType,
    it : usize,
    phase_it: usize,
    last_improved: usize,
    qubo: QuboEvaluator,
    tabu_list: IntegerVector,
    best_objective: f64,
    diversification_intensity: f64,
    explored_moves: BinaryVector,
    successive_unsuccessful_phases: usize,
    last_diversification_initial_steps: BinaryVector,
    last_improved_initial_steps: BinaryVector,
    best_solution: BinaryVector,
    last_improved_tabu_list: IntegerVector,
    rng: StdRng,
}

impl TabuSearchState {
    fn new(
        qubo_instance: &QuboInstance,
        search_parameters: SearchParameters,
        start_solution: &BinaryVector,
    ) -> TabuSearchState {
        let matrix = qubo_instance.get_matrix().to_owned();
        let evaluator = QuboEvaluator::new(&matrix, start_solution.to_owned());
        let start_objective = evaluator.objective_of_curr_solution;
        let seed = search_parameters.seed;
        let mut initial_state = TabuSearchState {
            search_parameters,
            search_start_time: Instant::now(),
            phase_type: PhaseType::Search,
            it: 0,
            phase_it: 0,
            last_improved: 0,
            qubo: evaluator,
            tabu_list: IntegerVector::zeros(qubo_instance.size()),
            best_objective: start_objective,
            diversification_intensity: Default::default(),
            explored_moves: BinaryVector::from_elem(qubo_instance.size(), false),
            successive_unsuccessful_phases: 0,
            last_diversification_initial_steps: BinaryVector::from_elem(qubo_instance.size(), false),
            last_improved_initial_steps: BinaryVector::from_elem(qubo_instance.size(), false),
            best_solution: start_solution.to_owned(),
            last_improved_tabu_list: IntegerVector::zeros(qubo_instance.size()),
            rng: StdRng::seed_from_u64(seed as u64),
        };

        initial_state.diversification_intensity = initial_state.get_base_diversification_intensity();
        initial_state

    }

    // Tabu Search Iteration
    fn get_next(&mut self) -> Option<MoveReturnCode> {
        if self.check_phase_transition() {
            self.perform_phase_transition()
        }
        let swap = self.get_next_move();
        let secs_elapsed = self.search_start_time.elapsed().as_secs_f64();
        // termination criteria
        if swap.is_none() || secs_elapsed >= self.search_parameters.time_limit_seconds as f64 {
            None
        }
        else {
            Some(self.perform_move(swap.unwrap().0))
        }
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
                    self.explored_moves.assign(&self.last_improved_initial_steps);
                }
                // Search phase was unsuccessful, i.e. no global improvement found
                if self.last_improved == 0 {
                    self.successive_unsuccessful_phases += 1;
                    self.diversification_intensity = self.search_parameters.diversification_scaling_factor.powi(
                        self.successive_unsuccessful_phases as i32
                    ) * self.get_base_diversification_intensity();
                    // add initial steps of last diversification phase to explored moves
                    Zip::from(&mut self.explored_moves).and(&self.last_diversification_initial_steps).for_each(
                        |a, &b| *a = *a || b
                    );
                    self.last_diversification_initial_steps.mapv_inplace(|_| false);
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
                self.last_improved = 0;
                self.phase_type = PhaseType::Diversification;
            },
            PhaseType::Diversification => {
                self.tabu_list.iter_mut().for_each(|x| *x = (*x as isize -self.phase_it as isize).max(0) as usize);
                self.phase_it = 0;
                self.last_improved = 0;
                self.phase_type = PhaseType::Search;
            }
        };
    }

    // Select next variable swap, if a eligible swap exists (non-tabu or tabu with aspiration)
    fn get_next_move(&mut self) -> Option<(usize, MoveReturnCode)> {
        //let reject_zero_move: bool = self.rng.gen();
        let mut best_mv: Option<usize> = None;
        let mut best_obj_delta = f64::MAX;
        let mut global_improvement_found = false;
        let mut local_improvement_found = false;
        let mut permutation: Vec<usize> = (0..self.qubo.size).collect();
        permutation.shuffle(&mut self.rng);
        for flip_index in permutation {

            let is_tabu = self.tabu_list[flip_index] > self.phase_it;
            let orig_obj_delta = self.qubo.get_objective_delta_on_flip(flip_index);
            //TODO: maybe remove this, this is temp
            //if reject_zero_move && orig_obj_delta == 0.0 {continue}
            let additional_penalty = match self.phase_type {
                PhaseType::Search => 0.0,
                PhaseType::Diversification => self.get_diversification_penalty(flip_index)
            };


            // is best move found so far wrt. objective of the phase
            let is_best_local = orig_obj_delta + additional_penalty < best_obj_delta;
            // leads to best solution found throughout the search wrt. original objective
            let is_best_global = self.qubo.get_objective_of_curr_solution() + orig_obj_delta < self.best_objective;

            if is_best_global {
                global_improvement_found = true;
                if is_best_local {
                    best_mv = Some(flip_index);
                    best_obj_delta = orig_obj_delta + additional_penalty;
                }
            } else if !is_tabu {
                if orig_obj_delta < -0.0 {
                    local_improvement_found = true;
                }
                if is_best_local {
                    best_mv = Some(flip_index);
                    best_obj_delta = orig_obj_delta + additional_penalty;
                }

            }
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
        let prev_objective = self.qubo.objective_of_curr_solution;
        self.qubo.flip(flip_index);
        self.tabu_list[flip_index] = self.phase_it + self.search_parameters.tabu_tenure;

        // Update frequency matrix
        for index in 0..self.qubo.size {
            if !self.qubo.curr_solution[index] {
                continue
            }
        }

        self.it += 1;
        self.phase_it += 1;

        assert!(self.best_objective <= prev_objective);

        match self.phase_type {
            PhaseType::Search => {
                if self.last_improved != 0 && self.phase_it - self.last_improved <= self.search_parameters.blocking_move_number {
                    self.last_improved_initial_steps[flip_index] = true
                }
            },
            PhaseType::Diversification => {
                if self.phase_it <= self.search_parameters.blocking_move_number {
                    self.last_diversification_initial_steps[flip_index] = true
                }
            }
        };

        let return_code = if self.qubo.objective_of_curr_solution < self.best_objective {
            // Global Improvement
            self.last_improved = self.phase_it;
            self.last_improved_tabu_list.assign(&self.tabu_list);
            self.best_solution.assign(&self.qubo.curr_solution);
            self.best_objective = self.qubo.objective_of_curr_solution;
            self.last_improved_initial_steps.mapv_inplace(|_| false);
            MoveReturnCode::GlobalImprovement
        } else if self.qubo.objective_of_curr_solution < prev_objective {
            // Local Improvement
            MoveReturnCode::LocalImprovement
        } else {
            // Non-Improvement
            MoveReturnCode::NonImprovement
        };

        return_code
    }


    fn get_diversification_penalty(&self, flip_index: usize) -> f64 {
        assert_eq!(self.phase_type, PhaseType::Diversification);
        let frequency_penalty_sum = self.qubo.get_frequency_penalty_on_flip(flip_index);
        let scale_factor = match self.search_parameters.diversification_activation_function {
            ActivationFunction::Constant => { self.diversification_intensity }
            ActivationFunction::Linear => {
                let l = self.search_parameters.diversification_length as f64;
                ((l - self.phase_it as f64) / l) * self.diversification_intensity
            }
        };

        scale_factor * frequency_penalty_sum as f64
    }
    fn get_base_diversification_intensity(&self) -> f64 {
        self.search_parameters.diversification_base_factor * self.qubo.matrix_norm / self.qubo.frequency_norm
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand_pcg::Pcg32;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    //tests flip values for a "current" solution that is the zero vector
    #[test]
    fn test_qubo_evaluator_rand() {
        let seed = 0;
        let size = 10;
        let (min, max) = (-10., 10.);
        let mut rng = Pcg32::seed_from_u64(seed);
        let mut matrix = Matrix::random_using(
            (size, size), Uniform::new(min, max), &mut rng);
        //set entries to zero to make the matrix upper triangular
        for row_idx in 0..size {
            for column_idx in 0..row_idx {
                matrix[[row_idx, column_idx]] = 0.;
            }
        }
        let zero_vector = BinaryVector::from_vec(vec![false; size]);
        let evaluator = QuboEvaluator::new(&matrix, zero_vector);
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
        let evaluator = QuboEvaluator::new(&matrix, solution);
        assert_eq!(evaluator.get_objective_of_curr_solution(), -8.);
        assert_eq!(evaluator.get_objective_delta_on_flip(0), 2.);
        assert_eq!(evaluator.get_objective_delta_on_flip(1), 7.);
        assert_eq!(evaluator.get_objective_delta_on_flip(2), 9.);
    }
}
