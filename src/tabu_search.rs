use std::mem::take;
use std::time::Instant;
use ndarray::Array2;
/// QUBO tabu search functions

use crate::qubo::*;

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

// TODO: replace u32 with usize

struct ModelParameters {
    tenure_ratio: f64,
    diversification_base_factor: f64,
    diversification_scaling_factor: f64,
    improvement_threshold: u32,
    blocking_move_number: u32,
    activation_function: ActivationFunction,
}

enum ActivationFunction {
    // constant diversification intensity used
    Constant,
    // linearly descending intensity
    Linear,
}

struct SearchParameters {
    time_limit_seconds : f64,
    improvement_threshold: u32,
    diversification_length: u32,
    diversification_activation_function: ActivationFunction,
    tabu_tenure: u32,
}

struct QuboEvaluator {
    matrix: Matrix,
    size: u32,
    active: BinaryVector
}



impl QuboEvaluator {

    // objective delta when flipping a given index
    fn get_objective_delta_on_flip(&self, flip_index: u32) -> f64 {
        todo!()
    }

    // current objective
    fn get_objective(&self) -> f64 {

        todo!()
    }

    fn flip(&self, flip_index: u32) {

        todo!()
    }

    fn set_solution(&mut self, solution: BinaryVector) {
        todo!()
    }
}

enum PhaseType {
    Search,
    Diversification
}

struct TabuSearchState {
    search_parameters: SearchParameters,
    search_start_time : Instant, // TODO: maybe move this to a seperate SearchStatistics struct
    phase_type : PhaseType,
    phase_it: u32,
    last_improved: u32,
    qubo: QuboEvaluator,
    tabu_list: IntegerVector,
    best_objective: f64,
    frequency_matrix: IntegerMatrix,
    diversification_intensity: f64,
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

    fn new(qubo_instance: QuboInstance, model_parameters: ModelParameters, start_solution: BinaryVector) {

    }

    // Main function that performs a Tabu Search Iteration
    fn get_next(&mut self) {
        if self.check_phase_transition() {
            self.perform_phase_transition()
        }
        let swap_index = self.get_next_move();
        self.perform_move(swap_index);

        if self.check_termination() {
            self.finalize_search()
        }
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
                todo!()
                // TODO: check if phase was successful
                // TODO: if successful -> reset diversification intensity, reset additional tabu moves to empty vec
                // TODO: if unsuccessful -> scale diversification intensity, add previous first steps of diversification to tabu list
                // TODO      -> needs caching of first tabu moves probably in tabu search state
                // TODO: restore best solution together with its tabu_list
                // TODO: reset phase_it to 0 and update tabu_list list (substract phase_id from all)
                // TODO: mark all additional tabu moves tabu
            },
            PhaseType::Diversification => {
                //TODO: reset phase_it to 0 and update tabu_list (as prev)
                //TODO: simply change PhaseType
                todo!()
            }
        }
    }

    // Select next variable swap, if a eligble swap exists (non-tabu or tabu with aspiration)
    fn get_next_move(&self) -> Option<(u32, f64)> {
        let mut best_mv: Option<u32> = None;
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
            let is_best_global = self.qubo.get_objective() + orig_obj_delta < self.best_objective;

            if let (true, _, true) = (is_tabu, is_best_local, is_best_global) {

                // aspiration reached
                best_mv.take().replace(flip_index);
                best_obj_delta = orig_obj_delta + additional_penalty;
                if self.phase_type == PhaseType::Diversification {
                    break
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
    fn perform_move(&mut self, flip_index:u32) {
        self.qubo.flip(flip_index);
        self.tabu_list[flip_index] = self.phase_it + self.search_parameters.tabu_tenure;

        todo!()
        // update frequency measure -> get activated/dactivated entries -> add 1 if entry in Q is non-zero
    }


    fn get_diversification_penalty(&self, flip_index: u32) -> f64 {
        assert_eq!(self.phase_type, PhaseType::Diversification);
        let mut frequency_penalty_sum = 0;
        for i in 0..self.qubo.size {
            if self.qubo.active[i] {
                if self.qubo.matrix[(flip_index, i)] != 0.0 {
                    frequency_penalty_sum += self.frequency_matrix[(flip_index, i)]
                };
                if self.qubo.matrix[(i, flip_index)] != 0.0 {
                    frequency_penalty_sum += self.frequency_matrix[(i, flip_index)]
                };
            }
        }
        let scale_factor = match self.search_parameters.diversification_activation_function {
            ActivationFunction::Constant => {self.diversification_intensity}
            ActivationFunction::Linear => {
                let l = self.search_parameters.diversification_length as f64;
                ((l -self.phase_it as f64) / l) * self.diversification_intensity

            }
        };

        scale_factor * frequency_penalty_sum
    }
}

