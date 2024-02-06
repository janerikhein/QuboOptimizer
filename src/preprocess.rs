/// QUBO preprocessing functions

use crate::qubo::*;
use ndarray::{Array1, Array2, Axis};

/// NOTE:
/// This function serves as a usage hint in the greater project context.
/// Preprocessing rules 1-5
pub fn shrink(instance: QuboInstance) -> QuboInstance {
    let (fixations_of_solution_variables, tentative_modified_matrix, baseline_from_fixation)
        = compute_fixations_of_solution_variables(&instance);
    assert_eq!(fixations_of_solution_variables.len(), instance.size());
    assert_eq!(tentative_modified_matrix.nrows(), instance.size());
    assert_eq!(tentative_modified_matrix.ncols(), instance.size());
    let mut unfixed_indices = Vec::new();
    for idx in 0..instance.size() {
        if fixations_of_solution_variables[idx] == FixState::Unfixed {
            unfixed_indices.push(idx)
        }
    }
    let new_rows = tentative_modified_matrix.select(Axis(0), &unfixed_indices);
    let new_matrix = new_rows.select(Axis(1), &unfixed_indices);
    QuboInstance::new(new_matrix, baseline_from_fixation)
}

#[derive(Debug)]
struct SumCrossOfSameSignEntriesWithoutDiagonal {
    positive: Array1<Option<f64>>,
    negative: Array1<Option<f64>>,
    //always check that these two arrays have same length
}

impl SumCrossOfSameSignEntriesWithoutDiagonal {
    fn compute(matrix: &Array2<f64>)
               -> SumCrossOfSameSignEntriesWithoutDiagonal {
        assert!(matrix.is_square());

        let mut positive: Array1<Option<f64>> = Array1::default(matrix.nrows());
        let mut negative: Array1<Option<f64>> = Array1::default(matrix.nrows());
        positive.fill(Some(0.));
        negative.fill(Some(0.));

        for row_and_column_idx in 0..matrix.nrows() {
            //remark that one could speed this up by exploiting the upper triangular form
            let mut sum_of_positive_row_entries: f64 = matrix.row(row_and_column_idx).iter().filter(|&&x| x > 0.0).sum();
            let mut sum_of_negative_row_entries: f64 = matrix.row(row_and_column_idx).iter().filter(|&&x| x < 0.0).sum();
            let mut sum_of_positive_column_entries: f64 = matrix.column(row_and_column_idx).iter().filter(|&&x| x > 0.0).sum();
            let mut sum_of_negative_column_entries: f64 = matrix.column(row_and_column_idx).iter().filter(|&&x| x < 0.0).sum();
            if matrix[[row_and_column_idx, row_and_column_idx]] > 0. {
                sum_of_positive_row_entries -= matrix[[row_and_column_idx, row_and_column_idx]];
                sum_of_positive_column_entries -= matrix[[row_and_column_idx, row_and_column_idx]];
            }
            else if matrix[[row_and_column_idx, row_and_column_idx]] < 0. {
                sum_of_negative_row_entries -= matrix[[row_and_column_idx, row_and_column_idx]];
                sum_of_negative_column_entries -= matrix[[row_and_column_idx, row_and_column_idx]];
            }
            positive[row_and_column_idx] = Some(positive[row_and_column_idx].unwrap() + sum_of_positive_row_entries + sum_of_positive_column_entries);
            negative[row_and_column_idx] = Some(negative[row_and_column_idx].unwrap() + sum_of_negative_row_entries + sum_of_negative_column_entries);
        }
        SumCrossOfSameSignEntriesWithoutDiagonal {positive, negative}
    }

    fn update(&mut self, matrix: &Array2<f64>, newly_fixed_indices: &Vec<usize>) {
        assert!(matrix.is_square());
        assert_eq!(matrix.nrows(), self.positive.len());
        assert_eq!(matrix.nrows(), self.negative.len());

        for fixed_idx in newly_fixed_indices {
            assert!(*fixed_idx < self.positive.len());
            self.positive[*fixed_idx] = None;
            assert!(*fixed_idx < self.negative.len());
            self.negative[*fixed_idx] = None;
        }

        // for each unfixed index, remove all positive/negative entries corresponding to fixed
        // indices from the sum of the unfixed index
        for unfixed_idx in 0..matrix.nrows() {
            //check that the variable unfixed_idx actually refers to an unfixed index
            if self.positive[unfixed_idx].is_none() {
                assert!(self.negative[unfixed_idx].is_none());
                continue;
            }
            assert!(self.negative[unfixed_idx].is_some());

            for fixed_idx in newly_fixed_indices {
                assert_ne!(unfixed_idx, *fixed_idx);
                //here we use that matrix is upper triangular //todo: write a test function for upper triangular
                let (row_idx, column_idx) = if unfixed_idx < *fixed_idx {
                    (unfixed_idx, *fixed_idx)
                } else {
                    (*fixed_idx, unfixed_idx)
                };
                if matrix[[row_idx, column_idx]] > 0. {
                    self.positive[unfixed_idx] = Option::from(self.positive[unfixed_idx].unwrap()
                        - matrix[[row_idx, column_idx]]);
                }
                else if matrix[[row_idx, column_idx]] < 0. {
                    self.negative[unfixed_idx] = Option::from(self.negative[unfixed_idx].unwrap()
                        - matrix[[row_idx, column_idx]]);
                }
            }
        }
    }
}

#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Clone)]
pub enum FixState {
    Fixed(bool),    //fixed to 1 or fixed to 0
    Unfixed,
    Irrelevant
}

pub fn compute_fixations_of_solution_variables(instance: &QuboInstance) -> (Array1<FixState>, Matrix, f64) {
    let mut tentative_modified_matrix = instance.get_matrix().clone();
    let mut baseline_from_fixation = 0.;
    let mut fix_states: Array1<FixState>  = Array1::from(vec![FixState::Unfixed; instance.size()]);
    let mut sum_cross_of_same_sign_entries_without_diagonal
        = SumCrossOfSameSignEntriesWithoutDiagonal::compute(&tentative_modified_matrix);
    let mut fixed_idx_in_last_iteration = true;
    while fixed_idx_in_last_iteration {
        fixed_idx_in_last_iteration = false;
        //one could also iterate through the indices in a random ordering
        for row_and_column_idx in 0..instance.size() {
            let mut newly_fixed_indices: Vec<usize> = Vec::new();
            if fix_states[row_and_column_idx] != FixState::Unfixed {
                continue;
            }
            //Rule 5 (all zeros)
            if (sum_cross_of_same_sign_entries_without_diagonal.negative[row_and_column_idx].unwrap() == 0.)
                && (sum_cross_of_same_sign_entries_without_diagonal.positive[row_and_column_idx].unwrap() == 0.)
                && (tentative_modified_matrix[[row_and_column_idx, row_and_column_idx]] == 0.) {
                fix_states[row_and_column_idx] = FixState::Irrelevant;
                newly_fixed_indices.push(row_and_column_idx);
            }
            //Rule 1 (best case is still positive)
            else if tentative_modified_matrix[[row_and_column_idx, row_and_column_idx]]
                + sum_cross_of_same_sign_entries_without_diagonal.negative[row_and_column_idx].unwrap()
                >= 0. {
                fix_states[row_and_column_idx] = FixState::Fixed(false);
                newly_fixed_indices.push(row_and_column_idx);
            }
            //Rule 2 (worst case is still negative)
            else if tentative_modified_matrix[[row_and_column_idx, row_and_column_idx]]
                + sum_cross_of_same_sign_entries_without_diagonal.positive[row_and_column_idx].unwrap()
                <= 0. {
                fix_states[row_and_column_idx] = FixState::Fixed(true);
                newly_fixed_indices.push(row_and_column_idx);

                baseline_from_fixation += tentative_modified_matrix[[row_and_column_idx, row_and_column_idx]];
            }
            //Rule 4 (worst case is still negative, for two indices)
            else {
                for other_row_and_column_idx in 0..row_and_column_idx {
                    if fix_states[other_row_and_column_idx] != FixState::Unfixed {
                        continue;
                    }
                    if (tentative_modified_matrix[[other_row_and_column_idx, row_and_column_idx]] < 0.)
                        && (tentative_modified_matrix[[other_row_and_column_idx, row_and_column_idx]]
                        + tentative_modified_matrix[[other_row_and_column_idx, other_row_and_column_idx]]
                        + tentative_modified_matrix[[row_and_column_idx, row_and_column_idx]]
                        + sum_cross_of_same_sign_entries_without_diagonal.positive[row_and_column_idx].unwrap()
                        + sum_cross_of_same_sign_entries_without_diagonal.positive[other_row_and_column_idx].unwrap()
                        <= 0.) {
                        fix_states[row_and_column_idx] = FixState::Fixed(true);
                        newly_fixed_indices.push(row_and_column_idx);
                        fix_states[other_row_and_column_idx] = FixState::Fixed(true);
                        newly_fixed_indices.push(other_row_and_column_idx);

                        baseline_from_fixation
                            += tentative_modified_matrix[[row_and_column_idx, row_and_column_idx]]
                            + tentative_modified_matrix[[other_row_and_column_idx, other_row_and_column_idx]]
                            + tentative_modified_matrix[[other_row_and_column_idx, row_and_column_idx]];
                    }
                }
            }
            sum_cross_of_same_sign_entries_without_diagonal.update(&tentative_modified_matrix, &newly_fixed_indices);
            //update matrix and objective value
            //todo: check that matrix is upper triangular
            //note that we must to update not only the diagonal entries since we need to access non-diagonal entries in rule 4
            for newly_fixed_idx in &newly_fixed_indices {
                if fix_states[*newly_fixed_idx] == FixState::Fixed(true) {
                    for other_column_idx in 0..*newly_fixed_idx {
                        tentative_modified_matrix[[other_column_idx, other_column_idx]] += tentative_modified_matrix[[other_column_idx, *newly_fixed_idx]]
                    }
                    for other_row_idx in newly_fixed_idx+1..instance.size() {
                        tentative_modified_matrix[[other_row_idx, other_row_idx]] += tentative_modified_matrix[[*newly_fixed_idx, other_row_idx]]
                    }
                }
            }
            if ! newly_fixed_indices.is_empty() {
                fixed_idx_in_last_iteration = true;
            }
        }
    }
    (fix_states, tentative_modified_matrix, baseline_from_fixation)
}

//todo: write a good (better) test function

#[test]
fn preprocessing_on_very_small_instances() {
    let preprocess_tests =
        [("instances/preprocessing_very_small/identity_matrix", [FixState::Fixed(false), FixState::Fixed(false), FixState::Fixed(false)]),
            ("instances/preprocessing_very_small/negative_identity_matrix", [FixState::Fixed(true), FixState::Fixed(true), FixState::Fixed(true)]),
            ("instances/preprocessing_very_small/test_rules_1_2_matrix", [FixState::Fixed(false), FixState::Fixed(true), FixState::Fixed(false)]),
            ("instances/preprocessing_very_small/test_rule_4_matrix", [FixState::Fixed(true), FixState::Fixed(true), FixState::Fixed(true)]),
            ("instances/preprocessing_very_small/other_small_test_matrix", [FixState::Fixed(true), FixState::Fixed(true), FixState::Fixed(false)]),
            ("instances/preprocessing_very_small/zero_matrix", [FixState::Irrelevant, FixState::Irrelevant, FixState::Irrelevant])
        ];
    for preprocess_test in preprocess_tests {
        println!("{}", preprocess_test.0);
        let input_matrix = input::read_from_file(preprocess_test.0);
        let qubo_instance = QuboInstance::new(input_matrix, 0.);
        let (fixations_of_solution_vars, _tentative_modified_matrix, _baseline_from_fixation)
            = compute_fixations_of_solution_variables(&qubo_instance);
        for i in 0..3 {
            assert_eq!(fixations_of_solution_vars[i], preprocess_test.1[i]);
        }
    }
}

//TODO: should be either removed or put into the qubo.rs file
mod input {
    use std::fs::File;
    use std::io::{BufRead, BufReader, Read};
    use ndarray::{Array2, ShapeBuilder};
    use crate::qubo::{Matrix};

    ///reads in a matrix in the MTX-format (see assignment 7)
    pub(crate) fn read_from_file(filename: &str) -> Matrix {
        let mut num_rows = 0;
        let mut num_columns = 0;
        let mut num_nonzero_entries_left = 0;
        let mut matrix: Matrix = Default::default();

        let input: Box<dyn Read> = Box::new(File::open(filename).expect("Can't open file"));
        let input = BufReader::new(input);
        let mut num_comment_lines: usize = 0;
        let mut warned = false;
        for (line_no, line) in input.lines().enumerate() {
            let line = line.expect("Read error");
            // map() works on the string slices created by split() and removes leading and trailing whitespaces (if exist)
            let fields: Vec<&str> = line.split(' ').map(|s| s.trim()).collect();
            assert!(!fields.is_empty());
            if fields[0].starts_with('%') {
                num_comment_lines += 1;
                continue;
            }
            assert!(fields.len() >= 3);
            if (fields.len() != 3) && !warned {
                warned = true;
                eprintln!("Warning! There is a line with more than 3 (namely {}) entries in the given file.", fields.len());
            }

            if line_no - num_comment_lines == 0 {
                num_rows = fields[0]
                    .parse::<usize>()
                    .expect("Number of Rows is not a positive number");
                num_columns = fields[1]
                    .parse::<usize>()
                    .expect("Number of Columns is not a positive number");
                // question: is there a better/shorter way to code this?
                //nodes.resize(num_nodes, Node{ neighbors : Vec::new() });
                matrix = Array2::<f64>::zeros((num_rows, num_columns).f());
                num_nonzero_entries_left = fields[2]
                    .parse::<usize>()
                    .expect("Number of non-zero entries is not a positive number");
                //assert!(num_nonzero_entries_left < input.lines().count());
            } else {
                assert_ne!(num_rows, 0);
                assert_ne!(num_columns, 0);
                assert!(num_nonzero_entries_left > 0);
                // question: how to parameterize the error messages with line_no+1
                let row_of_entry: usize = fields[0]
                    .parse::<usize>()
                    .expect("The number for the row of an entry is not a positive number")
                    - 1;
                assert!(row_of_entry < num_rows);
                let column_of_entry: usize = fields[1]
                    .parse::<usize>()
                    .expect("The number for the column of an entry is not a positive number")
                    - 1;
                assert!(column_of_entry < num_columns);
                let mut idx_to_access_data: usize =2;
                while fields[idx_to_access_data].is_empty() {
                    idx_to_access_data += 1;
                }
                let data_of_entry: f64 = fields[idx_to_access_data]
                    .parse::<f64>()
                    .expect("Problem reading the data of an entry");
                assert_ne!(data_of_entry, 0.0);
                matrix[[row_of_entry, column_of_entry]] = data_of_entry;
                num_nonzero_entries_left -= 1;
            }
        }
        assert_eq!(num_nonzero_entries_left, 0);
        matrix
    }

}
