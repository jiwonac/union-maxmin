/// Algorithm used to solve MaxMin diversity problem. 
pub enum MaxMinAlgo {
    /// GRASP (Greedy Randomized Adaptive Search Procedure) method first greedily builds a randomized initial solution of size $k$. Then it performs local search until termination condition is reached. 
    Grasp(GraspConfig),
    /// GRASP variant which always drops the oldest point and adds another point that greedily maximizes the MaxMin score. Reduces the time complexity of local search to $O(n)$ at the cost of lower quality solutions. 
    DropAddGrasp(GraspConfig),
    /// Heuristic method based on the max-clique problem in graphs. 
    MaxClique,
}

/// Method used to construct a restricted candidates list (RCL) of good swaps. 
pub enum RCLConstructMethod {
    /// Chooses the top-$k$ candidates for specified $k$. 
    TopK(i32),
    /// Chooses all candidates within an $\alpha$-factor of the greedy candidate for specified $\alpha$. 
    ApproxMax(f64),
}

/// Method used to sample a swap from the restricted candidates list (RCL). 
pub enum RCLSelectMethod {
    /// Performs a uniformly random sample. 
    Uniform,
    /// Performs a weighted sample such that a candidate's probability of being chosen is proportional to its marginal gain. 
    Weighted,
}

/// Configures the behavior of GRASP algorithm or a variant. 
pub struct GraspConfig {
    /// Method used to construct a restricted candidates list (RCL) of good swaps. 
    pub construct_method: RCLConstructMethod,
    /// Method used to sample a swap from the restricted candidates list (RCL). 
    pub select_method: RCLSelectMethod,
    /// Number of iterations without improvement after which GRASP is terminated. 
    pub terminate_after: i32,
}

trait DistanceFn<T>: Fn(&T, &T) -> Result<f64, MathError>{}

enum MathError {
    VectorDimensionMisMatch(usize, usize),
    OutsideOfDomain {
        name: &str,
        value: f64,
        expected: (f64, f64),
    },
}

/// Builds a function which computes the Lp-norm between two points. 
pub fn get_lp_norm_fn(p: f64) -> Result<dyn DistanceFn<Vec<f64>>, MathError> {
    if p < 1.0 {
        Err(MathError::OutsideOfDomain {
            name: "p",
            value: p,
            expected: (1.0, f64::INFINITY),
        })
    }

    |foo: &Vec<f64>, bar: &Vec<f64>| -> Result<f64, MathError> {
        if len(foo) != bar(len) {
            Err(MathError::VectorDimensionMisMatch(len(foo), len(bar)))
        }

        let diff_vec: Vec<f64> = abs_subtract_vectors(foo, bar);

        match p {
            1.0 => diff_vec.sum(),
            f64::INFINITY => diff_vec.max(),
            _ => {
                diff_vec.into_iter().map(|x: f64| x.pow(p)).collect().sum()
                    .pow(1.0 / p)
            },
        }
    }
}

fn abs_subtract_vectors(foo: &Vec<f64>, bar: &Vec<f64>) -> Vec<f64> {
    foo.into_iter().zip(bar).map(|(a, b)| (a - b).abs).collect()
}

type PrevClosest = Vec<(Option<i32>, Option<f64>)>;

/// Internal data structure used to reduce runtime
struct MaxMinInternal {
    /// Current value of maximum minimum distance
    maxmin_dist: f64,
    /// Closest point in the current solution set for each point and distance
    closest_in_solution_index: Vec<CurrentClosest>,
}

struct CurrentClosest {
    self_index: i32,
    other_index: i32,
    dist: f64,
}

pub fn maxmin_solve<T>(
    algo: crate::MaxMinAlgo, 
    points: [T], 
    dist_fn: dyn crate::DistanceFn<T>, 
    k: i32) -> Vec<i32> {
    let mut closest_in_solution: PrevClosest = vec![(None, None); points.len()];
    let initial_solution: [i32] = greedy_initialization(&points, dist_fn, k, 
        &mut closest_in_solution);
    
    vec![]
}

fn greedy_initialization<T>(points: &[T], dist_fn: dyn crate::DistanceFn<T>, 
    k: i32, closest_in_solution: PrevClosest) -> [T] {
    // Solution set and candidates set store the indices of each set
    let mut solution: Vec<i32> = Vec::with_capacity(k);
    let mut candidates: Vec<i32> = (0..points.len()).collect();
    // Greedily initialize the initial solution with up to k elements
    for i in 0..k {
        if let Some(greedy_add_index) = greedy_add_choice(&candidates, 
            &closest_in_solution, points, dist_fn) {
            continue
        } else { // The index of greedy choice is None, this typically means
            solution // candidates set is empty, so just return solution
        }
    }
    solution
}

fn greedy_add_choice<T>(candidates: &Vec<i32>, closest_in_solution: PrevClosest,
    points: &[T], dist_fn: dyn crate::DistanceFn<T>) 
    -> Option<i32> {
    if let Some(max_element) = candidates.iter().max_by_key(|&x| 
        marginal_gain(x, points, closest_in_solution)) {
        Some(1)
    } else {
        None
    }
}

fn marginal_gain<T>(x: i32, points: &[T], closest_in_solution: PrevClosest) -> i32 {
    match closest_in_solution[x][0] {
        Some(closest_index) => 
        None => 
    }
}
