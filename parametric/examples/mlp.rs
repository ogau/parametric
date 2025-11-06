use parametric::Parametric;

// A neural network layer.
// The generic `P` makes this struct polymorphic over its parameter types. This allows it
// to represent either a concrete, trainable model (e.g., where `P` is `f64`) or an
// abstract model specification (e.g., where `P` defines parameter ranges).
#[derive(Parametric)]
struct Layer<const INPUTS: usize, const NEURONS: usize, P> {
    weights: [[P; INPUTS]; NEURONS],
    biases: [P; NEURONS],
}

impl<const INPUTS: usize, const NEURONS: usize> Layer<INPUTS, NEURONS, f64> {
    // Executes a forward pass through the layer.
    fn forward(&self, inputs: &[f64; INPUTS]) -> [f64; NEURONS] {
        let mut outputs = [0.0; NEURONS];
        for i in 0..NEURONS {
            // Calculate the dot product of weights and inputs, then add the bias.
            let dot_product: f64 = self.weights[i].iter().zip(inputs).map(|(w, x)| w * x).sum();
            outputs[i] = dot_product + self.biases[i];
        }
        outputs
    }
}

// A Multi-Layer Perceptron (MLP).
// Like `Layer`, the generic `P` allows this struct to be used for both defining
// a model's parameter ranges and for instantiating a trainable model with concrete values.
#[derive(Parametric)]
struct MLP<P> {
    hidden_layer: Layer<2, 3, P>,
    output_layer: Layer<3, 1, P>,
}

impl MLP<f64> {
    // Executes a forward pass through the entire network.
    fn forward(&self, x: [f64; 2]) -> f64 {
        // The hidden layer uses the tanh activation function.
        let hidden_activations = self.hidden_layer.forward(&x).map(f64::tanh);
        let [output] = self.output_layer.forward(&hidden_activations);
        output
    }
}

// Defines a search range for a single parameter.
#[derive(Clone, Copy, Debug)]
struct Range(f64, f64);
parametric::impl_arg!(Range);

fn main() {
    // Define the search range for weights and biases.
    let r_w = Range(-10.0, 10.0);
    let r_b = Range(-1.0, 1.0);

    // Create a model specification using `Range` as the parameter type `P`.
    // This structure defines the shape and parameter bounds of the network.
    let model_spec = MLP {
        hidden_layer: Layer {
            weights: [[r_w; _]; _],
            biases: [r_b; _],
        },
        output_layer: Layer {
            weights: [[r_w; _]],
            biases: [r_b; _],
        },
    };

    // `extract_map_defaults` serves two purposes:
    // 1. It extracts the parameter ranges from `model_spec` into a flat `bounds` vector.
    // 2. It creates a `model` instance with the same structure, but with its fields
    //    (the `P` type) initialized to the default for `f64` (0.0).
    let (mut model, bounds) = parametric::extract_map_defaults(model_spec);

    // Training data for the XOR problem.
    let x = [
        [0., 0.], // Expected output: 0.
        [0., 1.], // Expected output: 1.
        [1., 0.], // Expected output: 1.
        [1., 1.], // Expected output: 0.
    ];
    let y = [0., 1., 1., 0.];

    // --- Differential Evolution Setup ---
    let pop_size: usize = 20;
    let f_weight: f64 = 0.5; // Mutation factor.
    let cr_prob: f64 = 0.8; // Crossover probability.
    let generations: usize = 350;

    // Run the differential evolution algorithm to find the optimal model parameters.
    let (optimal_x, final_error) = differential_evolution(
        // The objective function to minimize (mean squared error).
        |params: &[f64]| {
            // Inject the current parameters from the optimizer into the model instance.
            parametric::inject_from_slice(&mut model, params).unwrap();

            // Calculate the mean squared error for the current parameters.
            let mut error = 0.0;
            for (xi, y_true) in std::iter::zip(x, y) {
                let y_pred = model.forward(xi);
                error += (y_pred - y_true).powi(2);
            }
            error
        },
        bounds,
        pop_size,
        f_weight,
        cr_prob,
        generations,
    );

    println!("Optimal parameters: {:?}", optimal_x);
    println!("Final error: {:?}", final_error);

    // Verify the output of the trained MLP.
    parametric::inject_from_slice(&mut model, &optimal_x).unwrap();

    for (xi, y_true) in std::iter::zip(x, y) {
        let y_pred = model.forward(xi);
        println!("y_true: {y_true}, y_pred: {y_pred:.4}");
    }
}

fn differential_evolution(
    mut func: impl FnMut(&[f64]) -> f64,
    bounds: Vec<Range>,
    pop_size: usize,
    f_weight: f64,
    cr_prob: f64,
    generations: usize,
) -> (Vec<f64>, f64) {
    let dim = bounds.len();

    let mut rng = fastrand::Rng::default();

    let mut population: Vec<Vec<f64>> = (0..pop_size)
        .map(|_| {
            (0..dim)
                .map(|i| {
                    let Range(lb, ub) = bounds[i];
                    rng.f64() * (ub - lb) + lb
                })
                .collect()
        })
        .collect();

    let mut fitness: Vec<f64> = population.iter().map(|agent| func(agent)).collect();
    let mut next_population = population.clone();
    let mut next_fitness = fitness.clone();
    let mut trial_vec = vec![0.0; dim];

    for _ in 0..generations {
        for i in 0..pop_size {
            let mut r = [0; 3];
            let mut filled = 0;
            while filled < 3 {
                let candidate = rng.usize(0..pop_size);
                if candidate != i && !r[..filled].contains(&candidate) {
                    r[filled] = candidate;
                    filled += 1;
                }
            }
            let [r1, r2, r3] = r;

            let j_rand = rng.usize(0..dim);

            for j in 0..dim {
                if rng.f64() < cr_prob || j == j_rand {
                    let mutant =
                        population[r1][j] + f_weight * (population[r2][j] - population[r3][j]);
                    let Range(lb, ub) = bounds[j];
                    trial_vec[j] = mutant.clamp(lb, ub);
                } else {
                    trial_vec[j] = population[i][j];
                }
            }

            let trial_fitness = func(&trial_vec);
            if trial_fitness < fitness[i] {
                next_population[i].copy_from_slice(&trial_vec);
                next_fitness[i] = trial_fitness;
            } else {
                next_population[i].copy_from_slice(&population[i]);
                next_fitness[i] = fitness[i];
            }
        }
        std::mem::swap(&mut population, &mut next_population);
        std::mem::swap(&mut fitness, &mut next_fitness);
    }

    let (best_idx, _) = fitness
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    (population[best_idx].clone(), fitness[best_idx])
}
