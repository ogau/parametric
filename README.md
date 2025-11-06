# Parametric

[![Crates.io](https://img.shields.io/crates/v/parametric.svg)](https://crates.io/crates/parametric)
[![Docs.rs](https://docs.rs/parametric/badge.svg)](https://docs.rs/parametric)

The `parametric` crate provides a bridge between complex, hierarchical data structures (like neural networks or simulation models) and optimization algorithms that operate on a flat vector of parameters.

```rust
// instead of raw slices and positional access...
fn evaluate(x: &[f64]) -> f64 {
    let &[a, b, c] = x else { panic!() };
    a - b*c
}

// ...define a named, structured type.
#[derive(Parametric)]
struct Formula<P> { a: P, b: P, c: P }

fn evaluate(x: &Formula<f64>) -> f64 {
    x.a - x.b * x.c
}
```

## The Problem

Many optimization and machine learning algorithms (like evolutionary strategies, gradient descent, etc.) are designed to work with a simple, flat `Vec<f64>` of parameters. However, the models we want to optimize often have a more complex, nested structure.

This creates a painful "impedance mismatch":

1.  **Manual Flattening:** You have to write tedious and error-prone boilerplate code to flatten your structured model parameters into a vector for the optimizer.
2.  **Manual Injection:** Inside the optimization loop, you must write the reverse logic to "inject" the flat vector of parameters back into your model's structure to evaluate its performance.
3.  **Brittleness:** Every time you change your model's structure (e.g., add a layer to a neural network), you have to meticulously update both the flattening and injection code.

## The Solution

`parametric` solves this by using a derive macro to automate the mapping between your structured types and a flat representation. By defining your model with a generic parameter (e.g., `struct Model<P>`), the macro automatically generates the necessary logic for this conversion.

The core workflow is:

1.  **Define a generic struct:** Create your model structure (e.g., `MLP<P>`) using a generic type `P` for the parameters. Add `#[derive(Parametric)]`.
2.  **Create a specification:** Instantiate your model with a type that describes parameter properties, like search ranges (e.g., `MLP<Range>`). This defines the parameter space.
3.  **Extract and Map:** Use `parametric::extract_map_defaults` to convert your specification (`MLP<Range>`) into two things:
    *   A runnable model instance with concrete value types (`MLP<f64>`).
    *   A flat `Vec` of the parameter specifications (`Vec<Range>`) that can be passed directly to an optimizer.
4.  **Inject:** Inside your objective function, use `parametric::inject_from_slice` to efficiently update the model instance with the flat parameter slice provided by the optimizer.

This approach eliminates boilerplate, reduces errors, and cleanly decouples the model's definition from its parameterization.

## Usage Example: Training a Neural Network

Here's a minimal example of defining a Multi-Layer Perceptron (MLP), specifying its parameter search space, and training it with a differential evolution algorithm.

See `examples/mlp.rs` for the complete code.

```rust
use parametric::Parametric;

// 1. DEFINE GENERIC, PARAMETRIC STRUCTS
// The same generic struct is used for two purposes:
// 1. As `MLP<Range>` to define the parameter search space (the specification).
// 2. As `MLP<f64>` to create a runnable model instance.

#[derive(Parametric)]
struct Layer<const I: usize, const O: usize, P> {
    weights: [[P; I]; O],
    biases: [P; O],
}

#[derive(Parametric)]
struct MLP<P> {
    hidden_layer: Layer<2, 3, P>,
    output_layer: Layer<3, 1, P>,
}

// Business logic is implemented on the concrete type.
impl MLP<f64> {
    fn forward(&self, x: [f64; 2]) -> f64 {
        let hidden_activations = self.hidden_layer.forward(&x).map(f64::tanh);
        let [output] = self.output_layer.forward(&hidden_activations);
        output
    }
}

// Implementation detail for Layer<f64>
impl<const I: usize, const O: usize> Layer<I, O, f64> {
    fn forward(&self, inputs: &[f64; I]) -> [f64; O] {
        let mut outputs = [0.0; O];
        for i in 0..O {
            let dot_product: f64 = self.weights[i].iter().zip(inputs).map(|(w, x)| w * x).sum();
            outputs[i] = dot_product + self.biases[i];
        }
        outputs
    }
}

// Define a custom type to represent a parameter's search range.
#[derive(Clone, Copy)]
struct Range(f64, f64);
// Make it compatible with the `parametric` crate.
parametric::impl_parametric_arg!(Range);

fn main() {
    // 2. CREATE A MODEL SPECIFICATION
    // Instantiate the model using `Range` to define the search space.
    let model_spec = MLP {
        hidden_layer: Layer {
            weights: [[Range(-10.0, 10.0); 2]; 3],
            biases: [Range(-1.0, 1.0); 3],
        },
        output_layer: Layer {
            weights: [[Range(-10.0, 10.0); 3]; 1],
            biases: [Range(-1.0, 1.0); 1],
        },
    };

    // 3. EXTRACT a runnable instance and a FLAT vector of bounds.
    // `model` is an MLP<f64> initialized with default values (0.0).
    // `bounds` is a Vec<Range> that the optimizer can use.
    let (mut model, bounds) = parametric::extract_map_defaults(model_spec);

    // XOR training data
    let x_train = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
    let y_train = [0., 1., 1., 0.];

    // The objective function for the optimizer.
    // It takes a flat slice of parameters.
    let objective_fn = |params: &[f64]| {
        // 4. INJECT the flat slice back into the structured model.
        parametric::inject_from_slice(&mut model, params).unwrap();

        let mut error = 0.0;
        for (xi, yi) in std::iter::zip(x_train, y_train) {
            let y_pred = model.forward(xi);
            error += (y_pred - yi).powi(2);
        }
        error
    };

    // Use any optimizer that works with a flat parameter vector and bounds.
    let (optimal_params, _final_error) = differential_evolution(
        objective_fn,
        bounds, // The flat bounds vector from `extract_map_defaults`
        /* optimizer settings... */
    );

    // The final, trained model is ready to use.
    parametric::inject_from_slice(&mut model, &optimal_params).unwrap();
    println!("Prediction for [1., 0.]: {}", model.forward([1.0, 0.0]));
}

// Dummy optimizer function for demonstration.
// See `examples/mlp.rs` for the complete example
fn differential_evolution(
    _objective_fn: impl Fn(&[f64]) -> f64,
    _bounds: Vec<Range>,
    /* ... */
) -> (Vec<f64>, f64) {
    // In a real scenario, this would be a proper optimization algorithm.
    let num_params = _bounds.len();
    (vec![0.5; num_params], 0.1)
}
```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
