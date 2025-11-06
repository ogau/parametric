//! A bridge between complex data structures and flat parameter vectors.
//!
//! The `parametric` crate solves the "impedance mismatch" between complex, hierarchical
//! models (like neural networks or physics simulations) and optimization algorithms that
//! operate on a simple, flat `Vec<f64>` of parameters.
//!
//! It allows you to define your model's structure once, generically, and then use that
//! single definition to specify parameter search spaces, create runnable instances, and
//! seamlessly update parameters during optimization.
//!
//! # Getting Started: A Complete Example
//!
//! Let's walk through a typical workflow.
//!
//! ```
//! use parametric::{Parametric, extract_map_defaults, inject_from_slice};
//!
//! // Define a model structure with a generic parameter `P`.
//! // The `Parametric` trait can be derived for any struct whose fields
//! // are also `Parametric`.
//! #[derive(Parametric, Debug, PartialEq)]
//! struct Model<P> {
//!     weights: Vec<P>,
//!     bias: P,
//!
//!     // You can skip fields containing auxiliary data.
//!     // These fields are moved (not cloned) during `map` operation.
//!     #[parametric(skip)]
//!     metadata: String,
//! }
//!
//! // 1. Define the search space for the model using a parameter type like `Range`.
//! let search_space = Model {
//!     weights: vec![0.0..10.0, -5.0..5.0],
//!     bias: -1.0..1.0,
//!
//!     // The auxiliary data is included here.
//!     metadata: "Example Model v1".to_string(),
//! };
//!
//! // 2. Use `extract_map_defaults` to separate the definition into two parts:
//! //    - A flat `Vec` of parameter specifications (the ranges) for the optimizer.
//! //    - A runnable model instance with a concrete type (e.g., `f64`) initialized
//! //      to default values.
//! let (mut runnable_model, flat_space) = extract_map_defaults::<f64, _>(search_space);
//!
//! assert_eq!(flat_space, vec![0.0..10.0, -5.0..5.0, -1.0..1.0]);
//! assert_eq!(
//!     runnable_model,
//!     Model {
//!         weights: vec![0.0, 0.0],
//!         bias: 0.0,
//!         metadata: "Example Model v1".to_string()
//!     }
//! );
//!
//! // 3. Inside an optimization loop, an algorithm proposes a new flat vector of parameters.
//! let new_params = vec![8.5, -2.0, 0.5];
//!
//! // 4. Use `inject_from_slice` to update the structured model with the new parameters.
//! //    This is the "injection" step, turning the flat vector back into a structured model.
//! inject_from_slice(&mut runnable_model, &new_params).unwrap();
//!
//! assert_eq!(
//!     runnable_model,
//!     Model {
//!         weights: vec![8.5, -2.0],
//!         bias: 0.5,
//!         metadata: "Example Model v1".to_string()
//!     }
//! );
//!
//! // Now `runnable_model` can be used to evaluate the cost function.
//! ```
//!
//! # The Core Idea
//!
//! ### The Challenge
//!
//! Optimization algorithms typically require parameters as a flat `Vec<f64>`, but real-world
//! models have nested, meaningful structures. This forces developers to write tedious and
//! error-prone boilerplate for:
//!
//! 1.  **Flattening:** Manually converting a structured model into a flat vector.
//! 2.  **Injection:** Writing the reverse logic to update the model from a flat vector inside the optimization loop.
//! 3.  **Maintenance:** Meticulously updating both flattening and injection code every time the model's architecture changes.
//!
//! ### The `parametric` Solution
//!
//! This crate automates the mapping between your structured types and a flat representation.
//! The core is the [`Parametric`] trait, which provides a generic way to traverse a data
//! structure and apply a function to each of its parameters.
//!
//! By deriving `Parametric` on a generic struct like `Model<P>`, you can instantiate it for different purposes:
//!
//! -   `Model<Range<f64>>`: Defines the search space for each parameter.
//! -   `Model<f64>`: Creates a runnable instance with concrete values.
//! -   `Model<i32>`, `Model<bool>`, ...: Holds any other type you need.
//!                                       This allows you to use the same model definition for
//!                                       discrete parameters, configuration flags, or other metadata,
//!                                       not just floating-point values.
//!
//! Helper functions like [`extract_map_defaults`] and [`inject_from_slice`] use the
//! [`Parametric`] trait to provide the bridge to the optimizer, eliminating boilerplate,
//! reducing errors, and cleanly decoupling the model's architecture from its parameterization.

pub use parametric_derive::Parametric;

mod impls;

/// A trait for types whose parameters can be collectively transformed or visited.
///
/// The `Parametric` trait provides a unified interface for recursively applying a closure
/// to each parameter within a potentially nested data structure, preserving its shape.
///
/// It enables two fundamental operations:
/// - **Mapping**: Creating a new structure by transforming each parameter (e.g., `Model<f64>` -> `Model<i32>`).
/// - **Visiting**: Inspecting or modifying each parameter in-place (e.g., collecting all `&f64` values).
///
/// # Concept
///
/// The trait allows you to descend into a complex structure, apply a function to each
/// base parameter, and then rebuild an identical structure with the new parameter type.
///
/// ```text
/// Vec<Option<Box<f64>>>    // A nested structure with f64 parameters
///           ↓
///   closure(f64) -> i32    // A transformation function
///           ↓
/// Vec<Option<Box<i32>>>    // A new structure with the same shape
/// ```
///
/// # Deriving vs. Manual Implementation
///
/// The trait can be automatically derived for structs whose fields are all `Parametric`.
/// For custom or primitive types, you may need to implement it manually.
///
/// ## Implementation Guide
///
/// 1.  **For Parameter Types (Base Case)** (e.g., `f64`, `i32`):
///     -   `Arg` should be `Self`.
///     -   `Mapped<U>` should be `U`.
///     -   The methods should apply the closure directly to `self`.
///
/// 2.  **For Container Types (Recursive Step)** (e.g., `Vec<T>` where `T: Parametric`):
///     -   `Arg` should be `T::Arg` (the parameter type of the contained items).
///     -   `Mapped<U>` should preserve the container's structure (e.g., `Vec<T::Mapped<U>>`).
///     -   The methods should recursively call the operation on the contained items.
pub trait Parametric: Sized {
    /// The parameter type reached by recursive descent through the model structure.
    ///
    /// Represents the fundamental building block that can be transformed or inspected.
    ///
    /// # Implementation Examples
    /// - For `f64` (a parameter type): `Arg = Self`
    /// - For `Vec<T> where T: Parametric`: `Arg = T::Arg`
    type Arg;

    /// The model structure with parameters retyped to `U`.
    ///
    /// # Implementation Examples
    /// - For `f64`: `Mapped<U> = U`
    /// - For `Vec<T> where T: Parametric`: `Mapped<U> = Vec<T::Mapped<U>>`
    type Mapped<U>;

    /// Applies a fallible transformation to each parameter, creating a new structure.
    ///
    /// This method consumes the original structure and returns a new one with the
    /// same shape but potentially a different parameter type. The transformation
    /// short-circuits, returning the first `Err` encountered.
    ///
    /// # Examples
    ///
    /// ```
    /// # use parametric::Parametric;
    /// fn convert_to_i32(x: f64) -> Result<i32, &'static str> {
    ///     if x.fract() == 0.0 {
    ///         Ok(x as i32)
    ///     } else {
    ///         Err("Cannot convert non-integer float")
    ///     }
    /// }
    ///
    /// let model_ok = vec![1.0, 2.0, -3.0];
    /// assert_eq!(Parametric::try_map(model_ok, convert_to_i32), Ok(vec![1, 2, -3]));
    ///
    /// let model_err = vec![1.0, 2.5, -3.0];
    /// assert!(Parametric::try_map(model_err, convert_to_i32).is_err());
    /// ```
    ///
    /// # See Also
    /// - [`map`]: For infallible transformations.
    fn try_map<U, F, E>(slf: Self, f: F) -> Result<Self::Mapped<U>, E>
    where
        F: FnMut(Self::Arg) -> Result<U, E>;

    /// Recursively visits each parameter by immutable reference.
    ///
    /// Useful for read-only operations like validation, aggregation, or logging without
    /// consuming the structure. The operation short-circuits, returning the first `Err`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use parametric::Parametric;
    /// // Validate that all parameters are non-negative.
    /// let model = vec![(1, 2), (6, -1)];
    ///
    /// let result = Parametric::try_visit(&model, |&x| {
    ///     if x < 0 {
    ///         Err("Negative parameter found")
    ///     } else {
    ///         Ok(())
    ///     }
    /// });
    ///
    /// assert!(result.is_err());
    /// ```
    /// # See Also
    /// - [`visit`]: For infallible visiting.
    fn try_visit<F, E>(slf: &Self, f: F) -> Result<(), E>
    where
        F: FnMut(&Self::Arg) -> Result<(), E>;

    /// Recursively visits each parameter by mutable reference, allowing in-place modification.
    ///
    /// The operation short-circuits, returning the first `Err`. Any modifications made
    /// before the error will persist.
    ///
    /// # Examples
    ///
    /// ```
    /// # use parametric::Parametric;
    /// // Double each parameter, but fail if any are negative.
    /// let mut model = vec![(1, 2), (6, -1)];
    ///
    /// let result = Parametric::try_visit_mut(&mut model, |x| {
    ///     if *x < 0 {
    ///         Err("Negative parameter found")
    ///     } else {
    ///         *x *= 2;
    ///         Ok(())
    ///     }
    /// });
    ///
    /// assert!(result.is_err());
    /// // Note: The model was still partially modified before the error.
    /// assert_eq!(model, [(2, 4), (12, -1)]);
    /// ```
    /// # See Also
    /// - [`visit_mut`]: For infallible mutable visiting.
    fn try_visit_mut<F, E>(slf: &mut Self, f: F) -> Result<(), E>
    where
        F: FnMut(&mut Self::Arg) -> Result<(), E>;

    /// Applies an transformation to each parameter.
    ///
    /// Convenient for transformations that cannot fail.
    ///
    /// # Examples
    /// ```
    /// # use parametric::Parametric;
    /// let model = [(1, 2), (6, -1)];
    /// assert_eq!(
    ///     Parametric::map(model, |x| x > 0),
    ///     [(true, true), (true, false)]
    /// );
    /// ```
    /// # See Also
    /// - [`try_map`]
    fn map<U, F>(slf: Self, mut f: F) -> Self::Mapped<U>
    where
        F: FnMut(Self::Arg) -> U,
    {
        match Self::try_map(slf, |x| Ok::<_, std::convert::Infallible>(f(x))) {
            Ok(mapped) => mapped,
            Err(never) => match never {},
        }
    }

    /// Visits each parameter by immutable reference.
    ///
    /// Convenient for operations that cannot fail.
    ///
    /// # Examples
    /// ```
    /// # use parametric::Parametric;
    /// let model = [(1, 2), (6, -1)];
    /// let mut parameters = vec![];
    /// Parametric::visit(&model, |&x| parameters.push(x));
    /// assert_eq!(parameters, [1, 2, 6, -1]);
    /// ```
    /// # See Also
    /// - [`try_visit`]
    fn visit<F>(slf: &Self, mut f: F)
    where
        F: FnMut(&Self::Arg),
    {
        _ = Self::try_visit(slf, |x| {
            f(x);
            Ok::<(), std::convert::Infallible>(())
        })
    }

    /// Visits each parameter by mutable reference.
    ///
    /// Convenient for operations that cannot fail.
    ///
    /// # Examples
    /// ```
    /// # use parametric::Parametric;
    /// let mut model = [(1, 2), (6, -1)];
    /// Parametric::visit_mut(&mut model, |x: &mut i32| *x *= 3);
    /// assert_eq!(model, [(3, 6), (18, -3)]);
    /// ```
    /// # See Also
    /// - [`try_visit_mut`]
    fn visit_mut<F>(slf: &mut Self, mut f: F)
    where
        F: FnMut(&mut Self::Arg),
    {
        _ = Self::try_visit_mut(slf, |x| {
            f(x);
            Ok::<(), std::convert::Infallible>(())
        })
    }
}

/// A low-level utility that maps a parametric structure while extracting a second set of values.
///
/// This function iterates through the `model`, applying `fork_fn` to each parameter.
/// The `fork_fn` returns a tuple `(Left, Right)`.
/// - The `Left` value is collected into the `accum` vector.
/// - The `Right` value is used to construct the new, mapped model structure.
///
/// It is an efficient building block for [`extract_map_defaults`] because it pre-allocates
/// the accumulator `Vec` and fills it without intermediate allocations.
pub fn fork_map<M, Left, Right>(
    model: M,
    accum: &mut Vec<Left>,
    fork_fn: impl Fn(M::Arg) -> (Left, Right),
) -> M::Mapped<Right>
where
    M: Parametric,
{
    let mut param_count = 0;
    M::visit(&model, |_| param_count += 1);

    accum.clear();
    accum.reserve_exact(param_count);

    let mapped = M::map(model, |param| {
        let (left_value, right_value) = fork_fn(param);

        accum.push(left_value);

        right_value
    });

    assert!(accum.len() == param_count);

    mapped
}

/// Separates a parametric structure into a flat vector of its parameters and a new
/// structure containing default values.
///
/// This is a key setup function for optimization. It takes a "specification" model
/// (e.g., `Model<Range>`) and produces:
/// 1. A new model of the same shape, but with its parameters replaced by `X::default()`
///    (e.g., `Model<f64>`). This becomes the "runnable" model.
/// 2. A flat `Vec` containing the original parameters (e.g., `Vec<Range>`). This can be
///    passed to an optimizer to define search bounds.
pub fn extract_map_defaults<X, M>(model: M) -> (M::Mapped<X>, Vec<M::Arg>)
where
    X: Default,
    M: Parametric,
{
    let mut accum = vec![];
    let model_mapped = fork_map(model, &mut accum, |t| (t, X::default()));

    (model_mapped, accum)
}

/// Updates the parameters of a model in-place from a flat slice.
///
/// This function is typically called within an optimization loop to inject a new set of
/// parameter values from the optimizer back into the structured model for evaluation.
///
/// # Errors
///
/// Returns an error if the `params` slice contains fewer elements than the number of
/// parameters in the `model`.
pub fn inject_from_slice<M>(model: &mut M, mut params: &[M::Arg]) -> Result<(), &'static str>
where
    M: Parametric<Arg: Copy>,
{
    let slice = &mut params;
    M::try_visit_mut(model, |dst| {
        let mapped = slice.split_first().map(|(first, rest)| {
            *dst = *first;
            *slice = rest;
        });
        mapped.ok_or(())
    })
    .map_err(|()| "`inject_from_slice`: not enough parameters to fill the instance")
}

/// Updates the parameters of a model in-place from an iterator.
///
/// This function is an alternative to [`inject_from_slice`] that works with any iterator.
/// It is useful when the parameter type `M::Arg` is not `Copy`.
///
/// # Errors
///
/// Returns an error if the `params_it` iterator yields fewer items than the number of
/// parameters in the `model`.
pub fn inject_from_iter<M, I>(model: &mut M, mut params_it: I) -> Result<(), &'static str>
where
    M: Parametric,
    I: Iterator<Item = M::Arg>,
{
    M::try_visit_mut(model, |dst| {
        let mapped = params_it.next().map(|param| *dst = param);
        mapped.ok_or(())
    })
    .map_err(|()| "`inject_from_iter`: not enough parameters to fill the instance")
}

#[cfg(test)]
mod tests {
    use super::*;

    type ComplexType = Vec<[Vec<(Vec<Box<[(Box<f64>, [f64; 3])]>>, f64)>; 2]>;

    fn make_struct() -> ComplexType {
        let boxed_vec: Box<[(Box<f64>, [f64; 3])]> = Box::new([
            (Box::new(4.), [-1., -2., -4.]),
            (Box::new(5.), [9., 9., 6.]),
            (Box::new(6.), [0., 1., 0.]),
        ]);

        let inner_tuple = (vec![boxed_vec], 2.5);

        let array_of_tuples = [vec![inner_tuple.clone()], vec![inner_tuple.clone()]];

        vec![array_of_tuples]
    }

    #[test]
    fn test1() {
        let structure = make_struct();

        let (_, vec) = extract_map_defaults::<(), _>(structure);
        assert_eq!(
            vec,
            [
                4., -1., -2., -4., //
                5., 9., 9., 6., //
                6., 0., 1., 0.,  //
                2.5, //
                4., -1., -2., -4., //
                5., 9., 9., 6., //
                6., 0., 1., 0.,  //
                2.5, //
            ]
        );
    }

    #[test]
    fn test2() {
        let mut structure = make_struct();
        let num_params = {
            let mut count = 0;
            Parametric::visit(&structure, |_| count += 1);
            count
        };

        let arange: Vec<_> = (0..num_params).map(|i| i as f64).collect();

        inject_from_slice(&mut structure, &arange).unwrap();

        let (_, vec) = extract_map_defaults::<(), _>(structure);

        assert_eq!(vec, arange);
    }
}
