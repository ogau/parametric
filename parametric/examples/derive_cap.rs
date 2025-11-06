use parametric::Parametric;

/// A simple demo model with mixed generics:
/// - The first non-lifetime and non-const generic (`P`) always is the "core" type.
/// - All operations are performed on this core type.
/// - Other generics are passed through as-is.
#[derive(Parametric, Debug)]
struct Model<'a, P, const N: usize, U: Sized>
where
    U: Into<String>,
{
    /// Just a single scalar value.
    scalar: P,

    /// A flat vector of values.
    vec: Vec<P>,

    /// A square 2D array (N x N).
    squared_array: [[P; N]; N],

    /// A nested tuple with some complexity.
    tuple: (P, (Box<P>, Option<P>)),

    /// A custom container that also derives `Parametric`.
    mat: Matrix<P>,

    /// Ignored during traversal; ownership transferred via move (not clone).
    #[parametric(skip)]
    another_generic: U,

    /// Also ignored, reference data only.
    #[parametric(skip)]
    aux_ref_data: &'a [bool; N],
}

/// Example container type with internal data.
#[derive(Parametric, Debug)]
struct Matrix<T> {
    /// The actual flat buffer of values.
    data: Box<[T]>,

    /// Shape information (skipped).
    #[parametric(skip)]
    shape: [usize; 2],
}

fn main() {
    const N: usize = 2;

    let model = Model {
        scalar: 10,
        vec: vec![601; N],
        squared_array: [[3; N]; N],
        tuple: (8, (Box::new(1), None)),
        mat: {
            let shape = [N, N];
            let data = vec![-1; shape[0] * shape[1]].into();
            Matrix { data, shape }
        },
        another_generic: "str",
        aux_ref_data: &[true, false],
    };

    // Visit each parameter of type `P` and print it
    Model::visit(&model, |x| print!("{x}, "));
    println!();

    // Transform all leaves into another type (`i32 -> f64`),
    // producing a new model with the same structure.
    let model_f64 = Model::map(model, |x| x as f64);

    // Extract all parameter values into `extracted`
    // while rebuilding the model with default values of another type `usize`
    let (model_usize, extracted) = parametric::extract_map_defaults::<usize, _>(model_f64);

    println!("Extracted: {:?}", extracted);
    println!("Zeroed model: {:#?}", model_usize);
}
