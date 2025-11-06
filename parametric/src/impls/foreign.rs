#[cfg(feature = "ndarray")]
impl<T, Ix> crate::Parametric for ndarray::Array<T, Ix>
where
    T: crate::Parametric,
    Ix: ndarray::Dimension,
{
    type Arg = T::Arg;
    type Mapped<U> = ndarray::Array<T::Mapped<U>, Ix>;

    fn try_map<U, F, E>(slf: Self, mut f: F) -> Result<Self::Mapped<U>, E>
    where
        F: FnMut(Self::Arg) -> Result<U, E>,
    {
        let shape = slf.dim();
        let data = Result::from_iter(slf.into_iter().map(|item| T::try_map(item, &mut f)));
        data.map(|data| Self::Mapped::from_shape_vec(shape, data).unwrap())
    }

    fn try_visit<F, E>(slf: &Self, mut f: F) -> Result<(), E>
    where
        F: FnMut(&Self::Arg) -> Result<(), E>,
    {
        slf.iter().try_for_each(|item| T::try_visit(item, &mut f))
    }

    fn try_visit_mut<F, E>(slf: &mut Self, mut f: F) -> Result<(), E>
    where
        F: FnMut(&mut Self::Arg) -> Result<(), E>,
    {
        slf.iter_mut().try_for_each(|item| T::try_visit_mut(item, &mut f))
    }
}
