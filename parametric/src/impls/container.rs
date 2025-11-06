#![allow(unused_mut)]

use crate::Parametric;

macro_rules! impl_container {
    (($self:ident, $f:ident)
    $(
        +
        $(const $constnum:ident: usize,)*
        $src_ty:ty => $dst_ty:ty,
        $impl_try_map:expr,
        $impl_map:expr,
        $impl_try_ref:expr,
        $impl_ref:expr,
        $impl_try_mut:expr,
        $impl_mut:expr,
    )*) => {$(
        impl<T: Parametric, $(const $constnum: usize,)*> Parametric for $src_ty {
            type Arg = T::Arg;
            type Mapped<U> = $dst_ty;

            #[inline]
            fn try_map<U, F, E>($self: Self, mut $f: F) -> Result<Self::Mapped<U>, E>
            where
                F: FnMut(Self::Arg) -> Result<U, E>,
            {
                $impl_try_map
            }

            #[inline]
            fn try_visit<F, E>($self: &Self, mut $f: F) -> Result<(), E>
            where
                F: FnMut(&Self::Arg) -> Result<(), E>,
            {
                $impl_try_ref
            }

            #[inline]
            fn try_visit_mut<F, E>($self: &mut Self, mut $f: F) -> Result<(), E>
            where
                F: FnMut(&mut Self::Arg) -> Result<(), E>,
            {
                $impl_try_mut
            }

            #[inline]
            fn map<U, F>($self: Self, mut $f: F) -> Self::Mapped<U>
            where
                F: FnMut(Self::Arg) -> U,
            {
                $impl_map
            }

            #[inline]
            fn visit<F>($self: &Self, mut $f: F)
            where
                F: FnMut(&Self::Arg),
            {
                $impl_ref
            }

            #[inline]
            fn visit_mut<F>($self: &mut Self, mut $f: F)
            where
                F: FnMut(&mut Self::Arg),
            {
                $impl_mut
            }
        }
    )*};
}

impl_container!(
    (slf, f)

    +
    Option<T> => Option<T::Mapped<U>>,
    slf.map(|item| T::try_map(item, f)).transpose(),
    slf.map(|item| T::map(item, f)),

    slf.as_ref().map_or(Ok(()), |item| T::try_visit(item, f)),
    _ = slf.as_ref().map(|item| T::visit(item, f)),

    slf.as_mut().map_or(Ok(()), |item| T::try_visit_mut(item, f)),
    _ = slf.as_mut().map(|item| T::visit_mut(item, f)),

    +
    Vec<T> => Vec<T::Mapped<U>>,
    slf.into_iter().map(|item| T::try_map(item, &mut f)).collect(),
    slf.into_iter().map(|item| T::map(item, &mut f)).collect(),

    slf.iter().try_for_each(|item| T::try_visit(item, &mut f)),
    slf.iter().for_each(|item| T::visit(item, &mut f)),

    slf.iter_mut().try_for_each(|item| T::try_visit_mut(item, &mut f)),
    slf.iter_mut().for_each(|item| T::visit_mut(item, &mut f)),

    +
    Box<[T]> => Box<[T::Mapped<U>]>,
    slf.into_iter().map(|item| T::try_map(item, &mut f)).collect(),
    slf.into_iter().map(|item| T::map(item, &mut f)).collect(),

    slf.iter().try_for_each(|item| T::try_visit(item, &mut f)),
    slf.iter().for_each(|item| T::visit(item, &mut f)),

    slf.iter_mut().try_for_each(|item| T::try_visit_mut(item, &mut f)),
    slf.iter_mut().for_each(|item| T::visit_mut(item, &mut f)),

    +
    Box<T> => Box<T::Mapped<U>>,
    T::try_map(*slf, f).map(Box::new),
    Box::new(T::map(*slf, f)),

    T::try_visit(&**slf, f),
    T::visit(&**slf, f),

    T::try_visit_mut(&mut **slf, f),
    T::visit_mut(&mut **slf, f),

    +
    std::cell::RefCell<T> => std::cell::RefCell<T::Mapped<U>>,
    T::try_map(slf.into_inner(), f).map(std::cell::RefCell::new),
    std::cell::RefCell::new(T::map(slf.into_inner(), f)),

    T::try_visit(&*slf.borrow(), f),
    T::visit(&*slf.borrow(), f),

    T::try_visit_mut(slf.get_mut(), f),
    T::visit_mut(slf.get_mut(), f),

    +
    const N: usize,
    [T; N] => [T::Mapped<U>; N],
    {
        // 10 years strong!
        let mut iter = slf.into_iter();
        array_init::try_array_init(|_| T::try_map(iter.next().unwrap(), &mut f))
    },
    {
        let mut iter = slf.into_iter();
        std::array::from_fn(|_| T::map(iter.next().unwrap(), &mut f))
    },

    slf.iter().try_for_each(|item| T::try_visit(item, &mut f)),
    slf.iter().for_each(|item| T::visit(item, &mut f)),

    slf.iter_mut().try_for_each(|item| T::try_visit_mut(item, &mut f)),
    slf.iter_mut().for_each(|item| T::visit_mut(item, &mut f)),
);

macro_rules! tuple_impl {
    ($first:ident, $($rest:ident),*) => {
        #[allow(non_snake_case)]
        impl<$first, $($rest),*> Parametric for ($first, $($rest,)*)
        where
            $first: Parametric,
            $( $rest: Parametric<Arg = $first::Arg>, )*
        {
            type Arg = $first::Arg;
            type Mapped<U> = ($first::Mapped<U>, $( $rest::Mapped<U>, )*);

            #[inline]
            fn try_map<U, F, E>(slf: Self, mut f: F) -> Result<Self::Mapped<U>, E>
            where
                F: FnMut(Self::Arg) -> Result<U, E>,
            {
                let ($first, $( $rest ),*) = slf;
                Ok((
                    Parametric::try_map($first, &mut f)?,
                    $( Parametric::try_map($rest, &mut f)? ),*
                ))
            }

            #[inline]
            fn map<U, F>(slf: Self, mut f: F) -> Self::Mapped<U>
            where
                F: FnMut(Self::Arg) -> U,
            {
                let ($first, $( $rest ),*) = slf;
                (
                    Parametric::map($first, &mut f),
                    $( Parametric::map($rest, &mut f) ),*
                )
            }

            #[inline]
            fn try_visit<F, E>(slf: &Self, mut f: F) -> Result<(), E>
            where
                F: FnMut(&Self::Arg) -> Result<(), E>,
            {
                let ($first, $( $rest ),*) = slf;
                Parametric::try_visit($first, &mut f)?;
                $( Parametric::try_visit($rest, &mut f)?; )*
                Ok(())
            }

            #[inline]
            fn visit<F>(slf: &Self, mut f: F)
            where
                F: FnMut(&Self::Arg),
            {
                let ($first, $( $rest ),*) = slf;
                Parametric::visit($first, &mut f);
                $( Parametric::visit($rest, &mut f); )*
            }

            #[inline]
            fn try_visit_mut<F, E>(slf: &mut Self, mut f: F) -> Result<(), E>
            where
                F: FnMut(&mut Self::Arg) -> Result<(), E>,
            {
                let ($first, $( $rest ),*) = slf;
                Parametric::try_visit_mut($first, &mut f)?;
                $( Parametric::try_visit_mut($rest, &mut f)?; )*
                Ok(())
            }

            #[inline]
            fn visit_mut<F>(slf: &mut Self, mut f: F)
            where
                F: FnMut(&mut Self::Arg),
            {
                let ($first, $( $rest ),*) = slf;
                Parametric::visit_mut($first, &mut f);
                $( Parametric::visit_mut($rest, &mut f); )*
            }
        }
    };
}

// tuple_impl!(A,);
tuple_impl!(A, B);
tuple_impl!(A, B, C);
// tuple_impl!(A, B, C, D);
// tuple_impl!(A, B, C, D, E);
// tuple_impl!(A, B, C, D, E, G);
// tuple_impl!(A, B, C, D, E, G, H);

// Result<T, E>
impl<T: Parametric, E> Parametric for Result<T, E> {
    type Arg = T::Arg;
    type Mapped<U> = Result<T::Mapped<U>, E>;

    fn try_map<U, F, Err>(slf: Self, f: F) -> Result<Self::Mapped<U>, Err>
    where
        F: FnMut(Self::Arg) -> Result<U, Err>,
    {
        match slf {
            Ok(item) => T::try_map(item, f).map(Ok),
            Err(e) => Ok(Err(e)),
        }
    }

    fn try_visit<F, Err>(slf: &Self, f: F) -> Result<(), Err>
    where
        F: FnMut(&Self::Arg) -> Result<(), Err>,
    {
        slf.as_ref()
            .map(|item| T::try_visit(item, f))
            .unwrap_or(Ok(()))
    }

    fn try_visit_mut<F, Err>(slf: &mut Self, f: F) -> Result<(), Err>
    where
        F: FnMut(&mut Self::Arg) -> Result<(), Err>,
    {
        slf.as_mut()
            .map(|item| T::try_visit_mut(item, f))
            .unwrap_or(Ok(()))
    }

    fn map<U, F>(slf: Self, mut f: F) -> Self::Mapped<U>
    where
        F: FnMut(Self::Arg) -> U,
    {
        slf.map(|item| T::map(item, f))
    }

    fn visit<F>(slf: &Self, mut f: F)
    where
        F: FnMut(&Self::Arg),
    {
        _ = slf.as_ref().map(|item| T::visit(item, f))
    }

    fn visit_mut<F>(slf: &mut Self, mut f: F)
    where
        F: FnMut(&mut Self::Arg),
    {
        _ = slf.as_mut().map(|item| T::visit_mut(item, f))
    }
}
