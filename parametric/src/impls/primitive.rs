/// Allows this type to be the target of a `map` operation from any source type,
/// e.g. `Parametric::map(|any: T| -> Self { /* ... */ })`
///
/// This implementation serves as a workaround until the specialization
/// feature is stabilized in Rust.
///
#[macro_export]
macro_rules! impl_arg {
    ( $( $t:ty ),* ) => {
        $( $crate::impl_arg!(@internal $t |); )*
    };

    ( generic@ $( $t:ty ),* ) => {
        $( $crate::impl_arg!(@internal $t, P |); )*
    };

    ( generic@ $( $t:ty where $($bounds:tt)* ),* ) => {
        $( $crate::impl_arg!(@internal $t, P |, $($bounds)*); )*
    };

    (@internal $t:ty $(, $gen:ident)? | $(, $($bounds:tt)*)? ) => {
        impl$(<$gen>)? $crate::Parametric for $t $(where $($bounds)*)? {
            type Arg = Self;

            type Mapped<U> = U;

            #[inline]
            fn try_map<U, F, E>(slf: Self, mut f: F) -> ::core::result::Result<U, E>
            where
                F: FnMut(Self::Arg) -> ::core::result::Result<U, E>,
            {
                f(slf)
            }

            #[inline]
            fn map<U, F>(slf: Self, mut f: F) -> U
            where
                F: FnMut(Self::Arg) -> U,
            {
                f(slf)
            }

            #[inline]
            fn try_visit<F, E>(slf: &Self, mut f: F) -> ::core::result::Result<(), E>
            where
                F: FnMut(&Self) -> ::core::result::Result<(), E>,
            {
                f(slf)
            }

            #[inline]
            fn visit<F>(slf: &Self, mut f: F)
            where
                F: FnMut(&Self),
            {
                f(slf)
            }

            #[inline]
            fn try_visit_mut<F, E>(slf: &mut Self, mut f: F) -> ::core::result::Result<(), E>
            where
                F: FnMut(&mut Self::Arg) -> ::core::result::Result<(), E>,
            {
                f(slf)
            }

            #[inline]
            fn visit_mut<F>(slf: &mut Self, mut f: F)
            where
                F: FnMut(&mut Self::Arg),
            {
                f(slf)
            }
        }
    };
}

impl_arg!(
    f32,
    f64,
    bool,
    usize,
    isize,
    i8,
    i16,
    i32,
    i64,
    i128,
    u8,
    u16,
    u32,
    u64,
    u128,
    ()
);

impl_arg!(generic@ core::ops::Range<P>);
