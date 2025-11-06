use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Ident, parse_macro_input};

mod parametric;

#[proc_macro_derive(Parametric, attributes(parametric, skip))]
pub fn general_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let param = first_gen_param(&input);

    let fields = get_fields(&input);

    let impl_ = parametric::implementation(&input, &fields, &param);

    TokenStream::from(quote! {
        #[automatically_derived]
        #impl_
    })
}

struct MaybeSkipField {
    name: Ident,
    skip: bool,
}

fn first_gen_param(input: &DeriveInput) -> Ident {
    let msg = "struct must be parameterized with at least one parameter";
    let param = input.generics.type_params().next().expect(msg);
    param.ident.clone()
}

fn get_fields(input: &DeriveInput) -> Vec<MaybeSkipField> {
    let Data::Struct(ref data) = input.data else {
        panic!("`Parametric` can only be derived for structs")
    };
    let syn::Fields::Named(fields) = &data.fields else {
        panic!("`Parametric` can only be derived for structs with named fields")
    };

    let match_parametric_skip = |attr: &syn::Attribute| {
        if !attr.path().is_ident("parametric") {
            return false;
        }

        match attr.parse_args::<Ident>() {
            Ok(ident) => ident == "skip",
            Err(_) => false,
        }
    };

    Vec::from_iter(fields.named.iter().map(|field| MaybeSkipField {
        name: field.ident.clone().unwrap(),
        skip: field.attrs.iter().any(match_parametric_skip),
    }))
}
