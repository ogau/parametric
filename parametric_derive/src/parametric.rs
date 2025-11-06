use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, GenericParam, Generics, Ident, parse_quote};

use crate::MaybeSkipField;

pub(crate) fn implementation(
    ast: &DeriveInput,
    fields: &[MaybeSkipField],
    param: &Ident,
) -> TokenStream {
    let path = quote! { ::parametric:: };
    let struct_name = &ast.ident;

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let mut new_where_clause = (where_clause.cloned()).unwrap_or_else(|| parse_quote! { where });

    (new_where_clause.predicates).push(parse_quote! { #param: #path Parametric });

    let assoc_mapped_ty_generics = make_mapped_type_generics(&ast.generics, param);

    let try_map_initializers = fields.iter().map(|field| {
        let field_name = &field.name;
        if field.skip {
            quote! { #field_name: slf.#field_name }
        } else {
            quote! { #field_name: #path Parametric::try_map(slf.#field_name, &mut f)? }
        }
    });

    let try_visit_ref_calls = fields.iter().filter(|f| !f.skip).map(|field| {
        let field_name = &field.name;
        quote! { #path Parametric::try_visit(&slf.#field_name, &mut f)? }
    });

    let try_visit_mut_calls = fields.iter().filter(|f| !f.skip).map(|field| {
        let field_name = &field.name;
        quote! { #path Parametric::try_visit_mut(&mut slf.#field_name, &mut f)? }
    });

    quote! {
        impl #impl_generics #path Parametric for #struct_name #ty_generics #new_where_clause {
            type Arg = #param::Arg;
            type Mapped<_U> = #struct_name #assoc_mapped_ty_generics;

            fn try_map<_U, _F, _E>(slf: Self, mut f: _F) -> Result<Self::Mapped<_U>, _E>
            where
                _F: FnMut(Self::Arg) -> Result<_U, _E>,
            {
                Ok(Self::Mapped {
                    #(#try_map_initializers),*
                })
            }

            fn try_visit<_F, _E>(slf: &Self, mut f: _F) -> Result<(), _E>
            where
                _F: FnMut(&Self::Arg) -> Result<(), _E>,
            {
                #(#try_visit_ref_calls;)*
                Ok(())
            }

            fn try_visit_mut<_F, _E>(slf: &mut Self, mut f: _F) -> Result<(), _E>
            where
                _F: FnMut(&mut Self::Arg) -> Result<(), _E>,
            {
                #(#try_visit_mut_calls;)*
                Ok(())
            }
        }
    }
}

fn make_mapped_type_generics(original_generics: &Generics, param: &Ident) -> TokenStream {
    let output_params = (original_generics.params.iter()).map(|p| match p {
        GenericParam::Type(ty_param) => {
            let ty_param_ident = &ty_param.ident;

            if ty_param_ident == param {
                quote! { #ty_param_ident::Mapped<_U> }
            } else {
                quote! { #ty_param_ident }
            }
        }
        GenericParam::Lifetime(lt_def) => {
            let lifetime = &lt_def.lifetime;
            quote! { #lifetime }
        }
        GenericParam::Const(const_def) => {
            let ident = &const_def.ident;
            quote! { #ident }
        }
    });

    quote! { <#(#output_params),*> }
}
