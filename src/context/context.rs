use serde::{Deserialize, Serialize};

use crate::symbol::{symbol_type::TypeHierarchy, transformation::Transformation};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Context {
    types: TypeHierarchy,
    transformations: Vec<Transformation>,
}

impl Context {
    pub fn empty() -> Self {
        Self {
            types: TypeHierarchy::new(),
            transformations: vec![],
        }
    }

    pub fn new(types: TypeHierarchy, transformations: Vec<Transformation>) -> Self {
        Self {
            types,
            transformations,
        }
    }

    pub fn get_types(&self) -> &TypeHierarchy {
        &self.types
    }

    pub fn get_transformations(&self) -> &Vec<Transformation> {
        &self.transformations
    }
}

#[cfg(test)]
mod tests {
    use crate::parsing::{interpretation::Interpretation, parser::Parser, tokenizer::Tokenizer};

    use super::*;

    #[test]
    fn test_context_initializes() {
        let context = Context::empty();
        assert_eq!(context.get_types(), &TypeHierarchy::new());
        assert_eq!(context.get_transformations(), &vec![]);

        let types = TypeHierarchy::chain(vec![
            "Complex".into(),
            "Real".into(),
            "Rational".into(),
            "Integer".into(),
        ])
        .unwrap();

        let equals_interpretation = Interpretation::infix_operator("=".into());
        let plus_interpretation = Interpretation::infix_operator("+".into());
        let times_interpretation = Interpretation::infix_operator("*".into());

        let mut parser = Parser::new(vec![
            equals_interpretation,
            plus_interpretation,
            times_interpretation,
        ]);

        let mut a_plus_b = parser
            .parse(
                &mut Tokenizer::new_with_tokens(vec!["+".to_string(), "=".to_string()])
                    .tokenize("a + b"),
            )
            .unwrap();
        let mut b_plus_a = parser
            .parse(
                &mut Tokenizer::new_with_tokens(vec!["+".to_string(), "=".to_string()])
                    .tokenize("b + a"),
            )
            .unwrap();
        let additive_commutativity = Transformation::new(a_plus_b, b_plus_a);

        let mut a_times_b = parser
            .parse(
                &mut Tokenizer::new_with_tokens(vec!["*".to_string(), "=".to_string()])
                    .tokenize("a * b"),
            )
            .unwrap();
        let mut b_times_a = parser
            .parse(
                &mut Tokenizer::new_with_tokens(vec!["*".to_string(), "=".to_string()])
                    .tokenize("b * a"),
            )
            .unwrap();
        let multiplicative_commutativity = Transformation::new(a_times_b, b_times_a);

        let transformations = vec![additive_commutativity, multiplicative_commutativity];
        let context = Context::new(types, transformations);
    }
}
