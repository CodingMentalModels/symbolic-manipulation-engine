use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::symbol::{
    symbol_type::{GeneratedType, Type, TypeError, TypeHierarchy},
    transformation::Transformation,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Context {
    types: TypeHierarchy,
    generated_types: Vec<GeneratedType>,
    transformations: Vec<Transformation>,
}

impl Context {
    pub fn empty() -> Self {
        Self {
            types: TypeHierarchy::new(),
            generated_types: vec![],
            transformations: vec![],
        }
    }

    pub fn new(
        types: TypeHierarchy,
        generated_types: Vec<GeneratedType>,
        transformations: Vec<Transformation>,
    ) -> Result<Self, ContextError> {
        match transformations
            .iter()
            .map(|transformation| types.binds_transformation_or_error(transformation))
            .next()
        {
            None | Some(Ok(())) => Ok(Self {
                types,
                generated_types,
                transformations,
            }),
            Some(Err(e)) => Err(ContextError::from(e)),
        }
    }

    pub fn get_types(&self) -> &TypeHierarchy {
        &self.types
    }

    pub fn get_generated_types(&self) -> &Vec<GeneratedType> {
        &self.generated_types
    }

    pub fn get_transformations(&self) -> &Vec<Transformation> {
        &self.transformations
    }

    pub fn serialize(&self) -> String {
        toml::to_string(self).unwrap()
    }

    pub fn deserialize(serialized: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(serialized)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextError {
    StatementIncludesTypesNotInHierarchy(HashSet<Type>),
    InvalidTypeErrorTransformation(TypeError),
}

impl From<TypeError> for ContextError {
    fn from(value: TypeError) -> Self {
        match value {
            TypeError::StatementIncludesTypesNotInHierarchy(types) => {
                Self::StatementIncludesTypesNotInHierarchy(types)
            }
            e => Self::InvalidTypeErrorTransformation(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::parsing::{interpretation::Interpretation, parser::Parser, tokenizer::Tokenizer};

    use super::*;

    #[test]
    fn test_context_expresses_group_theory() {
        let context = Context::empty();
        assert_eq!(context.get_types(), &TypeHierarchy::new());
        assert_eq!(context.get_transformations(), &vec![]);

        let mut types = TypeHierarchy::chain(vec!["Group Element".into()]).unwrap();
        types.add_chain(vec!["Operator".into(), "*".into()]);
        types.add_chain(vec!["=".into()]);

        let equals_interpretation = Interpretation::infix_operator("=".into(), 1);
        let times_interpretation = Interpretation::infix_operator("*".into(), 3);
        let g_interpretation = Interpretation::singleton("g", "Group Element".into());
        let one_interpretation = Interpretation::singleton("1", "Group Element".into());

        let parser = Parser::new(vec![
            equals_interpretation,
            times_interpretation,
            g_interpretation,
            one_interpretation,
        ]);

        let commutativity = Transformation::symmetry(
            "*".to_string(),
            "*".into(),
            ("a".to_string(), "b".to_string()),
            "Group Element".into(),
        );

        let associativity = Transformation::associativity(
            "*".to_string(),
            "*".into(),
            ("a".to_string(), "b".to_string(), "c".to_string()),
            "Group Element".into(),
        );

        let identity_from = parser
            .parse_from_string(vec!["*".to_string()], "g*1")
            .unwrap();
        let identity_to = parser
            .parse_from_string(vec!["*".to_string()], "g")
            .unwrap();
        let identity = Transformation::new(identity_from, identity_to);

        let inverse_from = parser
            .parse_from_string(vec!["*".to_string()], "g*inv(g)")
            .unwrap();
        assert!(false, "{:?}", inverse_from);
        let inverse_to = parser
            .parse_from_string(vec!["*".to_string()], "1")
            .unwrap();
        let inverse = Transformation::new(inverse_from, inverse_to);

        let transformations = vec![commutativity, associativity, identity, inverse];

        let context = Context::new(types.clone(), vec![], transformations.clone());

        assert_eq!(context.clone().unwrap().get_types(), &types);
        assert_eq!(
            context.clone().unwrap().get_transformations(),
            &transformations
        );
    }

    #[test]
    fn test_context_initializes() {
        let context = Context::empty();
        assert_eq!(context.get_types(), &TypeHierarchy::new());
        assert_eq!(context.get_transformations(), &vec![]);

        let mut types = TypeHierarchy::chain(vec![
            "Complex".into(),
            "Real".into(),
            "Rational".into(),
            "Integer".into(),
        ])
        .unwrap();
        types.add_chain(vec!["Operator".into(), "+".into()]);
        types.add_child_to_parent("*".into(), "Operator".into());
        let equals_interpretation = Interpretation::infix_operator("=".into(), 1);
        let plus_interpretation = Interpretation::infix_operator("+".into(), 2);
        let times_interpretation = Interpretation::infix_operator("*".into(), 3);

        let parser = Parser::new(vec![
            equals_interpretation,
            plus_interpretation,
            times_interpretation,
        ]);

        let a_plus_b = parser
            .parse(
                &mut Tokenizer::new_with_tokens(vec!["+".to_string(), "=".to_string()])
                    .tokenize("a + b"),
            )
            .unwrap();
        let b_plus_a = parser
            .parse(
                &mut Tokenizer::new_with_tokens(vec!["+".to_string(), "=".to_string()])
                    .tokenize("b + a"),
            )
            .unwrap();
        let additive_commutativity = Transformation::new(a_plus_b, b_plus_a);

        let a_times_b = parser
            .parse(
                &mut Tokenizer::new_with_tokens(vec!["*".to_string(), "=".to_string()])
                    .tokenize("a * b"),
            )
            .unwrap();
        let b_times_a = parser
            .parse(
                &mut Tokenizer::new_with_tokens(vec!["*".to_string(), "=".to_string()])
                    .tokenize("b * a"),
            )
            .unwrap();
        let multiplicative_commutativity = Transformation::new(a_times_b, b_times_a);

        let transformations = vec![additive_commutativity, multiplicative_commutativity];

        let context = Context::new(types.clone(), vec![], transformations.clone());

        assert_eq!(context.clone().unwrap().get_types(), &types);
        assert_eq!(
            context.clone().unwrap().get_transformations(),
            &transformations
        );
    }
}
