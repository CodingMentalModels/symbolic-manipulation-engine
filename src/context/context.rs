use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{
    parsing::interpretation::Interpretation,
    symbol::{
        symbol_type::{GeneratedType, Type, TypeError, TypeHierarchy},
        transformation::{Transformation, TransformationLattice},
    },
    workspace::workspace::Workspace,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Context {
    types: TypeHierarchy,
    generated_types: Vec<GeneratedType>,
    interpretations: Vec<Interpretation>,
    transformation_lattice: TransformationLattice,
}

impl Context {
    pub fn empty() -> Self {
        Self {
            types: TypeHierarchy::new(),
            generated_types: vec![],
            interpretations: vec![],
            transformation_lattice: TransformationLattice::empty(),
        }
    }

    pub fn new(
        types: TypeHierarchy,
        generated_types: Vec<GeneratedType>,
        interpretations: Vec<Interpretation>,
        transformation_lattice: TransformationLattice,
    ) -> Result<Self, ContextError> {
        match transformation_lattice
            .get_available_transformations()
            .iter()
            .map(|transformation| types.binds_transformation_or_error(transformation))
            .next()
        {
            None | Some(Ok(())) => Ok(Self {
                types,
                generated_types,
                interpretations,
                transformation_lattice,
            }),
            Some(Err(e)) => Err(ContextError::from(e)),
        }
    }

    pub fn from_workspace(workspace: &Workspace) -> Result<Self, ContextError> {
        let lattice = TransformationLattice::from_transformations(
            workspace
                .get_transformation_lattice()
                .get_available_transformations()
                .clone(),
        )
        .map_err(|_| {
            ContextError::InvalidTransformationLattice(
                workspace.get_transformation_lattice().clone(),
            )
        })?;
        Self::new(
            workspace.get_types().clone(),
            workspace.get_generated_types().clone(),
            workspace.get_interpretations().clone(),
            lattice,
        )
    }

    pub fn get_types(&self) -> &TypeHierarchy {
        &self.types
    }

    pub fn get_generated_types(&self) -> &Vec<GeneratedType> {
        &self.generated_types
    }

    pub fn get_interpretations(&self) -> &Vec<Interpretation> {
        &self.interpretations
    }

    pub fn get_transformation_lattice(&self) -> &TransformationLattice {
        &self.transformation_lattice
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
    InvalidTransformationLattice(TransformationLattice),
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
    use crate::{
        parsing::{interpretation::Interpretation, parser::Parser, tokenizer::Tokenizer},
        symbol::transformation::ExplicitTransformation,
    };

    use super::*;

    #[test]
    fn test_context_expresses_group_theory() {
        let context = Context::empty();
        assert_eq!(context.get_types(), &TypeHierarchy::new());
        assert_eq!(
            context.get_transformation_lattice(),
            &TransformationLattice::empty()
        );

        let mut types = TypeHierarchy::chain(vec!["Group Element".into(), "*".into()]).unwrap();
        types
            .add_child_to_parent("inv".into(), "Group Element".into())
            .unwrap();
        types
            .add_child_to_parent("1".into(), "Group Element".into())
            .unwrap();
        types.add_chain(vec!["=".into()]).unwrap();

        let equals_interpretation = Interpretation::infix_operator("=".into(), 1, "=".into());
        let times_interpretation = Interpretation::infix_operator("*".into(), 3, "*".into());
        let inverse_interpretation = Interpretation::function("inv".into(), 90);
        let g_interpretation = Interpretation::singleton("g", "Group Element".into());
        let one_interpretation = Interpretation::singleton("1", "1".into());

        let parser = Parser::new(vec![
            equals_interpretation,
            times_interpretation,
            inverse_interpretation,
            g_interpretation,
            one_interpretation,
        ]);

        let commutativity = ExplicitTransformation::symmetry(
            "*".to_string(),
            "*".into(),
            ("a".to_string(), "b".to_string()),
            "Group Element".into(),
        );

        let associativity = ExplicitTransformation::associativity(
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
        let identity = ExplicitTransformation::new(identity_from, identity_to);

        let inverse_from = parser
            .parse_from_string(vec!["*".to_string()], "g*inv(g)")
            .unwrap();
        let inverse_to = parser
            .parse_from_string(vec!["*".to_string()], "1")
            .unwrap();
        let inverse = ExplicitTransformation::new(inverse_from, inverse_to);

        let transformations = vec![commutativity, associativity, identity, inverse]
            .into_iter()
            .map(|t| t.into())
            .collect::<HashSet<Transformation>>();
        let lattice = TransformationLattice::from_transformations(transformations.clone()).unwrap();

        let context = Context::new(types.clone(), vec![], vec![], lattice.clone());

        assert_eq!(context.clone().unwrap().get_types(), &types);
        assert_eq!(
            context.clone().unwrap().get_transformation_lattice(),
            &lattice.clone()
        );
    }

    #[test]
    fn test_context_initializes() {
        let context = Context::empty();
        assert_eq!(context.get_types(), &TypeHierarchy::new());
        assert_eq!(
            context.get_transformation_lattice(),
            &TransformationLattice::empty()
        );

        let mut types = TypeHierarchy::chain(vec![
            "Complex".into(),
            "Real".into(),
            "Rational".into(),
            "Integer".into(),
        ])
        .unwrap();
        types.add_chain(vec!["Operator".into(), "+".into()]);
        types.add_child_to_parent("*".into(), "Operator".into());
        let equals_interpretation = Interpretation::infix_operator("=".into(), 1, "Integer".into());
        let plus_interpretation = Interpretation::infix_operator("+".into(), 2, "Integer".into());
        let times_interpretation = Interpretation::infix_operator("*".into(), 3, "Integer".into());

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
        let additive_commutativity = ExplicitTransformation::new(a_plus_b, b_plus_a);

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
        let multiplicative_commutativity = ExplicitTransformation::new(a_times_b, b_times_a);

        let transformations = vec![additive_commutativity, multiplicative_commutativity]
            .into_iter()
            .map(|t| t.into())
            .collect::<HashSet<Transformation>>();
        let lattice = TransformationLattice::from_transformations(transformations).unwrap();

        let context = Context::new(types.clone(), vec![], vec![], lattice.clone());

        assert_eq!(context.clone().unwrap().get_types(), &types);
        assert_eq!(
            context.clone().unwrap().get_transformation_lattice(),
            &lattice.clone()
        );
    }
}
