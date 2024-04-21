use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{
    context::context::Context,
    parsing::{
        interpretation::Interpretation,
        parser::{Parser, ParserError},
    },
    symbol::{
        symbol_node::{SymbolNode, SymbolNodeAddress},
        symbol_type::{GeneratedType, Type, TypeError, TypeHierarchy},
        transformation::{Transformation, TransformationError},
    },
};

type StatementIndex = usize;
type TransformationIndex = usize;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Workspace {
    types: TypeHierarchy,
    generated_types: Vec<GeneratedType>,
    interpretations: Vec<Interpretation>,
    statements: Vec<SymbolNode>,
    transformations: Vec<Transformation>,
    provenance: Vec<Provenance>,
}

impl Workspace {
    pub fn new(
        types: TypeHierarchy,
        generated_types: Vec<GeneratedType>,
        interpretations: Vec<Interpretation>,
    ) -> Workspace {
        Self {
            types,
            generated_types,
            interpretations,
            statements: vec![],
            transformations: vec![],
            provenance: vec![],
        }
    }

    pub fn get_types(&self) -> &TypeHierarchy {
        &self.types
    }

    pub fn try_import_context(&mut self, context: Context) -> Result<(), WorkspaceError> {
        let mut shared_types = self.types.get_shared_types(context.get_types());
        shared_types.remove(&Type::Object);
        if shared_types.len() > 0 {
            return Err(WorkspaceError::AttemptedToImportAmbiguousTypes(
                shared_types,
            ));
        }
        let new_types = self
            .types
            .union(context.get_types())
            .map_err(|e| WorkspaceError::from(e))?;

        if context.get_generated_types().len() > 0 {
            if self.generated_types.len() > 0
                && &self.generated_types != context.get_generated_types()
            {
                return Err(WorkspaceError::UnsupportedOperation("For simplicity, we don't support importing Contexts with different generated types than the workspace.".to_string()));
            }
        }

        let mut new_generated_types = self.generated_types.clone();
        new_generated_types.append(&mut context.get_generated_types().clone());

        if context.get_interpretations().len() > 0 {
            if self.interpretations.len() > 0
                && &self.interpretations != context.get_interpretations()
            {
                return Err(WorkspaceError::UnsupportedOperation("For simplicity, we don't support importing Contexts with different interpretations than the workspace.".to_string()));
            }
        }

        let mut new_interpretations = self.interpretations.clone();
        new_interpretations.append(&mut context.get_interpretations().clone());

        self.types = new_types;
        self.generated_types = new_generated_types;
        self.interpretations = new_interpretations;
        self.transformations = context.get_transformations().clone();
        Ok(())
    }

    pub fn parse_from_string(&mut self, s: &str) -> Result<SymbolNode, WorkspaceError> {
        let parser = Parser::new(self.interpretations.clone());
        parser
            .parse_from_string(parser.get_interpretation_custom_tokens(), s)
            .map_err(|e| e.into())
    }

    pub fn add_statement(&mut self, statement: SymbolNode) -> Result<(), WorkspaceError> {
        self.types
            .binds_statement_or_error(&statement)
            .map_err(|x| WorkspaceError::from(x))?;
        self.generate_types_in_bulk(vec![statement.clone()].into_iter().collect());
        self.statements.push(statement);
        self.provenance.push(Provenance::Hypothesis);
        Ok(())
    }

    pub fn add_transformation(
        &mut self,
        transformation: Transformation,
    ) -> Result<(), WorkspaceError> {
        self.types.binds_transformation_or_error(&transformation)?;
        self.generate_types_in_bulk(
            vec![
                transformation.get_from().clone(),
                transformation.get_to().clone(),
            ]
            .into_iter()
            .collect(),
        );
        self.transformations.push(transformation);
        Ok(())
    }

    pub fn try_transform_into(
        &mut self,
        desired: SymbolNode,
    ) -> Result<SymbolNode, WorkspaceError> {
        for statement in self.statements.iter() {
            for transform in self.transformations.iter() {
                match transform.try_transform_into(self.get_types(), &statement, &desired) {
                    Ok(output) => return Ok(output),
                    Err(e) => {}
                }
            }
        }
        return Err(WorkspaceError::NoTransformationsPossible);
    }

    pub fn transform_all(
        &mut self,
        transformation_index: TransformationIndex,
        statement_index: StatementIndex,
        substitutions: HashMap<String, String>,
    ) -> Result<SymbolNode, WorkspaceError> {
        if self.transformation_index_is_invalid(transformation_index) {
            return Err(WorkspaceError::InvalidTransformationIndex);
        }
        if self.statement_index_is_invalid(statement_index) {
            return Err(WorkspaceError::InvalidStatementIndex);
        }
        let transformation = self.transformations[transformation_index].clone();
        let statement = self.statements[statement_index].clone();
        let (transformed_statement, transformed_addresses) = transformation
            .transform_all(&statement, &substitutions)
            .map_err(|e| WorkspaceError::TransformationError(e))?;

        self.statements.push(transformed_statement.clone());
        self.provenance.push(Provenance::Derived((
            statement_index,
            transformation_index,
            transformed_addresses,
        )));

        return Ok(transformed_statement);
    }

    pub fn transform_at(
        &mut self,
        transformation_index: TransformationIndex,
        statement_index: StatementIndex,
        address: SymbolNodeAddress,
    ) -> Result<SymbolNode, WorkspaceError> {
        if self.transformation_index_is_invalid(transformation_index) {
            return Err(WorkspaceError::InvalidTransformationIndex);
        }
        if self.statement_index_is_invalid(statement_index) {
            return Err(WorkspaceError::InvalidStatementIndex);
        }
        let transformation = self.transformations[transformation_index].clone();
        let statement = self.statements[statement_index].clone();
        let transformed_statement = transformation
            .transform_at(&statement, address.clone())
            .map_err(|_| WorkspaceError::InvalidTransformationAddress)?;

        self.statements.push(transformed_statement.clone());
        self.provenance.push(Provenance::Derived((
            statement_index,
            transformation_index,
            vec![address],
        )));

        return Ok(transformed_statement);
    }

    pub fn get_provenance_lineage(
        &self,
        index: StatementIndex,
    ) -> Result<Vec<Provenance>, WorkspaceError> {
        let mut provenance = Vec::new();
        let mut current_index = index;
        loop {
            let current_provenance = self.get_provenance(current_index)?;
            provenance.push(current_provenance.clone());
            match current_provenance {
                Provenance::Hypothesis => break,
                Provenance::Derived((parent_index, _, _)) => current_index = parent_index,
            }
        }
        Ok(provenance)
    }

    pub fn get_provenance(&self, index: StatementIndex) -> Result<Provenance, WorkspaceError> {
        if self.statement_index_is_invalid(index) {
            return Err(WorkspaceError::InvalidStatementIndex);
        }

        Ok(self.provenance[index].clone())
    }

    pub fn to_string(&self) -> String {
        let mut to_return = Vec::new();
        let transformation_header = "Transformations:".to_string();
        to_return.push(transformation_header);

        self.transformations
            .iter()
            .enumerate()
            .for_each(|(index, transformation)| {
                to_return.push(format!("{}: {}", index, transformation.to_string()));
            });
        to_return.push("".to_string());

        let statement_header = "Statements:".to_string();
        to_return.push(statement_header);

        self.statements
            .iter()
            .enumerate()
            .for_each(|(index, statement)| {
                to_return.push(format!("{}: {}", index, statement.to_string()));
            });

        return to_return.join("\n");
    }

    pub fn serialize(&self) -> String {
        toml::to_string(self).unwrap()
    }

    pub fn deserialize(serialized: &str) -> Result<Workspace, toml::de::Error> {
        toml::from_str(serialized)
    }

    fn statement_index_is_invalid(&self, index: StatementIndex) -> bool {
        index >= self.statements.len() || index >= self.provenance.len()
    }

    fn transformation_index_is_invalid(&self, index: TransformationIndex) -> bool {
        index >= self.transformations.len()
    }

    fn generate_types_in_bulk(
        &mut self,
        statements: HashSet<SymbolNode>,
    ) -> Result<(), WorkspaceError> {
        for statement in statements {
            match self.generate_types(statement) {
                Ok(()) => {}
                Err(e) => {
                    return Err(e.into());
                }
            };
        }
        Ok(())
    }

    fn generate_types(&mut self, statement: SymbolNode) -> Result<(), WorkspaceError> {
        for generated_type in self.generated_types.iter() {
            let result: Option<TypeError> = generated_type
                .generate(&statement)
                .into_iter()
                .map(|(t, parents)| self.types.add_child_to_parents(t, parents))
                .filter(|r| r.is_err())
                .map(|r| r.expect_err("We just checked that it's an error."))
                .next();
            match result {
                Some(e) => return Err(e.into()),
                None => {}
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provenance {
    Hypothesis,
    Derived((TransformationIndex, StatementIndex, Vec<SymbolNodeAddress>)),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceError {
    InvalidStatementIndex,
    InvalidTransformationIndex,
    InvalidTransformationAddress,
    ParserError(ParserError),
    TransformationError(TransformationError),
    StatementContainsTypesNotInHierarchy(HashSet<Type>),
    IncompatibleTypeRelationships(HashSet<Type>),
    NoTransformationsPossible,
    InvalidTypeErrorTransformation(TypeError),
    AttemptedToImportAmbiguousTypes(HashSet<Type>),
    UnsupportedOperation(String),
}

impl From<ParserError> for WorkspaceError {
    fn from(value: ParserError) -> Self {
        Self::ParserError(value)
    }
}

impl From<TypeError> for WorkspaceError {
    fn from(value: TypeError) -> Self {
        match value {
            TypeError::StatementIncludesTypesNotInHierarchy(types) => {
                Self::StatementContainsTypesNotInHierarchy(types)
            }
            TypeError::IncompatibleTypeRelationships(types) => {
                Self::IncompatibleTypeRelationships(types)
            }
            e => Self::InvalidTypeErrorTransformation(e),
        }
    }
}

#[cfg(test)]
mod test_workspace {

    use std::unimplemented;

    use crate::{
        context::context::Context,
        parsing::{interpretation::Interpretation, parser::Parser},
        symbol::symbol_type::{GeneratedType, GeneratedTypeCondition, TypeHierarchy},
    };

    use super::*;

    //    #[test]
    //    fn test_workspace_try_transform_into() {
    //        unimplemented!();
    //    }

    #[test]
    fn test_workspace_adds_and_deletes_statement() {
        let types = TypeHierarchy::new();
        let mut workspace = Workspace::new(types, vec![], vec![]);
        let statement = SymbolNode::leaf_object("a".to_string());
        workspace.add_statement(statement);
        assert_eq!(workspace.statements.len(), 1);

        let statement = SymbolNode::leaf_object("b".to_string());
        workspace.add_statement(statement);

        assert_eq!(workspace.statements.len(), 2);
    }

    #[test]
    fn test_workspace_adds_statement_with_generated_type() {
        let plus = Interpretation::infix_operator("+".into(), 1, "Integer".into());
        let integer = GeneratedType::new(
            GeneratedTypeCondition::IsInteger,
            vec!["Integer".into()].into_iter().collect(),
        );

        let mut types = TypeHierarchy::chain(vec!["Real".into(), "Integer".into()]).unwrap();
        types.add_chain(vec!["+".into()]);
        let mut workspace = Workspace::new(types, vec![integer], vec![]);

        let parser = Parser::new(vec![plus]);
        let two_plus_two = parser
            .parse_from_string(vec!["+".to_string()], "2+2")
            .unwrap();

        assert_eq!(workspace.add_statement(two_plus_two), Ok(()));
        let mut expected =
            TypeHierarchy::chain(vec!["Real".into(), "Integer".into(), "2".into()]).unwrap();
        expected.add_chain(vec!["+".into()]);
        assert_eq!(workspace.types, expected);
    }

    #[test]
    fn test_workspace_transforms_statement_and_maintains_provenance() {
        let types = TypeHierarchy::new();
        let mut workspace = Workspace::new(types, vec![], vec![]);
        let statement = SymbolNode::leaf_object("a".to_string());
        assert_eq!(workspace.add_statement(statement), Ok(()));
        assert_eq!(workspace.statements.len(), 1);

        let transformation = Transformation::new(
            SymbolNode::leaf_object("a".to_string()),
            SymbolNode::leaf_object("b".to_string()),
        );
        workspace.add_transformation(transformation);
        assert_eq!(workspace.transformations.len(), 1);

        let _transformed = workspace.transform_all(0, 0, HashMap::new());
        assert_eq!(workspace.statements.len(), 2);
        assert_eq!(
            workspace.statements,
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::leaf_object("b".to_string())
            ]
        );

        assert_eq!(workspace.get_provenance(0), Ok(Provenance::Hypothesis));
        assert_eq!(
            workspace.get_provenance(1),
            Ok(Provenance::Derived((0, 0, vec![vec![]])))
        );

        assert_eq!(
            workspace.get_provenance_lineage(0),
            Ok(vec![Provenance::Hypothesis])
        );
        assert_eq!(
            workspace.get_provenance_lineage(1),
            Ok(vec![
                Provenance::Derived((0, 0, vec![vec![]])),
                Provenance::Hypothesis
            ])
        );

        workspace.add_transformation(Transformation::new(
            SymbolNode::leaf_object("b".to_string()),
            SymbolNode::leaf_object("=".to_string()),
        ));

        let _transformed = workspace.transform_at(1, 1, vec![]);
        assert_eq!(workspace.statements.len(), 3);
        assert_eq!(
            workspace.statements,
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::leaf_object("b".to_string()),
                SymbolNode::leaf_object("=".to_string())
            ]
        );

        assert_eq!(
            workspace.get_provenance(2),
            Ok(Provenance::Derived((1, 1, vec![vec![]])))
        );
    }

    #[test]
    fn test_workspace_imports_context() {
        let types = TypeHierarchy::new();
        let mut workspace = Workspace::new(types, vec![], vec![]);

        let mut types =
            TypeHierarchy::chain(vec!["Real".into(), "Rational".into(), "Integer".into()]).unwrap();
        types.add_chain(vec!["Operator".into(), "=".into()]);
        types.add_chain_to_parent(vec!["+".into()], "Operator".into());

        let equality_reflexivity = Transformation::reflexivity(
            "=".to_string(),
            "=".into(),
            "x".to_string(),
            "Real".into(),
        );

        let equality_symmetry = Transformation::symmetry(
            "=".to_string(),
            "=".into(),
            ("x".to_string(), "y".to_string()),
            "Real".into(),
        );

        let context = Context::new(
            types.clone(),
            vec![],
            vec![],
            vec![equality_reflexivity.clone(), equality_symmetry.clone()],
        )
        .unwrap();

        assert_eq!(workspace.try_import_context(context.clone()), Ok(()));
        assert_eq!(workspace.types, types);
        assert_eq!(workspace.transformations.len(), 2);

        let mut complex_types = TypeHierarchy::chain(vec![
            "Complex".into(),
            "Real".into(),
            "Rational".into(),
            "Integer".into(),
        ])
        .unwrap();

        complex_types.add_chain(vec!["Operator".into(), "=".into()]);
        complex_types.add_chain_to_parent(vec!["+".into()], "Operator".into());
        let ambiguous_context = Context::new(
            complex_types,
            vec![],
            vec![],
            vec![equality_reflexivity, equality_symmetry],
        )
        .unwrap();

        assert_eq!(
            workspace.try_import_context(ambiguous_context),
            Err(WorkspaceError::AttemptedToImportAmbiguousTypes(
                vec![
                    "Real".into(),
                    "Rational".into(),
                    "Integer".into(),
                    "=".into(),
                    "+".into(),
                    "Operator".into()
                ]
                .into_iter()
                .collect()
            ))
        );

        assert_eq!(workspace.transformations.len(), 2);

        let mut inverted_types =
            TypeHierarchy::chain(vec!["Rational".into(), "Real".into()]).unwrap();
        let conflicting_context = Context::new(inverted_types, vec![], vec![], vec![]).unwrap();

        assert_eq!(
            workspace.try_import_context(conflicting_context),
            Err(WorkspaceError::AttemptedToImportAmbiguousTypes(
                vec!["Real".into(), "Rational".into(),]
                    .into_iter()
                    .collect()
            ))
        );
    }
}
