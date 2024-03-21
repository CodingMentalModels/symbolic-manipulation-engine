use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::symbol::{
    symbol_node::{SymbolNode, SymbolNodeAddress},
    transformation::{Transformation, TransformationError},
};

type StatementIndex = usize;
type TransformationIndex = usize;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Workspace {
    statements: Vec<SymbolNode>,
    transformations: Vec<Transformation>,
    provenance: Vec<Provenance>,
}

impl Workspace {
    pub fn new() -> Workspace {
        Self::default()
    }

    pub fn add_statement(&mut self, statement: SymbolNode) {
        self.statements.push(statement);
        self.provenance.push(Provenance::Hypothesis);
    }

    pub fn add_transformation(&mut self, transformation: Transformation) {
        self.transformations.push(transformation);
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
            .transform_all(statement, substitutions)
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
            .transform_at(statement, address.clone())
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
    TransformationError(TransformationError),
}

#[cfg(test)]
mod test_workspace {

    use std::unimplemented;

    use crate::context::context::Context;

    use super::*;

    #[test]
    fn test_workspace_adds_and_deletes_statement() {
        let mut workspace = Workspace::new();
        let statement = SymbolNode::leaf_object("a".to_string());
        workspace.add_statement(statement);
        assert_eq!(workspace.statements.len(), 1);

        let statement = SymbolNode::leaf_object("b".to_string());
        workspace.add_statement(statement);

        assert_eq!(workspace.statements.len(), 2);
    }

    #[test]
    fn test_workspace_transforms_statement_and_maintains_provenance() {
        let mut workspace = Workspace::new();
        let statement = SymbolNode::leaf_object("a".to_string());
        workspace.add_statement(statement);
        assert_eq!(workspace.statements.len(), 1);

        let transformation = Transformation::new(
            SymbolNode::leaf_object("a".to_string()),
            SymbolNode::leaf_object("b".to_string()),
        );
        workspace.add_transformation(transformation);
        assert_eq!(workspace.transformations.len(), 1);

        workspace.transform_all(0, 0, HashMap::new());
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

        workspace.transform_at(1, 1, vec![]);
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
        //        let mut workspace = Workspace::new();
        //
        //        let context = Context::new();
        //
        //        assert_eq!(workspace.try_import_context(context), Ok(()));
        //
        //        let ambiguous_context = Context::new();
        //
        //        assert_eq!(
        //            workspace.try_import_context(context),
        //            Err(ContextError::AmbiguousTypes(vec![set_type]))
        //        );
        //
        //        assert_eq!(workspace.transformations.len(),);

        unimplemented!()
    }
}
