use std::collections::{HashMap, HashSet};

use crate::symbol::{symbol_node::{SymbolNode, SymbolNodeAddress}, transformation::{Transformation, TransformationError}};

type StatementIndex = usize;
type TransformationIndex = usize;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
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

    pub fn delete_statement(&mut self, index: StatementIndex) -> Result<SymbolNode, WorkspaceError> {
        if self.statement_index_is_invalid(index) {
            return Err(WorkspaceError::InvalidStatementIndex);
        }
        let to_return = self.statements.remove(index);
        let _provenance = self.provenance.remove(index);

        return Ok(to_return);
    }

    pub fn add_transformation(&mut self, transformation: Transformation) {
        self.transformations.push(transformation);
    }

    pub fn transform_all(&mut self, transformation_index: TransformationIndex, statement_index: StatementIndex, substitutions: HashMap<String, String>) -> Result<SymbolNode, WorkspaceError> {
        if self.transformation_index_is_invalid(transformation_index) {
            return Err(WorkspaceError::InvalidTransformationIndex);
        }
        if self.statement_index_is_invalid(statement_index) {
            return Err(WorkspaceError::InvalidStatementIndex);
        }
        let transformation = self.transformations[transformation_index].clone();
        let statement = self.statements[statement_index].clone();
        let (transformed_statement, transformed_addresses) = transformation.transform_all(statement, substitutions).map_err(|e| WorkspaceError::TransformationError(e))?;

        self.statements.push(transformed_statement.clone());
        self.provenance.push(Provenance::Derived((statement_index, transformation_index, transformed_addresses)));

        return Ok(transformed_statement);
    }

    pub fn transform_at(&mut self, transformation_index: TransformationIndex, statement_index: StatementIndex, address: SymbolNodeAddress) -> Result<SymbolNode, WorkspaceError> {
        if self.transformation_index_is_invalid(transformation_index) {
            return Err(WorkspaceError::InvalidTransformationIndex);
        }
        if self.statement_index_is_invalid(statement_index) {
            return Err(WorkspaceError::InvalidStatementIndex);
        }
        let transformation = self.transformations[transformation_index].clone();
        let statement = self.statements[statement_index].clone();
        let transformed_statement = transformation.transform_at(statement, address.clone()).map_err(|_| WorkspaceError::InvalidTransformationAddress)?;

        self.statements.push(transformed_statement.clone());
        self.provenance.push(Provenance::Derived((statement_index, transformation_index, vec![address])));

        return Ok(transformed_statement);
    }

    pub fn get_provenance_lineage(&self, index: StatementIndex) -> Result<Vec<Provenance>, WorkspaceError> {
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

    fn statement_index_is_invalid(&self, index: StatementIndex) -> bool {
        index >= self.statements.len() || index >= self.provenance.len()
    }

    fn transformation_index_is_invalid(&self, index: TransformationIndex) -> bool {
        index >= self.transformations.len()
    }

}

#[derive(Clone, Debug, PartialEq, Eq)]
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

        workspace.delete_statement(0);
        assert_eq!(workspace.statements.len(), 1);
        assert_eq!(workspace.statements, vec![SymbolNode::leaf_object("b".to_string())]);
    }

    #[test]
    fn test_workspace_transforms_statement_and_maintains_provenance() {
        let mut workspace = Workspace::new();
        let statement = SymbolNode::leaf_object("a".to_string());
        workspace.add_statement(statement);
        assert_eq!(workspace.statements.len(), 1);

        let transformation = Transformation::new(
            SymbolNode::leaf_object("a".to_string()),
            SymbolNode::leaf_object("b".to_string())
        );
        workspace.add_transformation(transformation);
        assert_eq!(workspace.transformations.len(), 1);

        workspace.transform_all(0, 0, HashMap::new());
        assert_eq!(workspace.statements.len(), 2);
        assert_eq!(workspace.statements, vec![SymbolNode::leaf_object("a".to_string()), SymbolNode::leaf_object("b".to_string())]);

        assert_eq!(workspace.get_provenance(0), Ok(Provenance::Hypothesis));
        assert_eq!(workspace.get_provenance(1), Ok(Provenance::Derived((0, 0, vec![vec![]]))));

        assert_eq!(workspace.get_provenance_lineage(0), Ok(vec![Provenance::Hypothesis]));
        assert_eq!(workspace.get_provenance_lineage(1), Ok(vec![Provenance::Derived((0, 0, vec![vec![]])), Provenance::Hypothesis]));

        workspace.add_transformation(Transformation::new(
            SymbolNode::leaf_object("b".to_string()),
            SymbolNode::leaf_object("=".to_string())
        ));

        workspace.transform_at(1, 1, vec![]);
        assert_eq!(workspace.statements.len(), 3);
        assert_eq!(workspace.statements, vec![SymbolNode::leaf_object("a".to_string()), SymbolNode::leaf_object("b".to_string()), SymbolNode::leaf_object("=".to_string())]);

        assert_eq!(workspace.get_provenance(2), Ok(Provenance::Derived((1, 1, vec![vec![]]))));

    }
    }