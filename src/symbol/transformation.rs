use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::symbol::symbol_node::{Symbol, SymbolNode, SymbolNodeError};
use crate::symbol::symbol_type::Type;

use super::symbol_node::SymbolNodeAddress;
use super::symbol_type::TypeHierarchy;

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transformation {
    pub from: SymbolNode,
    pub to: SymbolNode,
}

impl Transformation {
    pub fn new(from: SymbolNode, to: SymbolNode) -> Transformation {
        Transformation { from, to }
    }

    pub fn to_string(&self) -> String {
        format!("{} -> {}", self.from.to_string(), self.to.to_string())
    }

    pub fn get_variables(&self) -> HashSet<String> {
        let mut variables = self.from.get_symbols();
        variables.extend(self.to.get_symbols());
        variables.into_iter().map(|s| s.get_name()).collect()
    }

    pub fn transform_at(
        &self,
        statement: SymbolNode,
        address: SymbolNodeAddress,
    ) -> Result<SymbolNode, TransformationError> {
        let substatement_to_transform = statement.get_node(address.clone()).ok_or_else(|| {
            TransformationError::InvalidSymbolNode(SymbolNodeError::InvalidAddress)
        })?;
        match self.from.get_relabelling(&substatement_to_transform) {
            Ok(substitutions) => {
                let transformed_substatement =
                    self.transform(substatement_to_transform, substitutions)?;
                match statement.replace_node(address, transformed_substatement) {
                    Ok(transformed_statement) => Ok(transformed_statement),
                    Err(e) => Err(TransformationError::InvalidSymbolNode(e)),
                }
            }
            Err(e) => Err(TransformationError::InvalidSymbolNode(e)),
        }
    }

    pub fn transform_all(
        &self,
        statement: SymbolNode,
        substitutions: HashMap<String, String>,
    ) -> Result<(SymbolNode, Vec<SymbolNodeAddress>), TransformationError> {
        self.transform_all_from_address(statement, substitutions, Vec::new())
    }

    pub fn transform_all_from_address(
        &self,
        statement: SymbolNode,
        substitutions: HashMap<String, String>,
        address: SymbolNodeAddress,
    ) -> Result<(SymbolNode, Vec<SymbolNodeAddress>), TransformationError> {
        let children_transformation_result = statement
            .get_children()
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut child_address = address.clone();
                child_address.push(i);
                self.transform_all_from_address(c.clone(), substitutions.clone(), child_address)
            })
            .collect::<Result<Vec<(SymbolNode, Vec<SymbolNodeAddress>)>, TransformationError>>()?;
        let (transformed_children, transformed_children_addresses) = children_transformation_result
            .into_iter()
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let transformed_children_addresses: Vec<SymbolNodeAddress> = transformed_children_addresses
            .into_iter()
            .flatten()
            .collect();

        let new_statement = SymbolNode::new(statement.get_symbol().clone(), transformed_children);

        match self.try_transform(new_statement, substitutions) {
            Ok((transformed, true)) => {
                let children_addresses: Vec<SymbolNodeAddress> = transformed_children_addresses
                    .into_iter()
                    .map(|mut a| {
                        let mut new_head = address.clone();
                        new_head.append(&mut a);
                        new_head
                    })
                    .collect();
                let mut addresses = vec![address];
                addresses.extend(children_addresses.clone());
                Ok((transformed, addresses))
            }
            Ok((transformed, false)) => Ok((transformed, transformed_children_addresses)),
            Err(e) => Err(e),
        }
    }

    pub fn try_transform(
        &self,
        statement: SymbolNode,
        substitutions: HashMap<String, String>,
    ) -> Result<(SymbolNode, bool), TransformationError> {
        match self.transform(statement.clone(), substitutions) {
            Ok(transformed) => Ok((transformed, true)),
            Err(TransformationError::StatementDoesNotMatch) => Ok((statement, false)),
            Err(e) => Err(e),
        }
    }

    pub fn transform_strict(
        &self,
        statement: SymbolNode,
        substitutions: HashMap<String, String>,
    ) -> Result<SymbolNode, TransformationError> {
        if self.get_variables() != substitutions.keys().cloned().collect() {
            return Err(TransformationError::SubstitutionKeysMismatch);
        }

        self.transform(statement, substitutions)
    }

    pub fn transform(
        &self,
        statement: SymbolNode,
        substitutions: HashMap<String, String>,
    ) -> Result<SymbolNode, TransformationError> {
        statement
            .validate()
            .map_err(|e| TransformationError::InvalidSymbolNode(e))?;

        let substituted_from = self
            .from
            .relabel_all(substitutions.clone().into_iter().collect());
        let substituted_to = self.to.relabel_all(substitutions.into_iter().collect());

        // Check that we aren't substituting something illegal
        let empty_hierarchy = TypeHierarchy::new();
        if empty_hierarchy.is_generalized_by(&statement, &substituted_from) {
            Ok(substituted_to)
        } else {
            Err(TransformationError::StatementDoesNotMatch)
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransformationError {
    InvalidSymbolNode(SymbolNodeError),
    SubstitutionKeysMismatch,
    StatementDoesNotMatch,
    StatementTypesDoNotMatch,
}

#[cfg(test)]
mod test_transformation {
    use super::*;

    #[test]
    fn test_transformation_transforms() {
        let transformation = Transformation::new(
            SymbolNode::leaf_object("a".to_string()),
            SymbolNode::leaf_object("b".to_string()),
        );

        let statement = SymbolNode::leaf_object("c".to_string());

        let transformed = transformation.transform_strict(
            statement,
            vec![
                ("a".to_string(), "c".to_string()),
                ("b".to_string(), "d".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(transformed, Ok(SymbolNode::leaf_object("d".to_string())));

        let transformation = Transformation::new(
            SymbolNode::new(
                "=".into(),
                vec![
                    SymbolNode::leaf_object("a".to_string()),
                    SymbolNode::leaf_object("b".to_string()),
                ],
            ),
            SymbolNode::new(
                "=".into(),
                vec![
                    SymbolNode::leaf_object("b".to_string()),
                    SymbolNode::leaf_object("a".to_string()),
                ],
            ),
        );

        let self_equals_statement = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::leaf_object("a".to_string()),
            ],
        );

        let transformed = transformation.transform_strict(
            self_equals_statement.clone(),
            vec![
                ("a".to_string(), "a".to_string()),
                ("b".to_string(), "a".to_string()),
                ("=".to_string(), "=".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(transformed, Ok(self_equals_statement));
    }

    #[test]
    fn test_transformation_transforms_all() {
        let commutativity = Transformation::new(
            SymbolNode::new(
                "=".into(),
                vec![
                    SymbolNode::leaf_object("a".to_string()),
                    SymbolNode::leaf_object("b".to_string()),
                ],
            ),
            SymbolNode::new(
                "=".into(),
                vec![
                    SymbolNode::leaf_object("b".to_string()),
                    SymbolNode::leaf_object("a".to_string()),
                ],
            ),
        );

        let statement = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![
                        SymbolNode::leaf_object("a".to_string()),
                        SymbolNode::leaf_object("b".to_string()),
                    ],
                ),
                SymbolNode::leaf_object("c".to_string()),
            ],
        );

        let (transformed, addresses) = commutativity
            .transform_all(
                statement.clone(),
                vec![
                    ("a".to_string(), "a".to_string()),
                    ("b".to_string(), "b".to_string()),
                    ("c".to_string(), "c".to_string()),
                    ("=".to_string(), "=".to_string()),
                ]
                .into_iter()
                .collect(),
            )
            .unwrap();

        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("a".to_string()),
                    ],
                ),
                SymbolNode::leaf_object("c".to_string()),
            ],
        );
        assert_eq!(transformed, expected);
        assert_eq!(addresses, vec![vec![0]]);
    }

    #[test]
    fn test_transformation_transforms_at() {
        let transformation = Transformation::new(
            SymbolNode::leaf_object("c".to_string()),
            SymbolNode::leaf_object("d".to_string()),
        );

        let a_equals_b = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::leaf_object("b".to_string()),
            ],
        );

        let d_equals_b = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("d".to_string()),
                SymbolNode::leaf_object("b".to_string()),
            ],
        );

        let transformed = transformation.transform_at(a_equals_b.clone(), vec![0]);

        assert_eq!(transformed, Ok(d_equals_b));
    }
}

