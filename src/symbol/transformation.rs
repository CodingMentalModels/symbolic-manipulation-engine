use std::collections::{HashMap, HashSet};

use crate::symbol::symbol_node::{Symbol, SymbolNodeError, SymbolNode};
use crate::symbol::symbol_type::Type;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Transformation {
    pub from: SymbolNode,
    pub to: SymbolNode,
}

impl Transformation {

    pub fn new(from: SymbolNode, to: SymbolNode) -> Transformation {
        Transformation { from, to }
    }

    pub fn get_variables(&self) -> HashSet<String> {
        let mut variables = self.from.get_symbols();
        variables.extend(self.to.get_symbols());
        variables.into_iter().map(|s| s.get_name()).collect()
    }

    pub fn transform_all(&self, statement: SymbolNode, substitutions: HashMap<String, String>) -> Result<SymbolNode, TransformationError> {

        let transformed_children = statement.get_children().iter().map(
            |c| self.transform_all(c.clone(), substitutions.clone()))
            .collect::<Result<Vec<SymbolNode>, TransformationError>>()?;
        let new_statement = SymbolNode::new(statement.get_symbol().clone(), transformed_children);

        self.try_transform(new_statement, substitutions)

    }

    pub fn try_transform(&self, statement: SymbolNode, substitutions: HashMap<String, String>) -> Result<SymbolNode, TransformationError> {
        
        match self.transform(statement.clone(), substitutions) {
            Ok(transformed) => Ok(transformed),
            Err(TransformationError::StatementDoesNotMatch) => Ok(statement),
            Err(e) => Err(e)
        }
    }

    pub fn transform_strict(&self, statement: SymbolNode, substitutions: HashMap<String, String>) -> Result<SymbolNode, TransformationError> {
        
        if self.get_variables() != substitutions.keys().cloned().collect() {
            return Err(TransformationError::SubstitutionKeysMismatch);
        }

        self.transform(statement, substitutions)

    }

    pub fn transform(&self, statement: SymbolNode, substitutions: HashMap<String, String>) -> Result<SymbolNode, TransformationError> {
        
        statement.validate().map_err(|e| TransformationError::InvalidStatement(e))?;

        let substituted_from = self.from.relabel_all(substitutions.clone().into_iter().collect());
        let substituted_to = self.to.relabel_all(substitutions.into_iter().collect());

        if statement.is_generalized_by(&substituted_from) {
            Ok(substituted_to)
        } else {
            Err(TransformationError::StatementDoesNotMatch)
        }

    }

}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransformationError {
    InvalidStatement(SymbolNodeError),
    SubstitutionKeysMismatch,
    StatementDoesNotMatch,
    SttaementTypesDoNotMatch,
}


#[cfg(test)]
mod test_transformation {
    use super::*;

    #[test]
    fn test_transformation_transforms() {
        
        let transformation = Transformation::new(
            SymbolNode::leaf_object("a".to_string()),
            SymbolNode::leaf_object("b".to_string())
        );

        let statement = SymbolNode::leaf_object("c".to_string());

        let transformed = transformation.transform_strict(
            statement,
            vec![("a".to_string(), "c".to_string()), ("b".to_string(), "d".to_string())].into_iter().collect()
        );
        assert_eq!(transformed, Ok(SymbolNode::leaf_object("d".to_string())));


        let transformation = Transformation::new(
            SymbolNode::new_generic("=".to_string(),
        vec![
                    SymbolNode::leaf_object("a".to_string()),
                    SymbolNode::leaf_object("b".to_string())
                ]
            ),
            SymbolNode::new_generic("=".to_string(), 
                vec![
                    SymbolNode::leaf_object("b".to_string()),
                    SymbolNode::leaf_object("a".to_string())
                ]
            ),
        );

        let self_equals_statement = SymbolNode::new_generic("=".to_string(),
    vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::leaf_object("a".to_string())
            ]
        );

        let transformed = transformation.transform_strict(
            self_equals_statement.clone(),
            vec![("a".to_string(), "a".to_string()), ("b".to_string(), "a".to_string()), ("=".to_string(), "=".to_string())].into_iter().collect()
        );
        assert_eq!(transformed, Ok(self_equals_statement));

    }

    #[test]
    fn test_transformation_transforms_all() {
        
        let commutativity = Transformation::new(
            SymbolNode::new_generic("=".to_string(),
        vec![
                    SymbolNode::leaf_object("a".to_string()),
                    SymbolNode::leaf_object("b".to_string())
                ]
            ),
            SymbolNode::new_generic("=".to_string(), 
                vec![
                    SymbolNode::leaf_object("b".to_string()),
                    SymbolNode::leaf_object("a".to_string())
                ]
            ),
        );

        let statement = SymbolNode::new_generic(
            "=".to_string(),
            vec![
                SymbolNode::new_generic(
                    "=".to_string(),
                    vec![
                        SymbolNode::leaf_object("a".to_string()),
                        SymbolNode::leaf_object("b".to_string())
                    ]
                ),
                SymbolNode::leaf_object("c".to_string())
            ]
        );

        let transformed = commutativity.transform_all(statement.clone(), vec![("a".to_string(), "a".to_string()), ("b".to_string(), "b".to_string()), ("c".to_string(), "c".to_string()), ("=".to_string(), "=".to_string())].into_iter().collect());

        let expected = SymbolNode::new_generic(
            "=".to_string(),
            vec![
                SymbolNode::new_generic(
                    "=".to_string(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("a".to_string())
                    ]
                ),
                SymbolNode::leaf_object("c".to_string())
            ]
        );
        assert_eq!(transformed, Ok(expected));

    }
}