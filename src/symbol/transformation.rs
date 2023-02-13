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

    pub fn transform(&self, statement: SymbolNode, substitutions: HashMap<String, String>) -> Result<SymbolNode, TransformationError> {
        
        statement.validate().map_err(|e| TransformationError::InvalidStatement(e))?;

        if self.get_variables() != substitutions.keys().cloned().collect() {
            return Err(TransformationError::SubstitutionKeysMismatch);
        }

        let substituted_from = self.from.relabel_all(substitutions.clone().into_iter().collect());
        let substituted_to = self.to.relabel_all(substitutions.into_iter().collect());

        if statement == substituted_from {
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

        let transformed = transformation.transform(
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

        let transformed = transformation.transform(
            self_equals_statement.clone(),
            vec![("a".to_string(), "a".to_string()), ("b".to_string(), "a".to_string()), ("=".to_string(), "=".to_string())].into_iter().collect()
        );
        assert_eq!(transformed, Ok(self_equals_statement));

    }
}