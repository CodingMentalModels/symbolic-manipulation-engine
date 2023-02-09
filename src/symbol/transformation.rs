use std::collections::HashMap;

use crate::symbol::symbol_node::{Symbol, SymbolNode};
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

    pub fn transform(&self, statement: SymbolNode, substitutions: HashMap<String, String>) -> Result<(), TransformationError> {
        
        statement.validate().map_err(|e| TransformationError::InvalidStatement(e))?;

        if self.get_variables() != substitutions.keys().collect() {
            return Err(TransformationError::SubstitutionKeysMismatch);
        }

        let substituted_from = self.from.substitute(substitutions);
        let substituted_to = self.to.substitute(substitutions);

        if statement == substituted_from {
            Ok(substituted_to)
        } else {
            Err(TransformationError::StatementDoesNotMatch)
        }

    }
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

        let transformed = transformation.transform(statement, vec![("a".to_string(), "c".to_string()), ("b".to_string(), "d".to_string())]);
        assert_eq!(transformed, Ok(SymbolNode::leaf_object("d".to_string())));


    }
}