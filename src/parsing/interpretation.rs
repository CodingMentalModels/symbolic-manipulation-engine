use crate::symbol::symbol_type::Type;

use super::tokenizer::Token;

type ExpressionPrecidence = u8;


#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Interpretation {
    condition: InterpretationCondition,
    expression_type: ExpressionType,
    expression_precidence: ExpressionPrecidence,
    output_type: Type,
}

impl Interpretation {

    pub fn new(condition: InterpretationCondition, expression_type: ExpressionType, expression_precidence: ExpressionPrecidence, output_type: Type) -> Self {
        Interpretation {
            condition,
            expression_type,
            expression_precidence,
            output_type,
        }
    }

}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum InterpretationCondition {
    Matches(Token),
}


#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ExpressionType {
    Default,
    Prefix,
    Infix,
    Postfix,
    Outfix,
}

impl Default for ExpressionType {
    fn default() -> Self {
        ExpressionType::Default
    }
}

#[cfg(test)]
mod test_interpretation {
    use super::*;

    #[test]
    fn test_() {
        
    }
}