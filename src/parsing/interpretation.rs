use std::{collections::VecDeque, unimplemented};

use crate::{
    constants::*,
    parsing::parser::ParserError,
    symbol::{
        symbol_node::{Symbol, SymbolNode},
        symbol_type::Type,
    },
};

use super::{
    parser::{Parser, ParserResult},
    tokenizer::{Token, TokenStack},
};

pub type ExpressionPrecedence = u8;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Interpretation {
    condition: InterpretationCondition,
    expression_type: ExpressionType,
    expression_precidence: ExpressionPrecedence,
    output_type: InterpretedType,
}

impl Interpretation {
    pub fn new(
        condition: InterpretationCondition,
        expression_type: ExpressionType,
        expression_precidence: ExpressionPrecedence,
        output_type: InterpretedType,
    ) -> Self {
        Interpretation {
            condition,
            expression_type,
            expression_precidence,
            output_type,
        }
    }

    pub fn parentheses() -> Self {
        Interpretation::new(
            InterpretationCondition::Matches(Token::LeftParen),
            ExpressionType::Outfix(Token::RightParen),
            PARENTHESIS_PRECEDENCE,
            InterpretedType::PassThrough,
        )
    }

    pub fn any_object() -> Self {
        Interpretation::new(
            InterpretationCondition::IsObject,
            ExpressionType::Prefix,
            1,
            Type::Object.into(),
        )
    }

    pub fn get_expression_type(&self) -> ExpressionType {
        self.expression_type.clone()
    }

    pub fn get_expression_precidence(&self) -> ExpressionPrecedence {
        self.expression_precidence
    }

    pub fn get_symbol_node(
        &self,
        token: &Token,
        children: Vec<SymbolNode>,
    ) -> Result<SymbolNode, ParserError> {
        match &self.output_type {
            InterpretedType::PassThrough => {
                if children.len() != 1 {
                    Err(ParserError::InvalidPassThroughInterpretation(token.clone()))
                } else {
                    Ok(children[0].clone())
                }
            }
            InterpretedType::Type(t) => Ok(SymbolNode::new(
                Symbol::new(token.to_string(), t.clone()),
                children,
            )),
        }
    }

    pub fn satisfies_condition(&self, so_far: &Option<SymbolNode>, token: &Token) -> bool {
        let is_ok_expression_type = match self.expression_type {
            ExpressionType::Functional | ExpressionType::Prefix | ExpressionType::Outfix(_) => {
                so_far.is_none()
            }
            ExpressionType::Infix | ExpressionType::Postfix => so_far.is_some(),
        };
        if !is_ok_expression_type {
            return false;
        }
        match &self.condition {
            InterpretationCondition::Matches(token_to_match) => token == token_to_match,
            InterpretationCondition::IsObject => {
                if let Token::Object(_) = token {
                    return true;
                } else {
                    return false;
                }
            }
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum InterpretationCondition {
    Matches(Token),
    IsObject,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ExpressionType {
    Functional,
    Prefix,
    Infix,
    Postfix,
    Outfix(Token),
}

impl Default for ExpressionType {
    fn default() -> Self {
        ExpressionType::Prefix
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum InterpretedType {
    PassThrough,
    Type(Type),
}

impl From<Type> for InterpretedType {
    fn from(value: Type) -> Self {
        Self::Type(value)
    }
}

impl From<String> for InterpretedType {
    fn from(value: String) -> Self {
        Self::from(Type::from(value))
    }
}
impl From<&str> for InterpretedType {
    fn from(value: &str) -> Self {
        Self::from(Type::from(value))
    }
}

#[cfg(test)]
mod test_interpretation {
    use super::*;

    #[test]
    fn test_interpretation_satisfies_condition() {
        let f_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("f".to_string())),
            ExpressionType::Functional,
            0,
            "Function".into(),
        );

        assert!(f_interpretation.satisfies_condition(&None, &Token::Object("f".to_string())));
    }
}
