use std::{collections::VecDeque, unimplemented};

use crate::{
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

pub type ExpressionPrecidence = u8;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Interpretation {
    condition: InterpretationCondition,
    expression_type: ExpressionType,
    expression_precidence: ExpressionPrecidence,
    output_type: Type,
}

impl Interpretation {
    pub fn new(
        condition: InterpretationCondition,
        expression_type: ExpressionType,
        expression_precidence: ExpressionPrecidence,
        output_type: Type,
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
            ExpressionType::Outfix,
            PARENTHESES_PRECEDENCE,
            Type::Ignore_me_im_a_paren,
        )
    }

    pub fn any_object() -> Self {
        Interpretation::new(
            InterpretationCondition::IsObject,
            ExpressionType::Singleton,
            1,
            Type::Object,
        )
    }

    pub fn get_expression_type(&self) -> ExpressionType {
        self.expression_type
    }

    pub fn get_expression_precidence(&self) -> ExpressionPrecidence {
        self.expression_precidence
    }

    pub fn satisfies_condition(&self, so_far: &Option<SymbolNode>, token: &Token) -> bool {
        let is_ok_expression_type = match self.expression_type {
            ExpressionType::Singleton
            | ExpressionType::Functional
            | ExpressionType::Prefix
            | ExpressionType::Outfix => so_far.is_none(),
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ExpressionType {
    Singleton,
    Functional,
    Prefix,
    Infix,
    Postfix,
    Outfix,
}

impl Default for ExpressionType {
    fn default() -> Self {
        ExpressionType::Singleton
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
