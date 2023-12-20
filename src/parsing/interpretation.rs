use std::{collections::VecDeque, unimplemented};

use crate::symbol::{
    symbol_node::{Symbol, SymbolNode},
    symbol_type::Type,
};

use super::{
    parser::Parser,
    tokenizer::{Token, TokenStack},
};

type ExpressionPrecidence = u8;

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

    pub fn satisfies_condition(&self, so_far: &Option<SymbolNode>, token: &Token) -> bool {
        let is_ok_expression_type = match self.expression_type {
            ExpressionType::Functional | ExpressionType::Prefix | ExpressionType::Outfix => {
                so_far.is_none()
            }
            ExpressionType::Infix | ExpressionType::Postfix => so_far.is_some(),
        };
        if !is_ok_expression_type {
            return false;
        }
        match &self.condition {
            InterpretationCondition::Matches(token_to_match) => token == token_to_match,
        }
    }

    pub fn interpret(
        &self,
        parser: &Parser,
        so_far: Option<SymbolNode>,
        tokens: &mut TokenStack,
    ) -> Result<SymbolNode, String> {
        match self.expression_type {
            ExpressionType::Functional => {
                let function_token = tokens.pop().ok_or(
                    "Ran out of tokens while interpreting functional expression.".to_string(),
                )?;
                let function = Symbol::new(function_token.to_string(), self.output_type.clone());
                tokens.pop_and_assert_or_error(
                    Token::LeftParen,
                    "Expected left parenthesis while interpreting functional expression."
                        .to_string(),
                )?;
                let mut children = Vec::new();
                while let Some(token) = tokens.pop() {
                    children.push(parser.parse(tokens)?);
                    let delimiter = tokens.pop().ok_or(
                        "Missing delimiter while interpreting functional expression.".to_string(),
                    )?;
                    if delimiter == Token::RightParen {
                        break;
                    }
                    if delimiter != Token::Comma {
                        return Err(
                            "Invalid delimiter while interpreting functional expression."
                                .to_string(),
                        );
                    }
                }
                let to_return = SymbolNode::new(function, children);
                return Ok(to_return);
            }
            _ => {
                unimplemented!()
            }
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum InterpretationCondition {
    Matches(Token),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ExpressionType {
    Functional,
    Prefix,
    Infix,
    Postfix,
    Outfix,
}

impl Default for ExpressionType {
    fn default() -> Self {
        ExpressionType::Functional
    }
}

#[cfg(test)]
mod test_interpretation {
    use super::*;

    #[test]
    fn test_() {}
}
