use crate::parsing::interpretation::{ExpressionType, Interpretation, InterpretationCondition};
use crate::parsing::tokenizer::{Token, Tokenizer};
use crate::symbol::symbol_node::{Symbol, SymbolNode};
use crate::symbol::symbol_type::Type;

use super::tokenizer::TokenStack;

#[derive(Debug, PartialEq, Eq)]
pub struct Parser {
    interpretations: Vec<Interpretation>,
}

impl Parser {
    pub fn new(interpretations: Vec<Interpretation>) -> Self {
        Parser { interpretations }
    }

    pub fn parse(&self, tokens: &mut TokenStack) -> Result<SymbolNode, String> {
        tokens.remove_whitespace();

        let left_arg = None;

        self.parse_next(left_arg, tokens)
    }

    pub fn parse_next(
        &self,
        so_far: Option<SymbolNode>,
        tokens: &mut TokenStack,
    ) -> Result<SymbolNode, String> {
        let token = tokens
            .pop()
            .ok_or("Tried to parse_next but no tokens remained.".to_string())?;
        self.interpret(so_far, tokens)
    }

    pub fn interpret(
        &self,
        so_far: Option<SymbolNode>,
        tokens: &mut TokenStack,
    ) -> Result<SymbolNode, String> {
        let token = tokens
            .peek()
            .ok_or("Tried to interpret token but no tokens remained.".to_string())?;
        for interpretation in self.interpretations.iter() {
            if interpretation.satisfies_condition(&so_far, &token) {
                return interpretation.interpret(self, so_far, tokens);
            }
        }
        return Err("No valid interpretation.".to_string());
    }
}

#[cfg(test)]
mod test_parser {
    use std::boxed;

    use super::*;

    #[test]
    fn test_parser_parses() {
        let mut tokens = Tokenizer::new_with_tokens(vec!["+".to_string(), "=".to_string()])
            .tokenize("2 + 2 = 4");
        let integer_type = Type::new_from_object("Integer".to_string());
        let boxed_integer = Box::new(integer_type.clone());

        let plus_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("+".to_string())),
            ExpressionType::Infix,
            0,
            Type::Generic(
                vec![integer_type.clone(), integer_type.clone()],
                boxed_integer.clone(),
            ),
        );
        let equals_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("=".to_string())),
            ExpressionType::Infix,
            1,
            Type::Generic(
                vec![integer_type.clone(), integer_type.clone()],
                boxed_integer.clone(),
            ),
        );

        let mut parser = Parser::new(vec![plus_interpretation, equals_interpretation]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new(
                    "=".to_string(),
                    Type::Generic(
                        vec![integer_type.clone(), integer_type.clone()],
                        boxed_integer.clone()
                    )
                ),
                vec![
                    SymbolNode::new(
                        Symbol::new(
                            "+".to_string(),
                            Type::Generic(
                                vec![integer_type.clone(), integer_type.clone()],
                                boxed_integer.clone()
                            )
                        ),
                        vec![
                            SymbolNode::leaf_object("2".to_string()),
                            SymbolNode::leaf_object("2".to_string())
                        ]
                    ),
                    SymbolNode::leaf_object("4".to_string())
                ]
            ))
        )
    }
}
