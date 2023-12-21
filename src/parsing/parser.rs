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

        let mut left_arg = Some(self.parse_next(None, tokens)?);

        while let Some(_token) = tokens.peek() {
            match self.parse_next(left_arg.clone(), tokens) {
                Ok(node) => {
                    left_arg = Some(node);
                }
                Err(_e) => {
                    return Ok(left_arg.unwrap());
                }
            }
        }
        return Ok(left_arg.unwrap());
    }

    pub fn parse_next(
        &self,
        so_far: Option<SymbolNode>,
        tokens: &mut TokenStack,
    ) -> Result<SymbolNode, String> {
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
        let object_interpretation = Interpretation::new(
            InterpretationCondition::IsObject,
            ExpressionType::Singleton,
            0,
            Type::Object,
        );
        let mut interpretations = self.interpretations.clone();
        interpretations.push(object_interpretation);
        for interpretation in interpretations.iter() {
            if interpretation.satisfies_condition(&so_far, &token) {
                return interpretation.interpret(self, so_far, tokens);
            }
        }

        return Err(format!("No valid interpretation for {:?}", token).to_string());
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

    #[test]
    fn test_parser_parses_functional() {
        let mut tokens = Tokenizer::new_with_tokens(vec![]).tokenize("f(x, y, z)");

        let f_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("f".to_string())),
            ExpressionType::Functional,
            0,
            Type::new_generic_function_with_arguments(3),
        );
        let parser = Parser::new(vec![f_interpretation]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new(
                    "f".to_string(),
                    Type::new_generic_function_with_arguments(3)
                ),
                vec![
                    SymbolNode::leaf_object("x".to_string()),
                    SymbolNode::leaf_object("y".to_string()),
                    SymbolNode::leaf_object("z".to_string()),
                ]
            ))
        )
    }

    #[test]
    fn test_parser_simple() {
        let mut tokens = Tokenizer::new_with_tokens(vec![]).tokenize("x");
        assert_eq!(tokens.len(), 1);

        let parser = Parser::new(vec![]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(parsed, Ok(SymbolNode::leaf_object("x".to_string())));
    }
}
