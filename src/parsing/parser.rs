use crate::parsing::interpretation::{ExpressionType, Interpretation, InterpretationCondition};
use crate::parsing::tokenizer::{Token, Tokenizer};
use crate::symbol::symbol_node::{Symbol, SymbolNode};
use crate::symbol::symbol_type::Type;

use super::tokenizer::TokenStack;

pub type ParserResult = Result<SymbolNode, ParserError>;

#[derive(Debug, PartialEq, Eq)]
pub struct Parser {
    interpretations: Vec<Interpretation>,
}

impl Parser {
    pub fn new(mut interpretations: Vec<Interpretation>) -> Self {
        interpretations.push(Interpretation::any_object());
        Self::new_raw(interpretations)
    }

    pub fn new_raw(interpretations: Vec<Interpretation>) -> Self {
        Parser { interpretations }
    }

    pub fn combine(self, other: Self) -> Self {
        let mut new_interpretations = self.interpretations;
        new_interpretations.extend(other.interpretations);
        Self::new_raw(new_interpretations)
    }

    pub fn parse(&self, tokens: &mut TokenStack) -> ParserResult {
        tokens.remove_whitespace();

        let mut left_arg = Some(self.parse_next(None, tokens)?);

        while let Some(_token) = tokens.peek() {
            match self.parse_next(left_arg.clone(), tokens) {
                Ok(node) => {
                    left_arg = Some(node);
                }
                Err(ParserError::NoTokensRemainingToInterpret)
                | Err(ParserError::NoValidInterpretation(_)) => {
                    return Ok(left_arg.unwrap());
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
        return Ok(left_arg.unwrap());
    }

    pub fn parse_next(&self, so_far: Option<SymbolNode>, tokens: &mut TokenStack) -> ParserResult {
        self.interpret(so_far, tokens)
    }

    pub fn interpret(&self, so_far: Option<SymbolNode>, tokens: &mut TokenStack) -> ParserResult {
        let token = tokens
            .peek()
            .ok_or(ParserError::NoTokensRemainingToInterpret)?;
        for interpretation in self.interpretations.iter() {
            if interpretation.satisfies_condition(&so_far, &token) {
                return interpretation.interpret(self, so_far, tokens);
            }
        }

        return Err(ParserError::NoValidInterpretation(token));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParserError {
    NoValidInterpretation(Token),
    NoTokensRemainingToInterpret,
    ExpectedButFound(Token, Token),
    ExpectedLeftArgument,
}

#[cfg(test)]
mod test_parser {

    use super::*;

    #[test]
    fn test_parser_parses() {
        let mut tokens = Tokenizer::new_with_tokens(vec!["+".to_string(), "=".to_string()])
            .tokenize("2 + 2 = 4");
        let integer_type = Type::new_from_object("Integer".to_string());
        let boxed_integer = Box::new(integer_type.clone());

        let plus_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Custom("+".to_string())),
            ExpressionType::Infix,
            0,
            Type::Generic(
                vec![integer_type.clone(), integer_type.clone()],
                boxed_integer.clone(),
            ),
        );
        let equals_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Custom("=".to_string())),
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
    fn test_parser_parses_p_implies_q() {
        let mut tokens = Tokenizer::new_with_tokens(vec!["=>".to_string()]).tokenize("p=>q");

        let implies_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Custom("=>".to_string())),
            ExpressionType::Infix,
            0,
            Type::new_generic_function_with_arguments(2),
        );

        let mut parser = Parser::new(vec![implies_interpretation]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new(
                    "=>".to_string(),
                    Type::new_generic_function_with_arguments(2)
                ),
                vec![
                    SymbolNode::leaf_object("p".to_string()),
                    SymbolNode::leaf_object("q".to_string()),
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
