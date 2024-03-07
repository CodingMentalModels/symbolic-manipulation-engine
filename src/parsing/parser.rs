use crate::parsing::interpretation::{ExpressionType, Interpretation, InterpretationCondition};
use crate::parsing::tokenizer::{Token, Tokenizer};
use crate::symbol::symbol_node::{Symbol, SymbolNode};
use crate::symbol::symbol_type::Type;

use super::interpretation::{ExpressionPrecedence, InterpretedType};
use super::tokenizer::TokenStack;

pub type ParserResult = Result<SymbolNode, ParserError>;

#[derive(Debug, PartialEq, Eq)]
pub struct Parser {
    interpretations: Vec<Interpretation>,
}

impl Parser {
    pub fn new(mut interpretations: Vec<Interpretation>) -> Self {
        interpretations.push(Interpretation::parentheses());
        interpretations.push(Interpretation::comma());
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

    pub fn parse(&self, token_stack: &mut TokenStack) -> Result<SymbolNode, ParserError> {
        token_stack.remove_whitespace();
        let to_return = self.parse_expression(token_stack, 0)?;
        return Ok(to_return.split_delimiters().collapse_delimiters());
    }

    fn parse_expression(
        &self,
        token_stack: &mut TokenStack,
        min_precedence: ExpressionPrecedence,
    ) -> Result<SymbolNode, ParserError> {
        let mut left_expression = if let Some(token) = token_stack.pop() {
            println!(
                "Parsing left_expression: {:?} ({:?}) with min_precedence {:?}",
                token,
                token_stack.to_string(),
                min_precedence
            );
            match self.get_interpretation(&None, &token) {
                Some(interpretation) => match interpretation.get_expression_type() {
                    ExpressionType::Singleton => interpretation.get_symbol_node(&token, vec![])?,
                    ExpressionType::Prefix => {
                        let right_expression = self.parse_expression(
                            token_stack,
                            interpretation.get_expression_precedence() + 1,
                        )?;
                        interpretation.get_symbol_node(&token, vec![right_expression])?
                    }
                    ExpressionType::Outfix(closing_token) => {
                        let contained_expression = self.parse_expression(token_stack, 0)?; // Reset precedence for inner expression
                        token_stack.pop_and_assert_or_error(closing_token)?;
                        interpretation.get_symbol_node(&token, vec![contained_expression])?
                    }
                    t => return Err(ParserError::InvalidLeftExpressionType(token, t)),
                },
                None => return Err(ParserError::NoValidInterpretation(token)),
            }
        } else {
            return Err(ParserError::NoTokensRemainingToInterpret);
        };

        while let Some(next_token) = token_stack.peek() {
            println!(
                "Parsing next token: {:?} ({:?})",
                next_token,
                token_stack.to_string()
            );
            let interpretation =
                match self.get_interpretation(&Some(left_expression.clone()), &next_token) {
                    Some(interpretation) => interpretation,
                    None => {
                        println!(
                            "No valid interpretation of the next token so returning: {:?} ({:?})",
                            next_token,
                            token_stack.to_string()
                        );
                        break;
                    }
                };

            if interpretation.get_expression_precedence() < min_precedence {
                println!(
                    "Expression precedence is below minimum {:?} vs. {:?}",
                    interpretation.get_expression_precedence(),
                    min_precedence
                );
                break;
            }

            token_stack.pop(); // Consume the token

            if interpretation.get_expression_type() == ExpressionType::Infix {
                println!(
                    "Parsing infix: {:?} ({:?})",
                    next_token,
                    token_stack.to_string()
                );
                let right_expression = self.parse_expression(
                    token_stack,
                    interpretation.get_expression_precedence() + 1,
                )?;
                let mut children = vec![left_expression, right_expression];
                left_expression = interpretation.get_symbol_node(&next_token, children)?;
            }
        }

        println!("Returning expression: {:?}", left_expression);
        Ok(left_expression)
    }

    pub fn get_interpretation(
        &self,
        so_far: &Option<SymbolNode>,
        token: &Token,
    ) -> Option<Interpretation> {
        for interpretation in self.interpretations.iter() {
            if interpretation.satisfies_condition(so_far, token) {
                return Some(interpretation.clone());
            }
        }

        return None;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParserError {
    NoValidInterpretation(Token),
    NoTokensRemainingToInterpret,
    ExpectedButFound(Token, Token),
    ExpectedLeftArgument(Token),
    InvalidExpressionType(Token, ExpressionType),
    InvalidLeftExpressionType(Token, ExpressionType),
    InvalidPassThroughInterpretation(Token),
}

#[cfg(test)]
mod test_parser {

    use super::*;

    #[test]
    fn test_parser_parses() {
        let mut tokens = Tokenizer::new_with_tokens(vec!["+".to_string(), "=".to_string()])
            .tokenize("2 + 2 = 4");
        let integer_type = Type::new("Integer".to_string());

        let plus_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("+".to_string())),
            ExpressionType::Infix,
            2,
            "Plus".into(),
        );
        let equals_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("=".to_string())),
            ExpressionType::Infix,
            1,
            "Equals".into(),
        );

        let mut parser = Parser::new(vec![plus_interpretation, equals_interpretation]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new("=".to_string(), "Equals".into(),),
                vec![
                    SymbolNode::new(
                        Symbol::new("+".to_string(), "Plus".into(),),
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
            InterpretationCondition::Matches(Token::Object("=>".to_string())),
            ExpressionType::Infix,
            1,
            "=>".into(),
        );

        let mut parser = Parser::new(vec![implies_interpretation]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new("=>".to_string(), "=>".into(),),
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
            ExpressionType::Prefix,
            1,
            "Function".into(),
        );

        let parser = Parser::new(vec![f_interpretation]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new("f".to_string(), "Function".into(),),
                vec![
                    SymbolNode::leaf_object("x".to_string()),
                    SymbolNode::leaf_object("y".to_string()),
                    SymbolNode::leaf_object("z".to_string()),
                ]
            ))
        )
    }

    #[test]
    fn test_parser_parses_outfix() {
        let mut tokens = Tokenizer::new_with_tokens(vec!["|".to_string()]).tokenize("|x|");

        let pipe_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("|".to_string())),
            ExpressionType::Outfix(Token::Object("|".to_string())),
            1,
            "Absolute Value".into(),
        );

        let parser = Parser::new(vec![pipe_interpretation.clone()]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new("|".to_string(), "Absolute Value".into(),),
                vec![SymbolNode::leaf_object("x".to_string()),]
            ))
        );

        let mut tokens =
            Tokenizer::new_with_tokens(vec!["|".to_string(), "+".to_string()]).tokenize("|a+b|");

        let addition_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("+".to_string())),
            ExpressionType::Infix,
            2,
            "Plus".into(),
        );

        let parser = Parser::new(vec![pipe_interpretation, addition_interpretation.clone()]);
        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new("|".to_string(), "Absolute Value".into(),),
                vec![SymbolNode::new(
                    Symbol::new("+".to_string(), "Plus".into()),
                    vec![
                        SymbolNode::leaf(Symbol::new_object("a".to_string())),
                        SymbolNode::leaf(Symbol::new_object("b".to_string()))
                    ]
                )]
            ))
        );

        let mut tokens =
            Tokenizer::new_with_tokens(vec!["|".to_string(), "+".to_string()]).tokenize("a+(b+c)");

        let parser = Parser::new(vec![Interpretation::parentheses(), addition_interpretation]);
        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new("+".to_string(), "Plus".into(),),
                vec![
                    SymbolNode::leaf(Symbol::new_object("a".to_string())),
                    SymbolNode::new(
                        Symbol::new("+".to_string(), "Plus".into()),
                        vec![
                            SymbolNode::leaf(Symbol::new_object("b".to_string())),
                            SymbolNode::leaf(Symbol::new_object("c".to_string()))
                        ]
                    )
                ]
            ))
        );
    }

    #[test]
    fn test_parser_simple() {
        let mut tokens = Tokenizer::new_with_tokens(vec![]).tokenize("x");
        assert_eq!(tokens.len(), 1);

        let parser = Parser::new(vec![]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(parsed, Ok(SymbolNode::leaf_object("x".to_string())));
    }

    #[test]
    fn test_parser_gets_interpretation() {
        let mut tokens = Tokenizer::new_with_tokens(vec![]).tokenize("x + y + z)");

        let parser = Parser::new(vec![]);
        let token = tokens.pop().unwrap();
        let maybe_interpretation = parser.get_interpretation(&None, &token);
        assert!(maybe_interpretation.is_some());
        let interpretation = maybe_interpretation.unwrap();
        assert!(interpretation.satisfies_condition(&None, &token));
        assert_eq!(
            interpretation.get_expression_type(),
            ExpressionType::Singleton
        );
    }
}
