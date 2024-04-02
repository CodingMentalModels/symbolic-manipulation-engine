use crate::parsing::interpretation::{ExpressionType, Interpretation};
use crate::parsing::tokenizer::{Token, Tokenizer};
use crate::symbol::symbol_node::SymbolNode;
use crate::symbol::symbol_type::GeneratedType;

use super::interpretation::ExpressionPrecedence;
use super::tokenizer::TokenStack;

#[derive(Debug, PartialEq, Eq)]
pub struct Parser {
    interpretations: Vec<Interpretation>,
}

impl Parser {
    pub fn new(mut interpretations: Vec<Interpretation>) -> Self {
        interpretations.push(Interpretation::parentheses());
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
        return Ok(to_return);
    }

    pub fn parse_from_string(
        &self,
        custom_tokens: Vec<String>,
        s: &str,
    ) -> Result<SymbolNode, ParserError> {
        let mut tokenizer = Tokenizer::new_with_tokens(custom_tokens);
        let mut token_stack = tokenizer.tokenize(s);
        self.parse(&mut token_stack)
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
                    ExpressionType::Functional => {
                        println!("Parsing functional: {:?} ({:?})", token, token_stack);
                        let mut args = Vec::new();
                        token_stack.pop_and_assert_or_error(Token::LeftParen)?;
                        if token_stack.peek() != Some(Token::RightParen) {
                            // There may be no args
                            while let Some(_arg_token) = token_stack.peek() {
                                let arg_expression = self.parse_expression(token_stack, 0)?; // Reset precedence for arg
                                args.push(arg_expression);
                                if token_stack.peek() == Some(Token::RightParen) {
                                    token_stack.pop();
                                    break;
                                }
                                token_stack.pop_and_assert_or_error(Token::Comma)?;
                            }
                        }
                        interpretation.get_symbol_node(&token, args)?
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

    use crate::{
        constants::DEFAULT_PRECEDENCE,
        parsing::interpretation::{InterpretationCondition, InterpretedType},
        symbol::{
            symbol_node::Symbol,
            symbol_type::{GeneratedTypeCondition, Type},
        },
    };

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
        );
    }

    #[test]
    fn test_parser_parses_functional() {
        let f_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("f".to_string())),
            ExpressionType::Functional,
            1,
            "Function".into(),
        );

        let parser = Parser::new(vec![f_interpretation]);

        let mut tokens = Tokenizer::new_with_tokens(vec![]).tokenize("f()");

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new("f".to_string(), "Function".into(),),
                vec![]
            ))
        );

        let mut tokens = Tokenizer::new_with_tokens(vec![]).tokenize("f(x, y, z)");

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
        );
    }

    #[test]
    fn test_parser_parses_from_string() {
        let operators = vec![("^", 6), ("*", 5), ("/", 5), ("+", 4), ("-", 4), ("=", 3)];
        let operator_names: Vec<_> = operators
            .clone()
            .into_iter()
            .map(|(name, _precedence)| name.to_string())
            .collect();

        let operators_interpretations: Vec<_> = operators
            .clone()
            .into_iter()
            .map(|(name, precedence)| {
                Interpretation::new(
                    InterpretationCondition::Matches(Token::Object(name.to_string())),
                    ExpressionType::Infix,
                    precedence,
                    "Operator".into(),
                )
            })
            .collect();

        let parser = Parser::new(operators_interpretations.clone());

        let pythagorean_theorem =
            parser.parse_from_string(operator_names.clone(), "a^2 + b^2 = c^2");

        let a_squared = parser
            .parse_from_string(operator_names.clone(), "a^2")
            .unwrap();
        let b_squared = parser
            .parse_from_string(operator_names.clone(), "b^2")
            .unwrap();
        let c_squared = parser
            .parse_from_string(operator_names.clone(), "c^2")
            .unwrap();

        assert_eq!(
            a_squared.clone(),
            SymbolNode::new(
                Symbol::new("^".to_string(), "Operator".into()),
                vec![
                    SymbolNode::leaf_object("a".to_string()),
                    SymbolNode::leaf_object("2".to_string()),
                ]
            )
        );

        let expected = SymbolNode::new(
            Symbol::new("=".to_string(), "Operator".into()),
            vec![
                SymbolNode::new(
                    Symbol::new("+".to_string(), "Operator".into()),
                    vec![a_squared, b_squared],
                ),
                c_squared,
            ],
        );

        assert_eq!(pythagorean_theorem, Ok(expected));

        let expected = SymbolNode::new(
            Symbol::new("*".to_string(), "*".into()),
            vec![
                SymbolNode::leaf_object("g".to_string()),
                SymbolNode::new(
                    Symbol::new("inv".to_string(), "Operator".into()),
                    vec![SymbolNode::leaf_object("g".to_string())],
                ),
            ],
        );
        let inverse_from = parser
            .parse_from_string(vec!["*".to_string(), "inv".to_string()], "g*inv(g)")
            .unwrap();
        assert_eq!(inverse_from, expected);
    }

    #[test]
    fn test_parser_parses_nested() {
        let functions = vec!["omega", "f", "g", "h"];

        let function_interpretations: Vec<_> = functions
            .clone()
            .into_iter()
            .map(|name| {
                Interpretation::new(
                    InterpretationCondition::Matches(Token::Object(name.to_string())),
                    ExpressionType::Functional,
                    1,
                    "Function".into(),
                )
            })
            .collect();

        let parser = Parser::new(function_interpretations.clone());

        let mut tokens = Tokenizer::new_with_tokens(
            functions
                .clone()
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        )
        .tokenize("omega(f(g(h())))");

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new("omega".to_string(), "Function".into(),),
                vec![SymbolNode::new(
                    Symbol::new("f".to_string(), "Function".into()),
                    vec![SymbolNode::new(
                        Symbol::new("g".to_string(), "Function".into(),),
                        vec![SymbolNode::new(
                            Symbol::new("h".to_string(), "Function".into()),
                            vec![]
                        ),]
                    )]
                ),]
            ))
        );

        let mut tokens = Tokenizer::new_with_tokens(
            functions
                .clone()
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        )
        .tokenize("f(g(x), h(y))");

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new("f".to_string(), "Function".into(),),
                vec![
                    SymbolNode::new(
                        Symbol::new("g".to_string(), "Function".into()),
                        vec![SymbolNode::leaf_object("x".to_string()),]
                    ),
                    SymbolNode::new(
                        Symbol::new("h".to_string(), "Function".into()),
                        vec![SymbolNode::leaf_object("y".to_string()),]
                    ),
                ]
            ))
        );

        let mut tokens = Tokenizer::new_with_tokens(
            functions
                .clone()
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
        )
        .tokenize("omega(f(g(w, x, y), h(z)))");

        let parser = Parser::new(function_interpretations);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new(
                Symbol::new("omega".to_string(), "Function".into()),
                vec![SymbolNode::new(
                    Symbol::new("f".to_string(), "Function".into(),),
                    vec![
                        SymbolNode::new(
                            Symbol::new("g".to_string(), "Function".into()),
                            vec![
                                SymbolNode::leaf_object("w".to_string()),
                                SymbolNode::leaf_object("x".to_string()),
                                SymbolNode::leaf_object("y".to_string()),
                            ]
                        ),
                        SymbolNode::new(
                            Symbol::new("h".to_string(), "Function".into()),
                            vec![SymbolNode::leaf_object("z".to_string()),]
                        ),
                    ]
                )]
            ))
        );
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

    #[test]
    fn test_parser_parses_generated_type() {
        let plus = Interpretation::infix_operator("+".into(), 1);
        let integer_condition = GeneratedTypeCondition::IsInteger;
        let integer = Interpretation::new(
            integer_condition.clone().into(),
            ExpressionType::Singleton,
            DEFAULT_PRECEDENCE,
            InterpretedType::SameAsValue,
        );

        let also_integer = Interpretation::generated_type(integer_condition);
        assert_eq!(integer, also_integer);

        let parser = Parser::new(vec![plus, integer]);
        let two_plus_two = parser
            .parse_from_string(vec!["+".to_string()], "2+2")
            .unwrap();

        let expected = SymbolNode::new(
            Symbol::new("+".to_string(), "+".into()),
            vec![
                SymbolNode::singleton("2".to_string()),
                SymbolNode::singleton("2".to_string()),
            ],
        );
        assert_eq!(two_plus_two, expected);
    }
}
