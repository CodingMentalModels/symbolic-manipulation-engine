use crate::parsing::interpretation::{ExpressionType, Interpretation};
use crate::parsing::tokenizer::{Token, Tokenizer};
use crate::symbol::symbol_node::SymbolNode;

use super::interpretation::{ExpressionPrecedence, InterpretationCondition, InterpretedType};
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
        let to_return = self.parse_inner(token_stack)?;
        if !token_stack.is_empty() {
            return Err(ParserError::DoneParsingButTokensRemain(
                token_stack.as_vector(),
            ));
        }
        return Ok(to_return);
    }

    fn parse_inner(&self, token_stack: &mut TokenStack) -> Result<SymbolNode, ParserError> {
        token_stack.remove_whitespace();
        self.parse_expression(token_stack, 0)
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
                        let mut args = Vec::new();
                        let left = token_stack.pop_or_error()?;
                        let (_left_token, right_token) = self.get_functional_parentheses(left)?;
                        if token_stack.peek() != Some(right_token.clone()) {
                            // There may be no args
                            while let Some(_arg_token) = token_stack.peek() {
                                let arg_expression = self.parse_expression(token_stack, 0)?; // Reset precedence for arg
                                args.push(arg_expression);
                                if token_stack.peek() == Some(right_token.clone()) {
                                    token_stack.pop();
                                    break;
                                }
                                token_stack.pop_and_assert_or_error(Token::Comma)?;
                            }
                        } else {
                            token_stack.pop(); // Remove the right token
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
            let interpretation =
                match self.get_interpretation(&Some(left_expression.clone()), &next_token) {
                    Some(interpretation) => interpretation,
                    None => {
                        break;
                    }
                };

            if interpretation.get_expression_precedence() < min_precedence {
                break;
            }

            token_stack.pop(); // Consume the token

            if interpretation.get_expression_type() == ExpressionType::Infix {
                let right_expression = self.parse_expression(
                    token_stack,
                    interpretation.get_expression_precedence() + 1,
                )?;
                let children = vec![left_expression, right_expression];
                left_expression = interpretation.get_symbol_node(&next_token, children)?;
            }
        }

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

    pub fn get_interpretation_custom_tokens(&self) -> Vec<String> {
        self.interpretations
            .iter()
            .filter_map(|interpretation| match interpretation.get_condition() {
                InterpretationCondition::Matches(t) => Some(t.to_string()),
                _ => None,
            })
            .collect()
    }

    pub fn get_functional_parentheses(
        &self,
        left_token: Token,
    ) -> Result<(Token, Token), ParserError> {
        let interpretation = self
            .interpretations
            .iter()
            .filter(|interpretation| {
                interpretation.get_condition()
                    == InterpretationCondition::Matches(left_token.clone())
            })
            .map(
                |interpretation| match interpretation.get_expression_type() {
                    ExpressionType::Outfix(right_token) => Some((left_token.clone(), right_token)),
                    _ => None,
                },
            )
            .flatten()
            .next();
        match interpretation {
            None => Err(ParserError::NoFunctionalInterpretation(left_token)),
            Some(interpretation) => Ok(interpretation),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParserError {
    NoValidInterpretation(Token),
    NoFunctionalInterpretation(Token),
    NoTokensRemainingToInterpret,
    DoneParsingButTokensRemain(Vec<Token>),
    ExpectedButFound(Token, Token),
    ExpectedLeftArgument(Token),
    InvalidExpressionType(Token, ExpressionType),
    InvalidLeftExpressionType(Token, ExpressionType),
    InvalidPassThroughInterpretation(Token),
    ArbitraryReturningHadNonOneChildren(usize),
}

#[cfg(test)]
mod test_parser {

    use crate::{
        constants::DEFAULT_PRECEDENCE,
        parsing::interpretation::{InterpretationCondition, InterpretedType},
        symbol::{
            symbol_node::{Symbol, SymbolNodeRoot},
            symbol_type::GeneratedTypeCondition,
        },
    };

    use super::*;

    #[test]
    fn test_parser_parses_arbitrary() {
        let custom_tokens = vec!["+".to_string(), "=".to_string()];
        let interpretations = vec![
            Interpretation::arbitrary_functional("Any".into(), 99, "Real".into()),
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::singleton("x", "Real".into()),
            Interpretation::singleton("y", "Real".into()),
        ];
        let parser = Parser::new(interpretations);
        let arbitrary_of = |s: &str| {
            SymbolNode::arbitrary(
                SymbolNode::leaf(Symbol::new(s.to_string(), "Real".into())),
                "Real".into(),
            )
        };
        let expected = SymbolNode::new(
            SymbolNodeRoot::Symbol(Symbol::new_with_same_type_as_value("=")),
            vec![arbitrary_of("x"), arbitrary_of("y")],
        );
        assert_eq!(
            parser
                .parse_from_string(custom_tokens.clone(), "Any(x)=Any(y)")
                .unwrap(),
            expected
        );
    }

    #[test]
    fn test_parser_parses() {
        let mut tokens = Tokenizer::new_with_tokens(vec!["+".to_string(), "=".to_string()])
            .tokenize("2 + 2 = 4");

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

        let parser = Parser::new(vec![plus_interpretation, equals_interpretation]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new_from_symbol(
                Symbol::new("=".to_string(), "Equals".into(),).into(),
                vec![
                    SymbolNode::new_from_symbol(
                        Symbol::new("+".to_string(), "Plus".into(),).into(),
                        vec![SymbolNode::leaf_object("2"), SymbolNode::leaf_object("2")]
                    ),
                    SymbolNode::leaf_object("4")
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

        let parser = Parser::new(vec![implies_interpretation]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new_from_symbol(
                Symbol::new("=>".to_string(), "=>".into(),).into(),
                vec![SymbolNode::leaf_object("p"), SymbolNode::leaf_object("q"),]
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
            Ok(SymbolNode::new_from_symbol(
                Symbol::new("f".to_string(), "Function".into(),).into(),
                vec![]
            ))
        );

        let mut tokens = Tokenizer::new_with_tokens(vec![]).tokenize("f(x, y, z)");

        let parsed = parser.parse(&mut tokens);

        assert_eq!(
            parsed,
            Ok(SymbolNode::new_from_symbol(
                Symbol::new("f".to_string(), "Function".into(),).into(),
                vec![
                    SymbolNode::leaf_object("x"),
                    SymbolNode::leaf_object("y"),
                    SymbolNode::leaf_object("z"),
                ]
            ))
        );
    }

    #[test]
    fn test_parser_gets_functional_parentheses() {
        let parser = Parser::new(vec![]);
        assert_eq!(
            parser.get_functional_parentheses(Token::LeftParen).unwrap(),
            (Token::LeftParen, Token::RightParen)
        );

        let parser = Parser::new(vec![
            Interpretation::parentheses_like(
                Token::Object("{".to_string()),
                Token::Object("}".to_string()),
            ),
            Interpretation::parentheses_like("|".into(), "|".into()),
        ]);
        assert_eq!(
            parser.get_functional_parentheses(Token::LeftParen).unwrap(),
            (Token::LeftParen, Token::RightParen)
        );
        assert_eq!(
            parser.get_functional_parentheses("{".into()).unwrap(),
            (
                Token::Object("{".to_string()),
                Token::Object("}".to_string())
            )
        );
        assert_eq!(
            parser.get_functional_parentheses("|".into()).unwrap(),
            (
                Token::Object("|".to_string()),
                Token::Object("|".to_string())
            )
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

        let mut interpretations = operators_interpretations.clone();
        interpretations.push(Interpretation::function(
            Token::Object("inv".to_string()),
            7,
        ));

        let parser = Parser::new(interpretations.clone());

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
            SymbolNode::new_from_symbol(
                Symbol::new("^".to_string(), "Operator".into()).into(),
                vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("2"),]
            )
        );

        let expected = SymbolNode::new_from_symbol(
            Symbol::new("=".to_string(), "Operator".into()).into(),
            vec![
                SymbolNode::new_from_symbol(
                    Symbol::new("+".to_string(), "Operator".into()).into(),
                    vec![a_squared, b_squared],
                ),
                c_squared,
            ],
        );

        assert_eq!(pythagorean_theorem, Ok(expected));

        let inverse_from = parser
            .parse_from_string(vec!["*".to_string(), "inv".to_string()], "g*inv(g)")
            .unwrap();

        let expected = SymbolNode::new_from_symbol(
            Symbol::new("*".to_string(), "Operator".into()),
            vec![
                SymbolNode::leaf_object("g"),
                SymbolNode::new_from_symbol(
                    Symbol::new("inv".to_string(), "inv".into()),
                    vec![SymbolNode::leaf_object("g")],
                ),
            ],
        );

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
            Ok(SymbolNode::new_from_symbol(
                Symbol::new("omega".to_string(), "Function".into(),),
                vec![SymbolNode::new_from_symbol(
                    Symbol::new("f".to_string(), "Function".into()),
                    vec![SymbolNode::new_from_symbol(
                        Symbol::new("g".to_string(), "Function".into(),),
                        vec![SymbolNode::new_from_symbol(
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
            Ok(SymbolNode::new_from_symbol(
                Symbol::new("f".to_string(), "Function".into(),),
                vec![
                    SymbolNode::new_from_symbol(
                        Symbol::new("g".to_string(), "Function".into()),
                        vec![SymbolNode::leaf_object("x"),]
                    ),
                    SymbolNode::new_from_symbol(
                        Symbol::new("h".to_string(), "Function".into()),
                        vec![SymbolNode::leaf_object("y"),]
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
            Ok(SymbolNode::new_from_symbol(
                Symbol::new("omega".to_string(), "Function".into()),
                vec![SymbolNode::new_from_symbol(
                    Symbol::new("f".to_string(), "Function".into(),),
                    vec![
                        SymbolNode::new_from_symbol(
                            Symbol::new("g".to_string(), "Function".into()),
                            vec![
                                SymbolNode::leaf_object("w"),
                                SymbolNode::leaf_object("x"),
                                SymbolNode::leaf_object("y"),
                            ]
                        ),
                        SymbolNode::new_from_symbol(
                            Symbol::new("h".to_string(), "Function".into()),
                            vec![SymbolNode::leaf_object("z"),]
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
            Ok(SymbolNode::new_from_symbol(
                Symbol::new("|".to_string(), "Absolute Value".into(),),
                vec![SymbolNode::leaf_object("x"),]
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
            Ok(SymbolNode::new_from_symbol(
                Symbol::new("|".to_string(), "Absolute Value".into(),),
                vec![SymbolNode::new_from_symbol(
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
            Ok(SymbolNode::new_from_symbol(
                Symbol::new("+".to_string(), "Plus".into(),),
                vec![
                    SymbolNode::leaf(Symbol::new_object("a".to_string())),
                    SymbolNode::new_from_symbol(
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

        assert_eq!(parsed, Ok(SymbolNode::leaf_object("x")));
    }

    #[test]
    fn test_parser_latex() {
        let mut tokens = Tokenizer::new_with_tokens(vec![]).tokenize("\\alpha");
        assert_eq!(tokens.len(), 1);

        let parser = Parser::new(vec![]);

        let parsed = parser.parse(&mut tokens);

        assert_eq!(parsed, Ok(SymbolNode::leaf_object("\\alpha")));

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

        let mut interpretations = operators_interpretations.clone();
        interpretations.push(Interpretation::function(
            Token::Object("inv".to_string()),
            7,
        ));

        let parser = Parser::new(interpretations.clone());

        let pythagorean_theorem =
            parser.parse_from_string(operator_names.clone(), "\\alpha^2 + \\beta^2 = \\gamma^2");

        let alpha_squared = parser
            .parse_from_string(operator_names.clone(), "\\alpha^2")
            .unwrap();
        let beta_squared = parser
            .parse_from_string(operator_names.clone(), "\\beta^2")
            .unwrap();
        let gamma_squared = parser
            .parse_from_string(operator_names.clone(), "\\gamma^2")
            .unwrap();

        assert_eq!(
            alpha_squared.clone(),
            SymbolNode::new_from_symbol(
                Symbol::new("^".to_string(), "Operator".into()),
                vec![
                    SymbolNode::leaf_object("\\alpha"),
                    SymbolNode::leaf_object("2"),
                ]
            )
        );

        let expected = SymbolNode::new_from_symbol(
            Symbol::new("=".to_string(), "Operator".into()),
            vec![
                SymbolNode::new_from_symbol(
                    Symbol::new("+".to_string(), "Operator".into()),
                    vec![alpha_squared, beta_squared],
                ),
                gamma_squared,
            ],
        );

        assert_eq!(pythagorean_theorem, Ok(expected));

        // TODO The below doesn't pass because f{x}{y} isn't valid syntax -- parser wants f{x, y}.
        // We need to work out from a design perspective whether we want this to be configurable or
        // always allowed

        //        let frac_alpha_beta = parser
        //            .parse_from_string(operator_names.clone(), "\\frac{\\alpha}{\\beta}")
        //            .unwrap();
        //        assert_eq!(frac_alpha_beta.get_root_name(), "\\frac".to_string());
        //        assert_eq!(frac_alpha_beta.get_n_children(), 2);
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
        let plus = Interpretation::infix_operator("+".into(), 1, "+".into());
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

        let expected = SymbolNode::new_from_symbol(
            Symbol::new("+".to_string(), "+".into()),
            vec![SymbolNode::singleton("2"), SymbolNode::singleton("2")],
        );
        assert_eq!(two_plus_two, expected);
    }
}
