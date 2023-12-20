use crate::parsing::tokenizer::{Tokenizer, Token};
use crate::parsing::interpretation::{Interpretation, InterpretationCondition, ExpressionType};
use crate::symbol::symbol_type::Type;
use crate::symbol::symbol_node::{SymbolNode, Symbol};

#[derive(Debug, PartialEq, Eq)]
pub struct Parser {
    interpretations: Vec<Interpretation>,
}

impl Parser {

    pub fn new(interpretations: Vec<Interpretation>) -> Self {
        Parser {
            interpretations
        }
    }

    pub fn parse(&mut self, tokens: Vec<Token>) -> Result<SymbolNode, String> {
        unimplemented!()
    }
}


#[cfg(test)]
mod test_parser {
    use std::boxed;


    use super::*;

    #[test]
    fn test_parser_parses() {

        let tokens = Tokenizer::new_with_tokens(vec!["+".to_string(), "=".to_string()]).tokenize("2 + 2 = 4");
        let integer_type = Type::new_from_object("Integer".to_string());
        let boxed_integer = Box::new(integer_type.clone());

        let plus_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("+".to_string())),
            ExpressionType::Infix,
            0,
            Type::Generic(vec![integer_type.clone(), integer_type.clone()], boxed_integer.clone()),
        );
        let equals_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("=".to_string())),
            ExpressionType::Infix,
            1,
            Type::Generic(vec![integer_type.clone(), integer_type.clone()], boxed_integer.clone())
        );

        let mut parser = Parser::new(vec![plus_interpretation, equals_interpretation]);

        let parsed = parser.parse(tokens);

        assert_eq!(
            parsed,
            Ok(
                SymbolNode::new(
                    Symbol::new("=".to_string(), Type::Generic(vec![integer_type.clone(), integer_type.clone()], boxed_integer.clone())
                ),
                    vec![
                        SymbolNode::new(
                            Symbol::new("+".to_string(), Type::Generic(vec![integer_type.clone(), integer_type.clone()], boxed_integer.clone())
                        ),
                        vec![
                            SymbolNode::leaf_object("2".to_string()),
                            SymbolNode::leaf_object("2".to_string())
                            ]
                        ),
                        SymbolNode::leaf_object("4".to_string())
                    ]
                )
            )
        )

    }

}

