use serde::{Deserialize, Serialize};
use ts_rs::TS;

use regex::Regex;

use crate::{
    constants::*,
    parsing::parser::ParserError,
    symbol::{
        symbol_node::{Symbol, SymbolNode},
        symbol_type::{GeneratedTypeCondition, Type},
    },
};

use super::tokenizer::Token;

pub type ExpressionPrecedence = u8;

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DisplayInterpretation {
    condition: InterpretationCondition,
    expression_type: ExpressionType,
    expression_precedence: ExpressionPrecedence,
    output_type: InterpretedType,
}

impl From<&Interpretation> for DisplayInterpretation {
    fn from(value: &Interpretation) -> Self {
        Self {
            condition: value.condition.clone(),
            expression_type: value.expression_type.clone(),
            expression_precedence: value.expression_precidence,
            output_type: value.output_type.clone(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
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
        Self::parentheses_like(Token::LeftParen, Token::RightParen)
    }

    pub fn parentheses_like(left: Token, right: Token) -> Self {
        Interpretation::new(
            InterpretationCondition::Matches(left),
            ExpressionType::Outfix(right),
            PARENTHESIS_PRECEDENCE,
            InterpretedType::PassThrough,
        )
    }
    pub fn comma() -> Self {
        Interpretation::new(
            InterpretationCondition::Matches(Token::Comma),
            ExpressionType::Infix,
            COMMA_PRECEDENCE,
            InterpretedType::Delimiter,
        )
    }

    pub fn any_object() -> Self {
        Interpretation::new(
            InterpretationCondition::IsObject,
            ExpressionType::Singleton,
            DEFAULT_PRECEDENCE,
            Type::Object.into(),
        )
    }

    pub fn singleton(name: &str, t: Type) -> Self {
        Interpretation::new(
            InterpretationCondition::Matches(name.into()),
            ExpressionType::Singleton,
            DEFAULT_PRECEDENCE,
            t.into(),
        )
    }

    pub fn generated_type(condition: GeneratedTypeCondition) -> Self {
        Self::new(
            condition.into(),
            ExpressionType::Singleton,
            DEFAULT_PRECEDENCE,
            InterpretedType::SameAsValue,
        )
    }

    pub fn prefix_operator(token: Token, precedence: u8, evaluates_to_type: Type) -> Self {
        Interpretation::new(
            InterpretationCondition::Matches(token.clone()),
            ExpressionType::Prefix,
            precedence,
            evaluates_to_type.into(),
        )
    }

    pub fn postfix_operator(token: Token, precedence: u8, evaluates_to_type: Type) -> Self {
        Interpretation::new(
            InterpretationCondition::Matches(token.clone()),
            ExpressionType::Postfix,
            precedence,
            evaluates_to_type.into(),
        )
    }

    pub fn infix_operator(token: Token, precedence: u8, evaluates_to_type: Type) -> Self {
        Interpretation::new(
            InterpretationCondition::Matches(token.clone()),
            ExpressionType::Infix,
            precedence,
            evaluates_to_type.into(),
        )
    }

    pub fn outfix_operator(
        tokens: (Token, Token),
        precedence: u8,
        evaluates_to_type: Type,
    ) -> Self {
        Interpretation::new(
            InterpretationCondition::Matches(tokens.0.clone()),
            ExpressionType::Outfix(tokens.1.clone()),
            precedence,
            evaluates_to_type.into(),
        )
    }

    pub fn function(token: Token, precedence: u8) -> Self {
        Interpretation::new(
            InterpretationCondition::Matches(token.clone()),
            ExpressionType::Functional,
            precedence,
            token.to_string().into(),
        )
    }

    pub fn arbitrary_functional(token: Token, precedence: u8, evaluates_to_type: Type) -> Self {
        Interpretation::new(
            InterpretationCondition::Matches(token.clone()),
            ExpressionType::Functional,
            precedence,
            InterpretedType::ArbitraryReturning(evaluates_to_type),
        )
    }
    pub fn get_condition(&self) -> InterpretationCondition {
        self.condition.clone()
    }

    pub fn get_expression_type(&self) -> ExpressionType {
        self.expression_type.clone()
    }

    pub fn get_expression_precedence(&self) -> ExpressionPrecedence {
        self.expression_precidence
    }

    pub fn get_output_type(&self) -> InterpretedType {
        self.output_type.clone()
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
            InterpretedType::Delimiter => Ok(SymbolNode::new(
                Symbol::new(token.to_string(), Type::Delimiter).into(),
                children,
            )),
            InterpretedType::SameAsValue => Ok(SymbolNode::new(
                Symbol::new(token.to_string(), token.to_string().into()).into(),
                children,
            )),
            InterpretedType::Type(t) => Ok(SymbolNode::new(
                Symbol::new(token.to_string(), t.clone()).into(),
                children,
            )),
            InterpretedType::ArbitraryReturning(t) => {
                if children
                    .iter()
                    .any(|child| child.contains_arbitrary_nodes())
                {
                    Err(ParserError::NestedArbitraryNodes)
                } else if children.len() != 1 {
                    Err(ParserError::ArbitraryReturningHadNonOneChildren(
                        children.len(),
                    ))
                } else {
                    let child = children[0].clone();
                    Ok(SymbolNode::arbitrary(
                        Symbol::new(token.to_string(), t.clone()),
                        child,
                    ))
                }
            }
        }
    }

    pub fn could_produce(&self, statement: &SymbolNode) -> bool {
        self.condition.could_produce(statement)
            && self.expression_type.could_produce(statement)
            && self.output_type.could_produce(statement)
    }

    pub fn update_match_string(
        &mut self,
        new_match_name: &str,
    ) -> Result<String, InterpretationError> {
        match &self.condition {
            InterpretationCondition::Matches(_old_match) => {
                self.condition = InterpretationCondition::Matches(new_match_name.into());
                Ok(new_match_name.to_string())
            }
            condition => Err(InterpretationError::NonMatchCondition(condition.clone())),
        }
    }

    pub fn satisfies_condition(&self, so_far: &Option<SymbolNode>, token: &Token) -> bool {
        let is_ok_expression_type = match self.expression_type {
            ExpressionType::Singleton
            | ExpressionType::Prefix
            | ExpressionType::Outfix(_)
            | ExpressionType::Functional => so_far.is_none(),
            ExpressionType::Infix | ExpressionType::Postfix => so_far.is_some(),
        };
        if !is_ok_expression_type {
            return false;
        }
        match &self.condition {
            InterpretationCondition::Matches(token_to_match) => token == token_to_match,
            InterpretationCondition::SatisfiesRegex(pattern) => {
                if let Ok(regex) = Regex::new(pattern) {
                    if let Token::Object(token_string) = token {
                        regex.is_match(&token_string)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            InterpretationCondition::IsObject => {
                if let Token::Object(_) = token {
                    return true;
                } else {
                    return false;
                }
            }
            InterpretationCondition::IsInteger => {
                return Self::is_integer(token);
            }
            InterpretationCondition::IsNumeric => {
                return Self::is_numeric(token);
            }
        }
    }

    fn is_integer(token: &Token) -> bool {
        if let Token::Object(n) = token {
            // TODO: This will fail on big enough numbers
            return n.parse::<i64>().is_ok();
        } else {
            return false;
        }
    }

    fn is_numeric(token: &Token) -> bool {
        if let Token::Object(n) = token {
            // TODO: This will fail on big enough numbers
            return n.parse::<f64>().is_ok();
        } else {
            return false;
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, TS)]
pub enum InterpretationError {
    NonMatchCondition(InterpretationCondition),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", content = "token", rename_all = "camelCase")]
#[ts(export)]
pub enum InterpretationCondition {
    Matches(Token),
    IsObject,
    IsInteger,
    IsNumeric,
    SatisfiesRegex(String),
}

impl From<GeneratedTypeCondition> for InterpretationCondition {
    fn from(value: GeneratedTypeCondition) -> Self {
        match value {
            GeneratedTypeCondition::IsInteger => Self::IsInteger,
            GeneratedTypeCondition::IsNumeric => Self::IsNumeric,
            GeneratedTypeCondition::SatisfiesRegex(s) => Self::SatisfiesRegex(s),
        }
    }
}

impl InterpretationCondition {
    fn could_produce(&self, statement: &SymbolNode) -> bool {
        match self {
            Self::IsObject => true,
            Self::IsNumeric => {
                Interpretation::is_numeric(&Token::Object(statement.get_root_as_string()))
            }
            Self::IsInteger => {
                Interpretation::is_integer(&Token::Object(statement.get_root_as_string()))
            }
            Self::SatisfiesRegex(pattern) => {
                if let Ok(regex) = Regex::new(pattern) {
                    regex.is_match(&statement.get_root().get_name())
                } else {
                    false
                }
            }
            Self::Matches(token) => {
                if let Token::Object(s) = token {
                    return s == &statement.get_root_as_string();
                } else {
                    return false;
                }
            }
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", content = "token", rename_all = "camelCase")]
#[ts(export)]
pub enum ExpressionType {
    Singleton,
    Prefix,
    Infix,
    Postfix,
    Outfix(Token),
    Functional,
}

impl Default for ExpressionType {
    fn default() -> Self {
        ExpressionType::Singleton
    }
}

impl ExpressionType {
    fn could_produce(&self, statement: &SymbolNode) -> bool {
        match self {
            Self::Singleton => !statement.has_children(),
            Self::Prefix | Self::Postfix | Self::Outfix(_) => statement.get_n_children() == 1,
            Self::Infix => statement.get_n_children() == 2,
            Self::Functional => statement.has_children(),
        }
    }

    pub fn try_parse(s: &str) -> Result<Self, ()> {
        // TODO Support Outfix
        match s.to_lowercase().as_ref() {
            "singleton" => Ok(Self::Singleton),
            "prefix" => Ok(Self::Prefix),
            "infix" => Ok(Self::Infix),
            "postfix" => Ok(Self::Postfix),
            "functional" => Ok(Self::Functional),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", content = "type", rename_all = "camelCase")]
#[ts(export)]
pub enum InterpretedType {
    PassThrough,
    Delimiter,
    SameAsValue,
    Type(Type),
    ArbitraryReturning(Type),
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

impl InterpretedType {
    fn could_produce(&self, statement: &SymbolNode) -> bool {
        match self {
            Self::PassThrough => false,
            Self::Delimiter => false,
            Self::SameAsValue => statement.has_same_value_as_type(),
            Self::Type(t) => t == &statement.get_evaluates_to_type(),
            Self::ArbitraryReturning(t) => t == &statement.get_evaluates_to_type(),
        }
    }
}

#[cfg(test)]
mod test_interpretation {
    use super::*;

    #[test]
    fn test_interpretation_satisfies_condition() {
        let f_interpretation = Interpretation::new(
            InterpretationCondition::Matches(Token::Object("f".to_string())),
            ExpressionType::Singleton,
            0,
            "Function".into(),
        );

        assert!(f_interpretation.satisfies_condition(&None, &Token::Object("f".to_string())));
    }
}
