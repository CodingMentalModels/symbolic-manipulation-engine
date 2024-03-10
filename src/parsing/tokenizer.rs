use std::collections::VecDeque;

use super::parser::ParserError;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Tokenizer {
    custom_tokens: Vec<String>,
}

impl Tokenizer {
    pub fn new_with_tokens(custom_tokens: Vec<String>) -> Tokenizer {
        Tokenizer { custom_tokens }
    }

    pub fn tokenize(&self, s: &str) -> TokenStack {
        let mut tokens = VecDeque::new();
        let custom_indices = self
            .custom_tokens
            .iter()
            .map(|token| s.match_indices(token))
            .flatten()
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let mut chars = s.chars().peekable();
        let mut i = 0;
        while let Some(c) = chars.next() {
            match c {
                '(' => tokens.push_back(Token::LeftParen),
                ')' => tokens.push_back(Token::RightParen),
                ',' => tokens.push_back(Token::Comma),
                _ => {
                    if c.is_whitespace() {
                        tokens.push_back(Token::Whitespace);
                        continue;
                    }
                    let mut token = String::new();
                    token.push(c);
                    let mut custom_token_found = false;
                    while let Some(&c) = chars.peek() {
                        if self.custom_tokens.contains(&token) {
                            tokens.push_back(Token::Object(token.clone()));
                            custom_token_found = true;
                            break;
                        }
                        if c == '('
                            || c == ')'
                            || c == ','
                            || c.is_whitespace()
                            || custom_indices.contains(&(i + 1))
                        {
                            break;
                        }
                        token.push(c);
                        chars.next();
                        i += 1;
                    }
                    if !custom_token_found {
                        tokens.push_back(Token::Object(token));
                    }
                }
            }
            i += 1;
        }
        TokenStack::new(tokens)
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct TokenStack(VecDeque<Token>);

impl TokenStack {
    pub fn new(tokens: VecDeque<Token>) -> Self {
        Self(tokens)
    }

    pub fn as_vector(&self) -> Vec<Token> {
        self.0.clone().into_iter().collect()
    }

    pub fn to_string(&self) -> String {
        self.as_vector()
            .into_iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn remove_whitespace(&mut self) {
        self.0 = self
            .0
            .clone()
            .into_iter()
            .filter(|t| t != &Token::Whitespace)
            .collect();
    }

    pub fn peek(&self) -> Option<Token> {
        if self.len() == 0 {
            return None;
        }
        Some(self.0[0].clone())
    }

    pub fn pop(&mut self) -> Option<Token> {
        self.0.pop_front()
    }

    pub fn pop_and_assert(&mut self, expected: Token) -> bool {
        let actual = self.pop();
        actual == Some(expected)
    }

    pub fn pop_and_assert_or_error(&mut self, expected: Token) -> Result<(), ParserError> {
        match self.pop() {
            None => Err(ParserError::NoTokensRemainingToInterpret),
            Some(actual) => {
                if actual != expected {
                    Err(ParserError::ExpectedButFound(expected, actual))
                } else {
                    Ok(())
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Token {
    LeftParen,
    RightParen,
    Comma,
    Whitespace,
    Object(String),
}

impl From<String> for Token {
    fn from(value: String) -> Self {
        Self::Object(value)
    }
}

impl From<&str> for Token {
    fn from(value: &str) -> Self {
        Self::Object(value.to_string())
    }
}

impl Token {
    pub fn to_string(&self) -> String {
        match self {
            Self::LeftParen => "(".to_string(),
            Self::RightParen => ")".to_string(),
            Self::Comma => ",".to_string(),
            Self::Whitespace => " ".to_string(),
            Self::Object(s) => s.clone(),
        }
    }
}

#[cfg(test)]
mod test_tokenizer {
    use super::*;

    #[test]
    fn test_tokenizer_tokenizes() {
        let tokenizer = Tokenizer::default();

        assert_eq!(
            tokenizer.tokenize("a").as_vector(),
            vec![Token::Object("a".to_string())]
        );
        assert_eq!(
            tokenizer.tokenize("f(a)").as_vector(),
            vec![
                Token::Object("f".to_string()),
                Token::LeftParen,
                Token::Object("a".to_string()),
                Token::RightParen
            ]
        );
        assert_eq!(
            tokenizer.tokenize("func(ab_c)").as_vector(),
            vec![
                Token::Object("func".to_string()),
                Token::LeftParen,
                Token::Object("ab_c".to_string()),
                Token::RightParen
            ]
        );
        assert_eq!(
            tokenizer.tokenize("func(ab,c)").as_vector(),
            vec![
                Token::Object("func".to_string()),
                Token::LeftParen,
                Token::Object("ab".to_string()),
                Token::Comma,
                Token::Object("c".to_string()),
                Token::RightParen
            ]
        );
        assert_eq!(
            tokenizer.tokenize("2 + 2 = 4").as_vector(),
            vec![
                Token::Object("2".to_string()),
                Token::Whitespace,
                Token::Object("+".to_string()),
                Token::Whitespace,
                Token::Object("2".to_string()),
                Token::Whitespace,
                Token::Object("=".to_string()),
                Token::Whitespace,
                Token::Object("4".to_string()),
            ]
        );
    }

    #[test]
    fn test_tokenizer_tokenizes_custom_tokens() {
        let mut tokenizer = Tokenizer::new_with_tokens(vec![
            "=".to_string(),
            "+".to_string(),
            "-".to_string(),
            "*".to_string(),
            "/".to_string(),
        ]);

        assert_eq!(
            tokenizer.tokenize("2 + 2 = 4").as_vector(),
            vec![
                Token::Object("2".to_string()),
                Token::Whitespace,
                Token::Object("+".to_string()),
                Token::Whitespace,
                Token::Object("2".to_string()),
                Token::Whitespace,
                Token::Object("=".to_string()),
                Token::Whitespace,
                Token::Object("4".to_string()),
            ]
        );

        assert_eq!(
            tokenizer.tokenize("2+2=4").as_vector(),
            vec![
                Token::Object("2".to_string()),
                Token::Object("+".to_string()),
                Token::Object("2".to_string()),
                Token::Object("=".to_string()),
                Token::Object("4".to_string()),
            ]
        );
    }
}
