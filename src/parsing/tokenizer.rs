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
        let mut chars = MultiPeek::new(s.chars());
        let mut next_token: Option<Token> = None;
        while let Some(c) = chars.peek() {
            match c {
                '(' => next_token = Some(Token::LeftParen),
                ')' => next_token = Some(Token::RightParen),
                ',' => next_token = Some(Token::Comma),
                _ => {
                    if c.is_whitespace() {
                        next_token = Some(Token::Whitespace);
                    } else {
                        for custom_token in self.custom_tokens.iter() {
                            if chars
                                .peek_many(custom_token.len())
                                .map(|s| s.into_iter().collect::<String>())
                                == Some(custom_token.clone())
                            {
                                next_token = Some(Token::Object(custom_token.clone()));
                            }
                        }
                    }
                    match next_token {
                        Some(ref t) => {
                            tokens.push_back(t.clone());
                            chars.next_many(t.len());
                        }
                        None => tokens.push_back(Token::Object(chars.collect())),
                    }
                }
            }
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

    pub fn len(&self) -> usize {
        self.to_string().len()
    }
}

struct MultiPeek<I: Iterator> {
    iter: I,
    peeked: Vec<I::Item>,
}

impl<I: Iterator> MultiPeek<I>
where
    I::Item: Clone,
{
    fn new(iter: I) -> Self {
        Self {
            iter,
            peeked: Vec::new(),
        }
    }

    // Peek at the very next item without consuming it
    fn peek(&mut self) -> Option<&I::Item> {
        if self.peeked.is_empty() {
            if let Some(item) = self.iter.next() {
                self.peeked.push(item);
            }
        }
        self.peeked.first()
    }

    // Peek at the next n items without consuming them
    fn peek_many(&mut self, n: usize) -> Option<&[I::Item]> {
        while self.peeked.len() < n {
            if let Some(item) = self.iter.next() {
                self.peeked.push(item);
            } else {
                break;
            }
        }
        if self.peeked.is_empty() {
            None
        } else {
            Some(&self.peeked[..self.peeked.len().min(n)])
        }
    }

    fn next_many(&mut self, n: usize) -> Vec<I::Item> {
        let mut result = Vec::new();

        // First, take as many items as possible from the already peeked ones
        while result.len() < n && !self.peeked.is_empty() {
            result.push(self.peeked.remove(0));
        }

        // If more items are needed, take them directly from the underlying iterator
        while result.len() < n {
            if let Some(item) = self.iter.next() {
                result.push(item);
            } else {
                break; // Stop if the iterator runs out of items
            }
        }

        result
    }
}

impl<I: Iterator> Iterator for MultiPeek<I>
where
    I::Item: Clone,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.peeked.is_empty() {
            Some(self.peeked.remove(0))
        } else {
            self.iter.next()
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
