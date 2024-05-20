use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::parser::ParserError;
use crate::constants::*;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Tokenizer {
    custom_tokens: Vec<String>,
    token_in_progress: Vec<char>,
    tokens: VecDeque<Token>,
}

impl Tokenizer {
    pub fn new_with_tokens(custom_tokens: Vec<String>) -> Tokenizer {
        Tokenizer {
            custom_tokens,
            token_in_progress: Vec::new(),
            tokens: VecDeque::new(),
        }
    }

    pub fn tokenize(&mut self, s: &str) -> TokenStack {
        self.token_in_progress = Vec::new();
        self.tokens = VecDeque::new();
        let mut queue = CharQueue::from_string(s.to_string());
        while let Some(_token) = queue.peek() {
            if queue.starts_with(&LPAREN.to_string()) {
                self.consume_token_in_progress();
                self.tokens.push_back(Token::LeftParen);
                queue.take();
            } else if queue.starts_with(&RPAREN.to_string()) {
                self.consume_token_in_progress();
                self.tokens.push_back(Token::RightParen);
                queue.take();
            } else if queue.starts_with(&COMMA.to_string()) {
                self.consume_token_in_progress();
                self.tokens.push_back(Token::Comma);
                queue.take();
            } else if queue.starts_with_whitespace() {
                self.consume_token_in_progress();
                self.tokens.push_back(Token::Whitespace);
                queue.take();
            } else {
                let mut found_custom_token = false;
                for custom_token in self.custom_tokens.clone().iter() {
                    if queue.starts_with(custom_token) {
                        self.consume_token_in_progress();
                        self.tokens
                            .push_back(Token::Object(queue.take_n_as_string(custom_token.len())));
                        found_custom_token = true;
                        break;
                    }
                }
                if !found_custom_token {
                    self.token_in_progress
                        .push(queue.take().expect("We know there's a next one."));
                }
            }
        }
        self.consume_token_in_progress();
        TokenStack::new(self.tokens.clone())
    }

    fn consume_token_in_progress(&mut self) {
        if self.token_in_progress.len() > 0 {
            self.tokens.push_back(Token::Object(
                self.token_in_progress.clone().into_iter().collect(),
            ));
            self.token_in_progress = Vec::new();
        }
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

    pub fn pop_or_error(&mut self) -> Result<Token, ParserError> {
        match self.pop() {
            None => Err(ParserError::NoTokensRemainingToInterpret),
            Some(actual) => Ok(actual),
        }
    }

    pub fn pop_and_assert(&mut self, expected: Token) -> bool {
        let actual = self.pop();
        actual == Some(expected)
    }

    pub fn pop_and_assert_or_error(&mut self, expected: Token) -> Result<(), ParserError> {
        self.pop_or_error().map(|actual| {
            if actual != expected {
                Err(ParserError::ExpectedButFound(expected, actual))
            } else {
                Ok(())
            }
        })?
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
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

impl From<&String> for Token {
    fn from(value: &String) -> Self {
        value.as_str().into()
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

pub struct CharQueue {
    queue: VecDeque<char>,
}

impl CharQueue {
    pub fn new(queue: VecDeque<char>) -> Self {
        Self { queue }
    }

    pub fn from_string(s: String) -> Self {
        Self::new(s.chars().into_iter().collect())
    }

    pub fn peek(&self) -> Option<char> {
        let to_return = self.peek_n(1);
        if to_return.len() == 0 {
            return None;
        } else {
            Some(to_return[0])
        }
    }

    pub fn peek_n(&self, n: usize) -> Vec<char> {
        if self.len() <= n {
            self.queue.clone().into_iter().collect()
        } else {
            self.queue.clone().into_iter().take(n).collect()
        }
    }

    pub fn take(&mut self) -> Option<char> {
        let to_return = self.take_n(1);
        if to_return.len() == 0 {
            None
        } else {
            Some(to_return[0])
        }
    }

    pub fn take_n(&mut self, n: usize) -> VecDeque<char> {
        if self.queue.len() <= n {
            std::mem::replace(&mut self.queue, VecDeque::new())
        } else {
            let mut taken = VecDeque::with_capacity(n);
            for _ in 0..n {
                // This unwrap is safe because we know there are at least n elements.
                taken.push_back(self.queue.pop_front().unwrap());
            }
            taken
        }
    }

    pub fn take_n_as_string(&mut self, n: usize) -> String {
        self.take_n(n).into_iter().collect::<String>()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn starts_with_whitespace(&self) -> bool {
        match self.peek() {
            None => false,
            Some(c) => c.is_whitespace(),
        }
    }

    pub fn starts_with(&self, s: &str) -> bool {
        self.to_string().starts_with(s)
    }

    pub fn to_string(&self) -> String {
        self.queue.clone().into_iter().collect()
    }
}

#[cfg(test)]
mod test_tokenizer {
    use super::*;

    #[test]
    fn test_tokenizer_tokenizes() {
        let mut tokenizer = Tokenizer::default();

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

    #[test]
    fn test_token_stack_pops() {
        let mut tokenizer = Tokenizer::new_with_tokens(vec![
            "=".to_string(),
            "+".to_string(),
            "-".to_string(),
            "*".to_string(),
            "/".to_string(),
        ]);

        let mut stack = tokenizer.tokenize("2 + 2 = 4");
        assert_eq!(stack.pop(), Some("2".into()));
        assert_eq!(stack.pop_or_error(), Ok(Token::Whitespace));
        assert_eq!(stack.pop_and_assert_or_error("+".into()), Ok(()));
        assert_eq!(stack.pop_and_assert_or_error(Token::Whitespace), Ok(()));
        assert_eq!(stack.pop_and_assert_or_error("2".into()), Ok(()));
        assert_eq!(stack.pop_and_assert_or_error(Token::Whitespace), Ok(()));
        assert_eq!(stack.pop_and_assert_or_error("=".into()), Ok(()));
        assert_eq!(stack.pop_and_assert_or_error(Token::Whitespace), Ok(()));
        assert_eq!(stack.pop_and_assert_or_error("4".into()), Ok(()));
        assert_eq!(stack.pop(), None);
        assert_eq!(
            stack.pop_or_error(),
            Err(ParserError::NoTokensRemainingToInterpret)
        );
        assert_eq!(
            stack.pop_and_assert_or_error(Token::Whitespace),
            Err(ParserError::NoTokensRemainingToInterpret)
        );

        let mut stack = tokenizer.tokenize("2 + 2 = 4");
        assert_eq!(
            stack.pop_and_assert_or_error(Token::Whitespace),
            Err(ParserError::ExpectedButFound(Token::Whitespace, "2".into()))
        );
    }

    #[test]
    fn test_char_queue_takes() {
        let mut queue = CharQueue::from_string("ABCDEFGH".to_string());
        assert_eq!(queue.peek(), Some('A'));
        assert_eq!(queue.peek_n(3), vec!['A', 'B', 'C']);
        assert_eq!(
            queue.peek_n(12),
            vec!['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        );

        assert_eq!(queue.take(), Some('A'));
        assert_eq!(queue.peek_n(12), vec!['B', 'C', 'D', 'E', 'F', 'G', 'H']);

        assert_eq!(queue.take_n(3), vec!['B', 'C', 'D']);
        assert_eq!(queue.take_n(10), vec!['E', 'F', 'G', 'H']);
    }

    #[test]
    fn test_char_queue_takes_n_as_string() {
        let mut queue = CharQueue::from_string("ABCDEFGH".to_string());

        assert_eq!(queue.take_n_as_string(1), "A".to_string());
        assert_eq!(queue.take_n_as_string(3), "BCD".to_string());
        assert_eq!(queue.take_n_as_string(10), "EFGH".to_string());
    }

    #[test]
    fn test_char_queue_starts_with_whitespace() {
        let queue = CharQueue::from_string("".to_string());
        assert!(!queue.starts_with_whitespace());

        let queue = CharQueue::from_string("123456789".to_string());
        assert!(!queue.starts_with_whitespace());

        let queue = CharQueue::from_string(" 123456789".to_string());
        assert!(queue.starts_with_whitespace());

        let queue = CharQueue::from_string("  123456789".to_string());
        assert!(queue.starts_with_whitespace());

        let queue = CharQueue::from_string("\n123456789".to_string());
        assert!(queue.starts_with_whitespace());

        let queue = CharQueue::from_string(" 2 = 4".to_string());
        assert!(queue.starts_with_whitespace());
    }

    #[test]
    fn test_char_queue_starts_with() {
        let mut queue = CharQueue::from_string("123456789".to_string());
        assert!(queue.starts_with(""));
        assert!(queue.starts_with("123"));
        assert!(!queue.starts_with("12345678999999999999"));
        assert!(!queue.starts_with("234"));
        let one = queue.take();
        assert!(queue.starts_with("234"));
        assert!(!queue.starts_with("2345678999999999999"));
    }
}
