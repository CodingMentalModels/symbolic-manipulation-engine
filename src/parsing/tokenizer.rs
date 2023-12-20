

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Tokenizer {
    custom_tokens: Vec<String>,
}

impl Tokenizer {

    pub fn new_with_tokens(custom_tokens: Vec<String>) -> Tokenizer {
        Tokenizer {
            custom_tokens,
        }
    }

    pub fn tokenize(&self, s: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let custom_indices = self.custom_tokens.iter().map(|token| s.match_indices(token)).flatten()
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let mut chars = s.chars().peekable();
        let mut i = 0;
        while let Some(c) = chars.next() {
            match c {
                '(' => tokens.push(Token::LeftParen),
                ')' => tokens.push(Token::RightParen),
                ',' => tokens.push(Token::Comma),
                _ => {
                    if c.is_whitespace() {
                        tokens.push(Token::Whitespace);
                        continue;
                    }
                    let mut token = String::new();
                    token.push(c);
                    let mut custom_token_found = false;
                    while let Some(&c) = chars.peek() {
                        if self.custom_tokens.contains(&token) {
                            tokens.push(Token::Custom(token.clone()));
                            custom_token_found = true;
                            break;
                        }
                        if c == '(' || c == ')' || c == ',' || c.is_whitespace() || custom_indices.contains(&(i + 1)) {
                            break;
                        }
                        token.push(c);
                        chars.next();
                        i += 1;
                    }
                    if !custom_token_found {
                        tokens.push(Token::Object(token));
                    }
                }
            }
            i += 1;
        }
        tokens
    }

}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Token {
    LeftParen,
    RightParen,
    Comma,
    Whitespace,
    Object(String),
    Custom(String),
}

impl Token {

}

#[cfg(test)]
mod test_tokenizer {
    use super::*;

    #[test]
    fn test_tokenizer_tokenizes() {
        
        let mut tokenizer = Tokenizer::default();

        assert_eq!(tokenizer.tokenize("a"), vec![Token::Object("a".to_string())]);
        assert_eq!(
            tokenizer.tokenize("f(a)"),
            vec![Token::Object("f".to_string()), Token::LeftParen, Token::Object("a".to_string()), Token::RightParen]
        );
        assert_eq!(
            tokenizer.tokenize("func(ab_c)"),
            vec![Token::Object("func".to_string()), Token::LeftParen, Token::Object("ab_c".to_string()), Token::RightParen]
        );
        assert_eq!(
            tokenizer.tokenize("func(ab,c)"),
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
            tokenizer.tokenize("2 + 2 = 4"),
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

        let mut tokenizer = Tokenizer::new_with_tokens(vec!["=".to_string(), "+".to_string(), "-".to_string(), "*".to_string(), "/".to_string()]);

        assert_eq!(
            tokenizer.tokenize("2 + 2 = 4"),
            vec![
                Token::Object("2".to_string()),
                Token::Whitespace,
                Token::Custom("+".to_string()),
                Token::Whitespace,
                Token::Object("2".to_string()),
                Token::Whitespace,
                Token::Custom("=".to_string()),
                Token::Whitespace,
                Token::Object("4".to_string()),
            ]
        );

        assert_eq!(
            tokenizer.tokenize("2+2=4"),
            vec![
                Token::Object("2".to_string()),
                Token::Custom("+".to_string()),
                Token::Object("2".to_string()),
                Token::Custom("=".to_string()),
                Token::Object("4".to_string()),
            ]
        );
        
    }
}