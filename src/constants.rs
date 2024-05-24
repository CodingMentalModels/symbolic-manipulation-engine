// Tokens
pub const LPAREN: char = '(';
pub const RPAREN: char = ')';
pub const COMMA: char = ',';

// Parsing
pub const DEFAULT_PRECEDENCE: u8 = 0;
pub const PARENTHESIS_PRECEDENCE: u8 = 99;
pub const COMMA_PRECEDENCE: u8 = 99;

// Serializing
pub const SERIALIZED_OBJECT_TYPE: &str = "__object_type_do_not_overload";
pub const SERIALIZED_DELIMITER_TYPE: &str = "__delimiter_type_do_not_overload";
