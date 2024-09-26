use log::debug;
use std::{
    collections::HashSet,
    ops::{Add, Div, Mul, Sub},
};

use serde::{Deserialize, Serialize};

use super::transformation::TransformationError;

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlgorithmType {
    Addition,
    Subtraction,
    Multiplication,
    Division,
}

impl AlgorithmType {
    pub fn all() -> HashSet<Self> {
        vec![
            Self::Addition,
            Self::Subtraction,
            Self::Multiplication,
            Self::Division,
        ]
        .into_iter()
        .collect()
    }

    pub fn to_string(&self) -> String {
        match self {
            Self::Addition => "Addition",
            Self::Subtraction => "Subtraction",
            Self::Multiplication => "Multiplication",
            Self::Division => "Division",
        }
        .to_string()
    }

    pub fn from_string(s: &str) -> Result<Self, TransformationError> {
        match s.to_ascii_lowercase().as_ref() {
            "addition" => Ok(Self::Addition),
            "subtraction" => Ok(Self::Subtraction),
            "multiplication" => Ok(Self::Multiplication),
            "division" => Ok(Self::Division),
            &_ => Err(TransformationError::UnableToParse(s.to_string())),
        }
    }

    pub fn transform(&self, left: &str, right: &str) -> Result<String, TransformationError> {
        debug!("{}.transform({}, {})", self.to_string(), left, right);
        let left_value = Rational::parse(left)?;
        let right_value = Rational::parse(right)?;
        let final_value = match self {
            Self::Addition => left_value + right_value,
            Self::Subtraction => left_value - right_value,
            Self::Multiplication => left_value * right_value,
            Self::Division => left_value / right_value,
        };
        debug!("final_value: {:?}", final_value);
        Ok(final_value.to_string())
    }
}

// TODO Use rug or another arbitrary precision crate
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rational {
    internal: f64,
}

impl Add<Rational> for Rational {
    type Output = Rational;
    fn add(self, rhs: Rational) -> Self::Output {
        Self::new(self.internal + rhs.internal)
    }
}

impl Sub<Rational> for Rational {
    type Output = Rational;

    fn sub(self, rhs: Rational) -> Self::Output {
        Self::new(self.internal - rhs.internal)
    }
}

impl Mul<Rational> for Rational {
    type Output = Rational;

    fn mul(self, rhs: Rational) -> Self::Output {
        Self::new(self.internal * rhs.internal)
    }
}

impl Div<Rational> for Rational {
    type Output = Rational;

    fn div(self, rhs: Rational) -> Self::Output {
        Self::new(self.internal / rhs.internal)
    }
}

impl Rational {
    pub fn new(internal: f64) -> Self {
        Self { internal }
    }

    pub fn parse(s: &str) -> Result<Self, TransformationError> {
        s.parse::<f64>()
            .map(|n| Self::new(n))
            .map_err(|_| TransformationError::UnableToParse(s.to_string()))
    }

    pub fn to_string(&self) -> String {
        self.internal.to_string()
    }
}
