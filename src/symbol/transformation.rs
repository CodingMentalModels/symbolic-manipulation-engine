use std::collections::{HashMap, HashSet};

use crate::parsing::interpretation::Interpretation;
use crate::symbol::symbol_node::{Symbol, SymbolNode, SymbolNodeError};
use crate::symbol::symbol_type::{Type, TypeError};
use serde::{Deserialize, Serialize};

use super::symbol_node::{SymbolName, SymbolNodeAddress};
use super::symbol_type::TypeHierarchy;

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Transformation {
    ExplicitTransformation(ExplicitTransformation),
    AdditionAlgorithm(AdditionAlgorithm),
    ApplyToBothSidesTransformation(ApplyToBothSidesTransformation),
}

impl From<ExplicitTransformation> for Transformation {
    fn from(t: ExplicitTransformation) -> Self {
        Self::ExplicitTransformation(t)
    }
}

impl From<AdditionAlgorithm> for Transformation {
    fn from(algorithm: AdditionAlgorithm) -> Self {
        Self::AdditionAlgorithm(algorithm)
    }
}

impl From<ApplyToBothSidesTransformation> for Transformation {
    fn from(t: ApplyToBothSidesTransformation) -> Self {
        Self::ApplyToBothSidesTransformation(t)
    }
}

impl Transformation {
    pub fn transform(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        match self {
            Self::ExplicitTransformation(t) => t.typed_relabel_and_transform(hierarchy, statement),
            Self::AdditionAlgorithm(t) => t.transform(hierarchy, statement),
            Self::ApplyToBothSidesTransformation(t) => t.transform(hierarchy, statement),
        }
    }
    pub fn joint_transform(
        &self,
        hierarchy: &TypeHierarchy,
        left: &SymbolNode,
        right: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        let statement = left.clone().join(right.clone());
        let in_order = self.transform(hierarchy, &statement);
        if in_order.is_ok() {
            return in_order;
        }

        let reversed = right.clone().join(left.clone());
        self.transform(hierarchy, &reversed)
    }

    pub fn try_transform_into(
        &self,
        hierarchy: &TypeHierarchy,
        from: &SymbolNode,
        to: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        let valid_transformations = self.get_valid_transformations(hierarchy, from);
        if valid_transformations.contains(to) {
            Ok(to.clone())
        } else {
            Err(TransformationError::NoValidTransformations)
        }
    }

    pub fn get_valid_transformations(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> HashSet<SymbolNode> {
        let mut base_case = vec![statement.clone()].into_iter().collect::<HashSet<_>>();
        match self.typed_relabel_and_transform_at(hierarchy, statement, vec![]) {
            Ok(result) => {
                base_case.insert(result);
            }
            _ => {}
        };

        let mut to_return = base_case.clone();
        for potentially_transformed in base_case.iter() {
            let new_statements =
                self.get_valid_child_transformations(hierarchy, potentially_transformed);
            to_return = to_return.union(&new_statements).cloned().collect();
        }

        return to_return;
    }

    fn get_valid_child_transformations(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> HashSet<SymbolNode> {
        let mut child_to_valid_transformations = HashMap::new();
        for child in statement.get_children() {
            let possible_transformations = self.get_valid_transformations(hierarchy, child);
            child_to_valid_transformations.insert(child, possible_transformations);
        }
        let mut new_statements = vec![statement.clone()].into_iter().collect::<HashSet<_>>();

        let n_subsets = 1 << child_to_valid_transformations.len();
        for bitmask in 0..n_subsets {
            // Bitmask indicates whether to take the child or its transformed versions
            // TODO: There's a massive opportunity for optimization here by eliminating the cases where
            // there are no transformations

            let mut transformed_statements =
                vec![statement.clone()].into_iter().collect::<HashSet<_>>();
            for (i, child) in statement.get_children().iter().enumerate() {
                let mut updated_statements = transformed_statements.clone();
                let should_transform_ith_child = bitmask & (1 << i) != 0;
                if should_transform_ith_child {
                    let transformed_children_set = child_to_valid_transformations
                        .get(child)
                        .expect("We constructed the map from the same vector.");
                    for c in transformed_children_set {
                        for transformed_statement in transformed_statements.iter() {
                            let updated_statement = transformed_statement
                                .clone()
                                .with_child_replaced(i, c.clone())
                                .expect("Child index is guaranteed to be in range.");
                            updated_statements.insert(updated_statement);
                        }
                    }
                } else {
                    // Do nothing; child is fine as is
                }
                transformed_statements = updated_statements;
            }
            new_statements = new_statements
                .union(&transformed_statements)
                .cloned()
                .collect();
        }
        new_statements
    }

    pub fn typed_relabel_and_transform_at(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
        address: SymbolNodeAddress,
    ) -> Result<SymbolNode, TransformationError> {
        let substatement_to_transform = statement.get_node(address.clone()).ok_or_else(|| {
            TransformationError::InvalidSymbolNodeError(SymbolNodeError::InvalidAddress)
        })?;
        let transformed_substatement = self.transform(hierarchy, &substatement_to_transform)?;
        match statement.replace_node(&address, &transformed_substatement) {
            Ok(transformed_statement) => Ok(transformed_statement),
            Err(e) => Err(TransformationError::InvalidSymbolNodeError(e)),
        }
    }

    pub fn to_interpreted_string(&self, interpretations: &Vec<Interpretation>) -> String {
        match self {
            Self::ExplicitTransformation(t) => t.to_interpreted_string(interpretations),
            Self::AdditionAlgorithm(a) => a.to_string(),
            Self::ApplyToBothSidesTransformation(t) => t.to_interpreted_string(interpretations),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdditionAlgorithm {
    operator: Symbol,
    input_type: Type,
}

impl AdditionAlgorithm {
    pub fn new(operator: Symbol, input_type: Type) -> Self {
        Self {
            operator,
            input_type,
        }
    }

    pub fn get_operator(&self) -> Symbol {
        self.operator.clone()
    }

    pub fn get_input_type(&self) -> Type {
        self.input_type.clone()
    }

    pub fn transform(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        if !statement.has_children() {
            if hierarchy
                .is_subtype_of(
                    &statement.get_symbol().get_evaluates_to_type(),
                    &self.input_type,
                )
                .map_err(|e| Into::<TransformationError>::into(e))?
            {
                return Ok(statement.clone());
            } else {
                return Err(TransformationError::NoValidTransformations);
            }
        }

        if statement.get_n_children() != 2 {
            return Err(TransformationError::NoValidTransformations);
        }

        let children = statement.get_children().clone();
        let left = self.transform(hierarchy, &children[0])?;
        let right = self.transform(hierarchy, &children[1])?;

        let left_value = Self::try_parse_number(&left.get_root_name())?;
        let right_value = Self::try_parse_number(&right.get_root_name())?;
        // TODO This will overflow on big enough numbers
        let final_value = left_value + right_value;
        Ok(SymbolNode::leaf(Symbol::new(
            final_value.to_string(),
            self.input_type.clone(),
        )))
    }

    pub fn to_string(&self) -> String {
        format!(
            "{} {} {}",
            self.input_type.to_string(),
            self.operator.get_name(),
            self.input_type.to_string()
        )
        .to_string()
    }

    fn try_parse_number(symbol_name: &SymbolName) -> Result<f64, TransformationError> {
        // TODO: This will fail on big enough numbers
        return symbol_name
            .parse::<f64>()
            .map_err(|_| TransformationError::UnableToParse(symbol_name.clone()));
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyToBothSidesTransformation {
    symbol: Symbol,
    transformation: ExplicitTransformation,
}

impl ApplyToBothSidesTransformation {
    pub fn new(symbol: Symbol, transformation: ExplicitTransformation) -> Self {
        Self {
            symbol,
            transformation,
        }
    }

    pub fn get_symbol(&self) -> &Symbol {
        &self.symbol
    }

    pub fn get_symbol_type(&self) -> Type {
        self.symbol.get_evaluates_to_type()
    }

    pub fn get_transformation(&self) -> &ExplicitTransformation {
        &self.transformation
    }

    pub fn transform(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        if statement.get_symbol() != &self.symbol {
            return Err(TransformationError::SymbolDoesntMatch(
                statement.get_symbol().clone(),
            ));
        }

        if statement.get_n_children() != 2 {
            return Err(TransformationError::ApplyToBothSidesCalledOnNChildren(
                statement.get_n_children(),
            ));
        }

        let left = statement.get_children()[0].clone();
        let right = statement.get_children()[1].clone();

        let left_transformed = self
            .transformation
            .typed_relabel_and_transform(hierarchy, &left)?;
        let right_transformed = self
            .transformation
            .typed_relabel_and_transform(hierarchy, &right)?;

        Ok(SymbolNode::new(
            self.symbol.clone().into(),
            vec![left_transformed, right_transformed],
        ))
    }

    pub fn to_interpreted_string(&self, interpretations: &Vec<Interpretation>) -> String {
        format!(
            "Apply {} to Both Sides of {}",
            self.transformation.to_interpreted_string(interpretations),
            self.symbol.to_string()
        )
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExplicitTransformation {
    pub from: SymbolNode,
    pub to: SymbolNode,
}

impl From<(SymbolNode, SymbolNode)> for ExplicitTransformation {
    fn from(value: (SymbolNode, SymbolNode)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<(Symbol, SymbolNode)> for ExplicitTransformation {
    fn from(value: (Symbol, SymbolNode)) -> Self {
        Self::new(value.0.into(), value.1)
    }
}

impl ExplicitTransformation {
    pub fn new(from: SymbolNode, to: SymbolNode) -> ExplicitTransformation {
        ExplicitTransformation { from, to }
    }

    pub fn get_from(&self) -> &SymbolNode {
        &self.from
    }

    pub fn get_to(&self) -> &SymbolNode {
        &self.to
    }

    pub fn reflexivity(
        operator_name: String,
        operator_type: Type,
        object_name: String,
        object_type: Type,
    ) -> Self {
        let object = SymbolNode::leaf(Symbol::new(object_name, object_type));
        let node = SymbolNode::new(
            Symbol::new(operator_name, operator_type).into(),
            vec![object.clone(), object.clone()],
        );
        ExplicitTransformation::new(node.clone(), node)
    }

    pub fn symmetry(
        operator_name: String,
        operator_type: Type,
        object_names: (String, String),
        object_type: Type,
    ) -> Self {
        let left = SymbolNode::leaf(Symbol::new(object_names.0, object_type.clone()));
        let right = SymbolNode::leaf(Symbol::new(object_names.1, object_type));
        let operator = Symbol::new(operator_name, operator_type);
        let from = SymbolNode::new(operator.clone().into(), vec![left.clone(), right.clone()]);
        let to = SymbolNode::new(operator.into(), vec![right.clone(), left.clone()]);
        ExplicitTransformation::new(from, to)
    }

    pub fn commutivity(
        operator_name: String,
        operator_type: Type,
        object_names: (String, String),
        object_type: Type,
    ) -> Self {
        Self::symmetry(operator_name, operator_type, object_names, object_type)
    }

    pub fn associativity(
        operator_name: String,
        operator_type: Type,
        object_names: (String, String, String),
        object_type: Type,
    ) -> Self {
        let a = SymbolNode::leaf(Symbol::new(object_names.0, object_type.clone()));
        let b = SymbolNode::leaf(Symbol::new(object_names.1, object_type.clone()));
        let c = SymbolNode::leaf(Symbol::new(object_names.2, object_type));
        let operator = Symbol::new(operator_name, operator_type);

        let a_b = SymbolNode::new(operator.clone().into(), vec![a.clone(), b.clone()]);
        let a_b_then_c = SymbolNode::new(operator.clone().into(), vec![a_b.clone(), c.clone()]);

        let b_c = SymbolNode::new(operator.clone().into(), vec![b.clone(), c.clone()]);
        let a_then_b_c = SymbolNode::new(operator.clone().into(), vec![a.clone(), b_c.clone()]);

        ExplicitTransformation::new(a_b_then_c, a_then_b_c)
    }

    pub fn to_string(&self) -> String {
        format!("{} -> {}", self.from.to_string(), self.to.to_string())
    }

    pub fn to_interpreted_string(&self, interpretations: &Vec<Interpretation>) -> String {
        format!(
            "{} -> {}",
            self.from.to_interpreted_string(interpretations),
            self.to.to_interpreted_string(interpretations)
        )
    }

    pub fn get_variables(&self) -> HashSet<String> {
        let mut variables = self.from.get_symbols();
        variables.extend(self.to.get_symbols());
        variables.into_iter().map(|s| s.get_name()).collect()
    }

    fn relabel_and_transform_at(
        &self,
        statement: &SymbolNode,
        address: SymbolNodeAddress,
    ) -> Result<SymbolNode, TransformationError> {
        let substatement_to_transform = statement.get_node(address.clone()).ok_or_else(|| {
            TransformationError::InvalidSymbolNodeError(SymbolNodeError::InvalidAddress)
        })?;
        let transformed_substatement = self.relabel_and_transform(&substatement_to_transform)?;
        match statement.replace_node(&address, &transformed_substatement) {
            Ok(transformed_statement) => Ok(transformed_statement),
            Err(e) => Err(TransformationError::InvalidSymbolNodeError(e)),
        }
    }

    pub fn typed_relabel_and_transform(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        let generalized_transform = self.generalize_to_fit(hierarchy, statement)?;
        generalized_transform.relabel_and_transform(statement)
    }

    fn relabel_and_transform(
        &self,
        statement: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        match self.from.get_relabelling(&statement) {
            Ok(relabellings) => self.transform(&statement, &relabellings),
            Err(e) => Err(TransformationError::InvalidSymbolNodeError(e)),
        }
    }

    fn generalize_to_fit(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> Result<Self, TransformationError> {
        let new_from = hierarchy
            .instantiate(&self.from, statement)
            .map_err(|e| Into::<TransformationError>::into(e))?;
        let substitutions = self
            .from
            .get_typed_relabelling(hierarchy, &new_from)
            .map_err(|e| Into::<TransformationError>::into(e))?;
        let new_to = substitutions.substitute(&self.to);
        Ok(Self::new(new_from, new_to))
    }

    fn transform_all(
        &self,
        statement: &SymbolNode,
        relabellings: &HashMap<String, String>,
    ) -> Result<(SymbolNode, Vec<SymbolNodeAddress>), TransformationError> {
        self.transform_all_from_address(statement, relabellings, Vec::new())
    }

    fn transform_all_from_address(
        &self,
        statement: &SymbolNode,
        relabellings: &HashMap<String, String>,
        address: SymbolNodeAddress,
    ) -> Result<(SymbolNode, Vec<SymbolNodeAddress>), TransformationError> {
        let children_transformation_result = statement
            .get_children()
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut child_address = address.clone();
                child_address.push(i);
                self.transform_all_from_address(c, relabellings, child_address)
            })
            .collect::<Result<Vec<(SymbolNode, Vec<SymbolNodeAddress>)>, TransformationError>>()?;
        let (transformed_children, transformed_children_addresses) = children_transformation_result
            .into_iter()
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let transformed_children_addresses: Vec<SymbolNodeAddress> = transformed_children_addresses
            .into_iter()
            .flatten()
            .collect();

        let new_statement =
            SymbolNode::new(statement.get_symbol().clone().into(), transformed_children);

        match self.try_transform(&new_statement, relabellings) {
            Ok((transformed, true)) => {
                let children_addresses: Vec<SymbolNodeAddress> = transformed_children_addresses
                    .into_iter()
                    .map(|mut a| {
                        let mut new_head = address.clone();
                        new_head.append(&mut a);
                        new_head
                    })
                    .collect();
                let mut addresses = vec![address];
                addresses.extend(children_addresses.clone());
                Ok((transformed, addresses))
            }
            Ok((transformed, false)) => Ok((transformed, transformed_children_addresses)),
            Err(e) => Err(e),
        }
    }

    fn try_transform(
        &self,
        statement: &SymbolNode,
        relabelling: &HashMap<String, String>,
    ) -> Result<(SymbolNode, bool), TransformationError> {
        match self.transform(statement, relabelling) {
            Ok(transformed) => Ok((transformed, true)),
            Err(TransformationError::StatementDoesNotMatch(_, _)) => Ok((statement.clone(), false)),
            Err(e) => Err(e),
        }
    }

    fn transform_strict(
        &self,
        statement: &SymbolNode,
        relabellings: &HashMap<String, String>,
    ) -> Result<SymbolNode, TransformationError> {
        if self.get_variables() != relabellings.keys().cloned().collect() {
            return Err(TransformationError::RelabellingsKeysMismatch);
        }

        self.transform(statement, relabellings)
    }

    fn transform(
        &self,
        statement: &SymbolNode,
        relabellings: &HashMap<String, String>,
    ) -> Result<SymbolNode, TransformationError> {
        statement
            .validate()
            .map_err(|e| TransformationError::InvalidSymbolNodeError(e))?;

        let relabelled_from = self
            .from
            .relabel_all(&relabellings.clone().into_iter().collect());
        let relabelled_to = self
            .to
            .relabel_all(&relabellings.clone().into_iter().collect());

        if statement == &relabelled_from {
            Ok(relabelled_to)
        } else {
            Err(TransformationError::StatementDoesNotMatch(
                statement.clone(),
                relabelled_from,
            ))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransformationError {
    ConflictingTypes(String, Type, Type),
    InvalidSymbolNodeError(SymbolNodeError),
    InvalidTypes(TypeError),
    RelabellingsKeysMismatch,
    StatementDoesNotMatch(SymbolNode, SymbolNode),
    SymbolDoesntMatch(Symbol),
    ApplyToBothSidesCalledOnNChildren(usize),
    StatementTypesDoNotMatch,
    NoValidTransformations,
    UnableToParse(SymbolName),
}

impl From<SymbolNodeError> for TransformationError {
    fn from(value: SymbolNodeError) -> Self {
        match value {
            SymbolNodeError::ConflictingTypes(name, t_0, t_1) => {
                Self::ConflictingTypes(name, t_0, t_1)
            }
            _ => Self::InvalidSymbolNodeError(value),
        }
    }
}

impl From<TypeError> for TransformationError {
    fn from(value: TypeError) -> Self {
        Self::InvalidTypes(value)
    }
}

#[cfg(test)]
mod test_transformation {
    use crate::{
        parsing::{interpretation::Interpretation, parser::Parser, tokenizer::Token},
        symbol::symbol_node::SymbolNodeRoot,
    };

    use super::*;

    #[test]
    fn test_transformation_joint_transforms() {
        let hierarchy = TypeHierarchy::chain(vec!["Proposition".into()]).unwrap();
        let parser = Parser::new(vec![
            Interpretation::singleton("p", "Proposition".into()),
            Interpretation::singleton("q", "Proposition".into()),
            Interpretation::infix_operator("^".into(), 1, "Proposition".into()),
            Interpretation::infix_operator("=>".into(), 2, "Proposition".into()),
            Interpretation::singleton("a", "Proposition".into()),
            Interpretation::singleton("b", "Proposition".into()),
            Interpretation::singleton("c", "Proposition".into()),
        ]);

        let as_proposition =
            |name: &str| Symbol::new(name.to_string(), "Proposition".into()).into();
        let from = SymbolNode::new(
            SymbolNodeRoot::Join,
            vec![as_proposition("p"), as_proposition("q")],
        );
        let p_and_q = parser
            .parse_from_string(vec!["^".to_string()], "p^q")
            .unwrap();
        let transform: Transformation = ExplicitTransformation::new(from, p_and_q.clone()).into();
        let actual = transform
            .joint_transform(&hierarchy, &as_proposition("p"), &as_proposition("q"))
            .unwrap();
        assert_eq!(actual, p_and_q);

        let actual = transform
            .joint_transform(&hierarchy, &as_proposition("a"), &as_proposition("b"))
            .unwrap();
        let expected = parser
            .parse_from_string(vec!["^".to_string()], "a^b")
            .unwrap();
        assert_eq!(actual, expected);

        let actual = transform
            .joint_transform(&hierarchy, &p_and_q, &as_proposition("b"))
            .unwrap();
        let expected = parser
            .parse_from_string(vec!["^".to_string()], "(p^q)^b")
            .unwrap();
        assert_eq!(actual, expected);

        let p_implies_q = parser
            .parse_from_string(vec!["=>".to_string()], "p=>q")
            .unwrap();
        let from = SymbolNode::new(
            SymbolNodeRoot::Join,
            vec![as_proposition("p"), p_implies_q.clone()],
        );
        let transform: Transformation =
            ExplicitTransformation::new(from, as_proposition("q")).into();

        let expected = as_proposition("q");
        assert_eq!(
            transform
                .joint_transform(&hierarchy, &as_proposition("p"), &p_implies_q)
                .unwrap(),
            expected
        );
        // Order shouldn't matter
        assert_eq!(
            transform
                .joint_transform(&hierarchy, &p_implies_q, &as_proposition("p"))
                .unwrap(),
            expected
        );
    }

    #[test]
    fn test_transformation_applies_algorithm() {
        let algorithm =
            AdditionAlgorithm::new(Symbol::new("+".to_string(), "Real".into()), "Real".into());
        let parser = Parser::new(vec![
            Interpretation::singleton("a", "Real".into()),
            Interpretation::singleton("b", "Real".into()),
            Interpretation::singleton("c", "Real".into()),
            Interpretation::singleton("1", "Real".into()),
            Interpretation::singleton("2", "Real".into()),
            Interpretation::infix_operator("+".into(), 1, "Real".into()),
        ]);
        let from = parser
            .parse_from_string(vec!["+".to_string()], "a+b+c")
            .unwrap();
        let hierarchy = TypeHierarchy::chain(vec!["Real".into()]).unwrap();
        assert_eq!(
            algorithm.transform(&hierarchy, &from),
            Err(TransformationError::UnableToParse("a".to_string()))
        );

        let from = parser
            .parse_from_string(vec!["+".to_string()], "1+2")
            .unwrap();
        assert_eq!(
            algorithm.transform(&hierarchy, &from),
            Ok(SymbolNode::leaf(Symbol::new(
                "3".to_string(),
                "Real".into()
            )))
        );
    }

    #[test]
    fn test_transformation_gets_valid_transformations() {
        let hierarchy =
            TypeHierarchy::chain(vec!["Real".into(), "Integer".into(), "=".into()]).unwrap();
        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
        ];
        let parser = Parser::new(interpretations);

        let custom_tokens = vec!["=".to_string()];

        let irrelevant_transform: Transformation = ExplicitTransformation::associativity(
            "*".to_string(),
            "*".into(),
            ("j".to_string(), "l".to_string(), "k".to_string()),
            "Irrelevant".into(),
        )
        .into();

        let x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "x=y")
            .unwrap();

        assert_eq!(
            irrelevant_transform.get_valid_transformations(&hierarchy, &x_equals_y),
            vec![x_equals_y.clone()].into_iter().collect()
        );
        let transformation: Transformation = ExplicitTransformation::commutivity(
            "=".to_string(),
            "=".into(),
            ("a".to_string(), "b".to_string()),
            "Integer".into(),
        )
        .into();
        let y_equals_x = parser
            .parse_from_string(custom_tokens.clone(), "y = x")
            .unwrap();
        assert_eq!(
            transformation.transform(&hierarchy, &x_equals_y),
            Ok(y_equals_x.clone())
        );
        assert_eq!(
            transformation.get_valid_transformations(&hierarchy, &x_equals_y),
            vec![x_equals_y.clone(), y_equals_x.clone()]
                .into_iter()
                .collect()
        );

        // TODO Figure out if we think this needs to work
        //        let conversion = Transformation::new(
        //            Symbol::new("x".to_string(), "Integer".into()).into(),
        //            Symbol::new("x".to_string(), "Real".into()).into(),
        //        );
        //        assert_eq!(
        //            conversion.get_valid_transformations(
        //                &hierarchy,
        //                &Symbol::new("1".to_string(), "Integer".into()).into()
        //            ),
        //            vec![
        //                Symbol::new("1".to_string(), "Integer".into()).into(),
        //                Symbol::new("1".to_string(), "Real".into()).into(),
        //            ]
        //            .into_iter()
        //            .collect()
        //        );
        //        assert_eq!(
        //            conversion
        //                .get_valid_transformations(&hierarchy, &x_equals_y.clone())
        //                .len(),
        //            4
        //        );

        let x_equals_y_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "x=y=z") // ((x=y)=z)
            .unwrap();

        let expected = vec![
            x_equals_y_equals_z.clone(),
            parser
                .parse_from_string(custom_tokens.clone(), "y=x=z") // ((x=y)=z) => ((y=x)=z)
                .unwrap(),
        ];

        assert_eq!(
            transformation.get_valid_child_transformations(&hierarchy, &x_equals_y_equals_z),
            expected.into_iter().collect()
        );

        let z_equals_x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "z=(x=y)")
            .unwrap();

        let expected = vec![
            z_equals_x_equals_y.clone(),
            parser
                .parse_from_string(custom_tokens.clone(), "z=(y=x)")
                .unwrap(),
        ];

        assert_eq!(
            transformation.get_valid_child_transformations(&hierarchy, &z_equals_x_equals_y),
            expected.into_iter().collect()
        );

        let expected = vec![
            x_equals_y_equals_z.clone(),
            parser
                .parse_from_string(custom_tokens.clone(), "y=x=z") // ((x=y)=z) => ((y=x)=z)
                .unwrap(),
            parser
                .parse_from_string(custom_tokens.clone(), "z=(x=y)") // ((x=y)=z) => (z=(x=y))
                .unwrap(),
            parser
                .parse_from_string(custom_tokens.clone(), "z=(y=x)") // ((x=y)=z) => (z=(y=x))
                .unwrap(),
        ];
        assert_eq!(
            transformation.get_valid_transformations(&hierarchy, &x_equals_y_equals_z),
            expected.into_iter().collect()
        );

        // Using overloaded names shouldn't matter
        let transformation: Transformation = ExplicitTransformation::commutivity(
            "=".to_string(),
            "=".into(),
            ("x".to_string(), "y".to_string()),
            "Integer".into(),
        )
        .into();

        let x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "x=y")
            .unwrap();

        let expected = vec![
            x_equals_y.clone(),
            parser
                .parse_from_string(custom_tokens.clone(), "y=x")
                .unwrap(),
        ];

        let actual = transformation.get_valid_transformations(&hierarchy, &x_equals_y);

        assert_eq!(actual, expected.into_iter().collect());

        let x_equals_y_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "x=y=z") // ((x=y)=z)
            .unwrap();

        let expected = vec![
            x_equals_y_equals_z.clone(),
            parser
                .parse_from_string(custom_tokens.clone(), "y=x=z") // ((x=y)=z) => ((y=x)=z)
                .unwrap(),
            parser
                .parse_from_string(custom_tokens.clone(), "z=(x=y)") // ((x=y)=z) => (z=(x=y))
                .unwrap(),
            parser
                .parse_from_string(custom_tokens.clone(), "z=(y=x)") // ((x=y)=z) => (z=(y=x))
                .unwrap(),
        ];

        let actual = transformation.get_valid_transformations(&hierarchy, &x_equals_y_equals_z);

        assert_eq!(actual, expected.into_iter().collect());
    }

    #[test]
    fn test_transformation_transforms() {
        let transformation =
            ExplicitTransformation::new(SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b"));

        let statement = SymbolNode::leaf_object("c");

        let transformed = transformation.transform_strict(
            &statement,
            &vec![
                ("a".to_string(), "c".to_string()),
                ("b".to_string(), "d".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(transformed, Ok(SymbolNode::leaf_object("d")));

        let transformation = ExplicitTransformation::new(
            SymbolNode::new(
                "=".into(),
                vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
            ),
            SymbolNode::new(
                "=".into(),
                vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("a")],
            ),
        );

        let self_equals_statement = SymbolNode::new(
            "=".into(),
            vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("a")],
        );

        let transformed = transformation.transform_strict(
            &self_equals_statement,
            &vec![
                ("a".to_string(), "a".to_string()),
                ("b".to_string(), "a".to_string()),
                ("=".to_string(), "=".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(transformed, Ok(self_equals_statement));
    }

    #[test]
    fn test_transformation_transforms_all() {
        let commutativity = ExplicitTransformation::new(
            SymbolNode::new(
                "=".into(),
                vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
            ),
            SymbolNode::new(
                "=".into(),
                vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("a")],
            ),
        );

        let statement = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );

        let (transformed, addresses) = commutativity
            .transform_all(
                &statement,
                &vec![
                    ("a".to_string(), "a".to_string()),
                    ("b".to_string(), "b".to_string()),
                    ("c".to_string(), "c".to_string()),
                    ("=".to_string(), "=".to_string()),
                ]
                .into_iter()
                .collect(),
            )
            .unwrap();

        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("a")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );
        assert_eq!(transformed, expected);
        assert_eq!(addresses, vec![vec![0]]);
    }

    #[test]
    fn test_transformation_generalizes_to_fit() {
        let hierarchy = TypeHierarchy::chain(vec!["Integer".into(), "=".into()]).unwrap();
        let trivial_t =
            ExplicitTransformation::new(SymbolNode::leaf_object("c"), SymbolNode::leaf_object("d"));

        let trivial = SymbolNode::leaf_object("c");
        assert_eq!(
            trivial_t.generalize_to_fit(&hierarchy, &trivial).unwrap(),
            trivial_t
        );

        let different_t =
            ExplicitTransformation::new(SymbolNode::leaf_object("a"), SymbolNode::leaf_object("d"));

        let different_name = SymbolNode::leaf_object("a");
        assert_eq!(
            different_t
                .generalize_to_fit(&hierarchy, &different_name)
                .unwrap(),
            different_t
        );

        let overloaded_t =
            ExplicitTransformation::new(SymbolNode::leaf_object("d"), SymbolNode::leaf_object("d"));
        let overloaded_name = SymbolNode::leaf_object("d");
        assert_eq!(
            overloaded_t
                .generalize_to_fit(&hierarchy, &overloaded_name)
                .unwrap(),
            overloaded_t
        );

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
        ];
        let parser = Parser::new(interpretations);

        let custom_tokens = vec!["=".to_string()];

        let symmetry = ExplicitTransformation::symmetry(
            "=".to_string(),
            "=".into(),
            ("a".to_string(), "b".to_string()),
            "Integer".into(),
        );

        let x_equals_y_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=z")
            .unwrap();

        let z_equals_x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "z=(x=y)")
            .unwrap();

        assert_eq!(
            symmetry
                .generalize_to_fit(&hierarchy, &x_equals_y_equals_z)
                .unwrap(),
            ExplicitTransformation::new(x_equals_y_equals_z.clone(), z_equals_x_equals_y.clone()),
            "\n\n{} \nvs. \n{}",
            symmetry
                .generalize_to_fit(&hierarchy, &x_equals_y_equals_z)
                .unwrap()
                .to_string(),
            ExplicitTransformation::new(x_equals_y_equals_z, z_equals_x_equals_y.clone())
                .to_string(),
        );

        let symmetry = ExplicitTransformation::symmetry(
            "=".to_string(),
            "=".into(),
            ("x".to_string(), "y".to_string()),
            "Integer".into(),
        );

        assert_eq!(
            symmetry
                .generalize_to_fit(&hierarchy, &x_equals_y_equals_z)
                .unwrap(),
            ExplicitTransformation::new(x_equals_y_equals_z.clone(), z_equals_x_equals_y.clone()),
            "\n\n{} \nvs. \n{}",
            symmetry
                .generalize_to_fit(&hierarchy, &x_equals_y_equals_z)
                .unwrap()
                .to_string(),
            ExplicitTransformation::new(x_equals_y_equals_z, z_equals_x_equals_y.clone())
                .to_string(),
        );

        // TODO: Figure out whether this needs to work
        //        let conversion = Transformation::new(
        //            Symbol::new("x".to_string(), "Integer".into()).into(),
        //            Symbol::new("x".to_string(), "Real".into()).into(),
        //        );
        //        assert_eq!(
        //            conversion
        //                .typed_generalize_to_fit(
        //                    &hierarchy,
        //                    &Symbol::new("1".to_string(), "Integer".into()).into(),
        //                )
        //                .unwrap(),
        //            Transformation::new(
        //                Symbol::new("1".to_string(), "Integer".into()).into(),
        //                Symbol::new("1".to_string(), "Real".into()).into(),
        //            )
        //        );
    }

    #[test]
    fn test_transformation_typed_transforms_at() {
        let hierarchy =
            TypeHierarchy::chain(vec!["Real".into(), "Integer".into(), "=".into()]).unwrap();
        let transformation: Transformation =
            ExplicitTransformation::new(SymbolNode::leaf_object("c"), SymbolNode::leaf_object("d"))
                .into();

        let a_equals_b = SymbolNode::new(
            "=".into(),
            vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
        );

        let d_equals_b = SymbolNode::new(
            "=".into(),
            vec![SymbolNode::leaf_object("d"), SymbolNode::leaf_object("b")],
        );

        let transformed =
            transformation.typed_relabel_and_transform_at(&hierarchy, &a_equals_b, vec![0]);

        assert_eq!(transformed, Ok(d_equals_b));

        let a_equals_b_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );

        let a_equals_d_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("d")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );
        let transformed = transformation.typed_relabel_and_transform_at(
            &hierarchy,
            &a_equals_b_equals_c,
            vec![0, 1],
        );

        assert_eq!(transformed, Ok(a_equals_d_equals_c));

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
        ];
        let parser = Parser::new(interpretations);

        let custom_tokens = vec!["=".to_string()];

        let transformation: Transformation = ExplicitTransformation::symmetry(
            "=".to_string(),
            "=".into(),
            ("a".to_string(), "b".to_string()),
            "Integer".into(),
        )
        .into();

        let x_equals_y_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=z")
            .unwrap();

        let z_equals_x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "z=(x=y)")
            .unwrap();

        assert_eq!(
            transformation.typed_relabel_and_transform_at(&hierarchy, &x_equals_y_equals_z, vec![]),
            Ok(z_equals_x_equals_y.clone())
        );

        let y_equals_x_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "(y=x)=z")
            .unwrap();

        assert_eq!(
            transformation.typed_relabel_and_transform_at(
                &hierarchy,
                &x_equals_y_equals_z,
                vec![0]
            ),
            Ok(y_equals_x_equals_z)
        );

        let transformation: Transformation = ExplicitTransformation::symmetry(
            "=".to_string(),
            "=".into(),
            ("x".to_string(), "y".to_string()),
            "Integer".into(),
        )
        .into();

        assert_eq!(
            transformation.typed_relabel_and_transform_at(&hierarchy, &x_equals_y_equals_z, vec![]),
            Ok(z_equals_x_equals_y)
        );

        // TODO: Figure out if we think this needs to work
        //        let conversion = Transformation::new(
        //            Symbol::new("x".to_string(), "Integer".into()).into(),
        //            Symbol::new("x".to_string(), "Real".into()).into(),
        //        );
        //        assert_eq!(
        //            conversion
        //                .typed_transform_at(
        //                    &hierarchy,
        //                    &Symbol::new("1".to_string(), "Integer".into()).into(),
        //                    vec![],
        //                )
        //                .unwrap(),
        //            Symbol::new("1".to_string(), "Real".into()).into(),
        //        );
    }

    #[test]
    fn test_transformation_transforms_at() {
        let transformation =
            ExplicitTransformation::new(SymbolNode::leaf_object("c"), SymbolNode::leaf_object("d"));

        let a_equals_b = SymbolNode::new(
            "=".into(),
            vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
        );

        let d_equals_b = SymbolNode::new(
            "=".into(),
            vec![SymbolNode::leaf_object("d"), SymbolNode::leaf_object("b")],
        );

        let transformed = transformation.relabel_and_transform_at(&a_equals_b, vec![0]);

        assert_eq!(transformed, Ok(d_equals_b));

        let a_equals_b_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );

        let a_equals_d_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("d")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );
        let transformed = transformation.relabel_and_transform_at(&a_equals_b_equals_c, vec![0, 1]);

        assert_eq!(transformed, Ok(a_equals_d_equals_c));
    }

    #[test]
    fn test_transformation_reflexivity() {
        let transformation = ExplicitTransformation::reflexivity(
            "=".to_string(),
            "Rational".into(),
            "x".to_string(),
            Type::Object,
        );

        let interpretations = vec![Interpretation::infix_operator(
            Token::Object("=".to_string()),
            3,
            "Rational".into(),
        )];

        let parser = Parser::new(interpretations);
        let expected = parser
            .parse_from_string(vec!["=".to_string()], "x=x")
            .unwrap();

        assert_eq!(transformation.from, expected);
        assert_eq!(transformation.to, expected);
    }

    #[test]
    fn test_transformation_symmetry() {
        let transformation = ExplicitTransformation::symmetry(
            "+".to_string(),
            "Complex".into(),
            ("x".to_string(), "y".to_string()),
            "Complex".into(),
        );

        let interpretations = vec![
            Interpretation::infix_operator(Token::Object("+".to_string()), 3, "Complex".into()),
            Interpretation::singleton("x", "Complex".into()),
            Interpretation::singleton("y", "Complex".into()),
        ];
        let parser = Parser::new(interpretations);

        let expected_from = parser
            .parse_from_string(vec!["+".to_string()], "x+y")
            .unwrap();

        let expected_to = parser
            .parse_from_string(vec!["+".to_string()], "y+x")
            .unwrap();

        assert_eq!(transformation.from, expected_from);
        assert_eq!(transformation.to, expected_to);
    }
}
