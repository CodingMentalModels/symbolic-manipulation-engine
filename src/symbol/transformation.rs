use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::parsing::parser::Parser;
use crate::symbol::symbol_node::{Symbol, SymbolNode, SymbolNodeError};
use crate::symbol::symbol_type::{Type, TypeError};

use super::symbol_node::SymbolNodeAddress;
use super::symbol_type::TypeHierarchy;

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transformation {
    pub from: SymbolNode,
    pub to: SymbolNode,
}

impl From<(SymbolNode, SymbolNode)> for Transformation {
    fn from(value: (SymbolNode, SymbolNode)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<(Symbol, SymbolNode)> for Transformation {
    fn from(value: (Symbol, SymbolNode)) -> Self {
        Self::new(value.0.into(), value.1)
    }
}

impl Transformation {
    pub fn new(from: SymbolNode, to: SymbolNode) -> Transformation {
        Transformation { from, to }
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
            Symbol::new(operator_name, operator_type),
            vec![object.clone(), object.clone()],
        );
        Transformation::new(node.clone(), node)
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
        let from = SymbolNode::new(operator.clone(), vec![left.clone(), right.clone()]);
        let to = SymbolNode::new(operator, vec![right.clone(), left.clone()]);
        Transformation::new(from, to)
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

        let a_b = SymbolNode::new(operator.clone(), vec![a.clone(), b.clone()]);
        let a_b_then_c = SymbolNode::new(operator.clone(), vec![a_b.clone(), c.clone()]);

        let b_c = SymbolNode::new(operator.clone(), vec![b.clone(), c.clone()]);
        let a_then_b_c = SymbolNode::new(operator.clone(), vec![a.clone(), b_c.clone()]);

        Transformation::new(a_b_then_c, a_then_b_c)
    }

    pub fn to_string(&self) -> String {
        format!("{} -> {}", self.from.to_string(), self.to.to_string())
    }

    pub fn get_variables(&self) -> HashSet<String> {
        let mut variables = self.from.get_symbols();
        variables.extend(self.to.get_symbols());
        variables.into_iter().map(|s| s.get_name()).collect()
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
        match self.typed_transform_at(hierarchy, statement, vec![]) {
            Ok(result) => {
                base_case.insert(result);
            }
            _ => {}
        };

        // For each base case element (the original statement and maybe the one with the root transformed)
        // Loop through all the subsets of children to transform and transform them to each
        // potential option
        let mut to_return = base_case.clone();
        for potentially_transformed in base_case {
            println!(
                "Transforming children of Potentially Transformed: {:?}",
                potentially_transformed
            );
            let mut transformed_children = HashMap::new();
            for child in potentially_transformed.get_children() {
                let child_transformations = self.get_valid_transformations(hierarchy, child);
                transformed_children.insert(child, child_transformations);
            }
            let n_subsets = 1 << transformed_children.len();
            for bitmask in 0..n_subsets {
                // Bitmask indicates whether to take the child or its transformed versions
                // TODO: There's a massive opportunity for optimization here by eliminating the cases where
                // there are no transformations
                println!("Bitmask: {:#018b}", bitmask);

                let mut new_statements = vec![potentially_transformed.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>();

                for (i, child) in potentially_transformed.get_children().iter().enumerate() {
                    if bitmask & (1 << i) != 0 {
                        let mut updated_statements = HashSet::new();
                        for statement_to_transform in &new_statements {
                            let transformed_children_set = transformed_children
                                .get(child)
                                .expect("We constructed the map from the same vector.");

                            for c in transformed_children_set {
                                let transformed_statement = statement_to_transform
                                    .clone()
                                    .with_child_replaced(i, c.clone())
                                    .expect("Child index is guaranteed to be in range.");
                                updated_statements.insert(transformed_statement);
                            }
                        }
                        new_statements = updated_statements;
                    } else {
                        // Do nothing; child is fine as is
                    }
                }
                to_return = to_return.union(&new_statements).cloned().collect();
            }
        }

        println!(
            "End get_valid_transformations({}).  Returning {} results:\n{:?}",
            statement.to_string(),
            to_return.len(),
            to_return
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("; ")
        );
        return to_return;
    }

    pub fn typed_transform_at(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
        address: SymbolNodeAddress,
    ) -> Result<SymbolNode, TransformationError> {
        let substatement_to_transform = statement.get_node(address.clone()).ok_or_else(|| {
            TransformationError::InvalidSymbolNode(SymbolNodeError::InvalidAddress)
        })?;
        let generalized_transform =
            self.typed_generalize_to_fit(hierarchy, &substatement_to_transform)?;
        generalized_transform.transform_at(statement, address)
    }

    pub fn transform_at(
        &self,
        statement: &SymbolNode,
        address: SymbolNodeAddress,
    ) -> Result<SymbolNode, TransformationError> {
        let substatement_to_transform = statement.get_node(address.clone()).ok_or_else(|| {
            TransformationError::InvalidSymbolNode(SymbolNodeError::InvalidAddress)
        })?;
        match self.from.get_relabelling(&substatement_to_transform) {
            Ok(relabellings) => {
                let transformed_substatement =
                    self.transform(&substatement_to_transform, &relabellings)?;
                match statement.replace_node(address, transformed_substatement) {
                    Ok(transformed_statement) => Ok(transformed_statement),
                    Err(e) => Err(TransformationError::InvalidSymbolNode(e)),
                }
            }
            Err(e) => Err(TransformationError::InvalidSymbolNode(e)),
        }
    }

    fn typed_generalize_to_fit(
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
        println!(
            "typed_generalize_to_fit_with:\nSubstitutions: {:?}",
            substitutions
        );
        let mut new_to = self.to.clone();
        for (f, t) in substitutions {
            new_to = new_to.replace_by_name(&f, &t)?;
        }
        Ok(Self::new(new_from, new_to))
    }

    fn generalize_to_fit(&self, statement: &SymbolNode) -> Result<Self, TransformationError> {
        let (relabelling, substitutions) = self
            .from
            .get_relabelling_and_leaf_substitutions(statement)
            .map_err(|e| Into::<TransformationError>::into(e))?;
        println!(
            "generalize_to_fit with ({:?}, {:?})",
            relabelling, substitutions
        );
        let mut new_from = self.from.clone();
        let mut new_to = self.to.clone();
        for (sub_from, sub_to) in substitutions {
            new_from = new_from.replace_all(&sub_from, &sub_to)?;
            new_to = new_to.replace_all(&sub_from, &sub_to)?;
        }
        println!("generalized to {:?} => {:?}", new_from, new_to);
        Ok(Self::new(new_from, new_to))
    }

    pub fn transform_all(
        &self,
        statement: &SymbolNode,
        relabellings: &HashMap<String, String>,
    ) -> Result<(SymbolNode, Vec<SymbolNodeAddress>), TransformationError> {
        self.transform_all_from_address(statement, relabellings, Vec::new())
    }

    pub fn transform_all_from_address(
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

        let new_statement = SymbolNode::new(statement.get_symbol().clone(), transformed_children);

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

    pub fn try_transform(
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

    pub fn transform_strict(
        &self,
        statement: &SymbolNode,
        relabellings: &HashMap<String, String>,
    ) -> Result<SymbolNode, TransformationError> {
        if self.get_variables() != relabellings.keys().cloned().collect() {
            return Err(TransformationError::RelabellingsKeysMismatch);
        }

        self.transform(statement, relabellings)
    }

    pub fn transform(
        &self,
        statement: &SymbolNode,
        relabellings: &HashMap<String, String>,
    ) -> Result<SymbolNode, TransformationError> {
        statement
            .validate()
            .map_err(|e| TransformationError::InvalidSymbolNode(e))?;

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
    InvalidSymbolNode(SymbolNodeError),
    InvalidTypes(TypeError),
    RelabellingsKeysMismatch,
    StatementDoesNotMatch(SymbolNode, SymbolNode),
    StatementTypesDoNotMatch,
    NoValidTransformations,
}

impl From<SymbolNodeError> for TransformationError {
    fn from(value: SymbolNodeError) -> Self {
        Self::InvalidSymbolNode(value)
    }
}

impl From<TypeError> for TransformationError {
    fn from(value: TypeError) -> Self {
        Self::InvalidTypes(value)
    }
}

#[cfg(test)]
mod test_transformation {
    use crate::parsing::{interpretation::Interpretation, tokenizer::Token};

    use super::*;

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

        let irrelevant_transform = Transformation::associativity(
            "*".to_string(),
            "*".into(),
            ("j".to_string(), "l".to_string(), "k".to_string()),
            "Irrelevant".into(),
        );

        let x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "x=y")
            .unwrap();

        assert_eq!(
            irrelevant_transform.get_valid_transformations(&hierarchy, &x_equals_y),
            vec![x_equals_y.clone()].into_iter().collect()
        );
        let transformation = Transformation::commutivity(
            "=".to_string(),
            "=".into(),
            ("a".to_string(), "b".to_string()),
            "Integer".into(),
        );
        let y_equals_x = parser
            .parse_from_string(custom_tokens.clone(), "y = x")
            .unwrap();
        assert_eq!(
            transformation.transform_at(&x_equals_y, vec![]),
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
        let transformation = Transformation::commutivity(
            "=".to_string(),
            "=".into(),
            ("x".to_string(), "y".to_string()),
            "Integer".into(),
        );

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
        let transformation = Transformation::new(
            SymbolNode::leaf_object("a".to_string()),
            SymbolNode::leaf_object("b".to_string()),
        );

        let statement = SymbolNode::leaf_object("c".to_string());

        let transformed = transformation.transform_strict(
            &statement,
            &vec![
                ("a".to_string(), "c".to_string()),
                ("b".to_string(), "d".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(transformed, Ok(SymbolNode::leaf_object("d".to_string())));

        let transformation = Transformation::new(
            SymbolNode::new(
                "=".into(),
                vec![
                    SymbolNode::leaf_object("a".to_string()),
                    SymbolNode::leaf_object("b".to_string()),
                ],
            ),
            SymbolNode::new(
                "=".into(),
                vec![
                    SymbolNode::leaf_object("b".to_string()),
                    SymbolNode::leaf_object("a".to_string()),
                ],
            ),
        );

        let self_equals_statement = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::leaf_object("a".to_string()),
            ],
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
        let commutativity = Transformation::new(
            SymbolNode::new(
                "=".into(),
                vec![
                    SymbolNode::leaf_object("a".to_string()),
                    SymbolNode::leaf_object("b".to_string()),
                ],
            ),
            SymbolNode::new(
                "=".into(),
                vec![
                    SymbolNode::leaf_object("b".to_string()),
                    SymbolNode::leaf_object("a".to_string()),
                ],
            ),
        );

        let statement = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![
                        SymbolNode::leaf_object("a".to_string()),
                        SymbolNode::leaf_object("b".to_string()),
                    ],
                ),
                SymbolNode::leaf_object("c".to_string()),
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
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("a".to_string()),
                    ],
                ),
                SymbolNode::leaf_object("c".to_string()),
            ],
        );
        assert_eq!(transformed, expected);
        assert_eq!(addresses, vec![vec![0]]);
    }

    #[test]
    fn test_transformation_typed_generalizes_to_fit() {
        let hierarchy = TypeHierarchy::chain(vec!["Integer".into(), "=".into()]).unwrap();
        let trivial_t = Transformation::new(
            SymbolNode::leaf_object("c".to_string()),
            SymbolNode::leaf_object("d".to_string()),
        );

        let trivial = SymbolNode::leaf_object("c".to_string());
        assert_eq!(
            trivial_t
                .typed_generalize_to_fit(&hierarchy, &trivial)
                .unwrap(),
            trivial_t
        );

        let different_t = Transformation::new(
            SymbolNode::leaf_object("a".to_string()),
            SymbolNode::leaf_object("d".to_string()),
        );

        let different_name = SymbolNode::leaf_object("a".to_string());
        assert_eq!(
            different_t
                .typed_generalize_to_fit(&hierarchy, &different_name)
                .unwrap(),
            different_t
        );

        let overloaded_t = Transformation::new(
            SymbolNode::leaf_object("d".to_string()),
            SymbolNode::leaf_object("d".to_string()),
        );
        let overloaded_name = SymbolNode::leaf_object("d".to_string());
        assert_eq!(
            overloaded_t
                .typed_generalize_to_fit(&hierarchy, &overloaded_name)
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

        let symmetry = Transformation::symmetry(
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
                .typed_generalize_to_fit(&hierarchy, &x_equals_y_equals_z)
                .unwrap(),
            Transformation::new(x_equals_y_equals_z.clone(), z_equals_x_equals_y.clone()),
            "\n\n{} \nvs. \n{}",
            symmetry
                .typed_generalize_to_fit(&hierarchy, &x_equals_y_equals_z)
                .unwrap()
                .to_string(),
            Transformation::new(x_equals_y_equals_z, z_equals_x_equals_y.clone()).to_string(),
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
    fn test_transformation_generalizes_to_fit() {
        let trivial_t = Transformation::new(
            SymbolNode::leaf_object("c".to_string()),
            SymbolNode::leaf_object("d".to_string()),
        );

        let trivial = SymbolNode::leaf_object("c".to_string());
        assert_eq!(trivial_t.generalize_to_fit(&trivial).unwrap(), trivial_t);

        let different_t = Transformation::new(
            SymbolNode::leaf_object("a".to_string()),
            SymbolNode::leaf_object("d".to_string()),
        );

        let different_name = SymbolNode::leaf_object("a".to_string());
        assert_eq!(
            different_t.generalize_to_fit(&different_name).unwrap(),
            different_t
        );

        let overloaded_t = Transformation::new(
            SymbolNode::leaf_object("d".to_string()),
            SymbolNode::leaf_object("d".to_string()),
        );
        let overloaded_name = SymbolNode::leaf_object("d".to_string());
        assert_eq!(
            overloaded_t.generalize_to_fit(&overloaded_name).unwrap(),
            overloaded_t
        );

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "Integer".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
        ];
        let parser = Parser::new(interpretations);

        let custom_tokens = vec!["=".to_string()];

        let symmetry = Transformation::symmetry(
            "=".to_string(),
            "=".into(),
            ("a".to_string(), "b".to_string()),
            "Integer".into(),
        );

        let x_equals_y_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=z")
            .unwrap();

        // We can't generalize because we don't know that "=" < "Integer" as a type
        assert!(symmetry.generalize_to_fit(&x_equals_y_equals_z).is_err(),);
    }

    #[test]
    fn test_transformation_typed_transforms_at() {
        let hierarchy =
            TypeHierarchy::chain(vec!["Real".into(), "Integer".into(), "=".into()]).unwrap();
        let transformation = Transformation::new(
            SymbolNode::leaf_object("c".to_string()),
            SymbolNode::leaf_object("d".to_string()),
        );

        let a_equals_b = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::leaf_object("b".to_string()),
            ],
        );

        let d_equals_b = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("d".to_string()),
                SymbolNode::leaf_object("b".to_string()),
            ],
        );

        let transformed = transformation.typed_transform_at(&hierarchy, &a_equals_b, vec![0]);

        assert_eq!(transformed, Ok(d_equals_b));

        let a_equals_b_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![
                        SymbolNode::leaf_object("a".to_string()),
                        SymbolNode::leaf_object("b".to_string()),
                    ],
                ),
                SymbolNode::leaf_object("c".to_string()),
            ],
        );

        let a_equals_d_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![
                        SymbolNode::leaf_object("a".to_string()),
                        SymbolNode::leaf_object("d".to_string()),
                    ],
                ),
                SymbolNode::leaf_object("c".to_string()),
            ],
        );
        let transformed =
            transformation.typed_transform_at(&hierarchy, &a_equals_b_equals_c, vec![0, 1]);

        assert_eq!(transformed, Ok(a_equals_d_equals_c));

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
        ];
        let parser = Parser::new(interpretations);

        let custom_tokens = vec!["=".to_string()];

        let transformation = Transformation::symmetry(
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
            transformation.typed_transform_at(&hierarchy, &x_equals_y_equals_z, vec![]),
            Ok(z_equals_x_equals_y)
        );

        let y_equals_x_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "(y=x)=z")
            .unwrap();

        assert_eq!(
            transformation.typed_transform_at(&hierarchy, &x_equals_y_equals_z, vec![0]),
            Ok(y_equals_x_equals_z)
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
        let transformation = Transformation::new(
            SymbolNode::leaf_object("c".to_string()),
            SymbolNode::leaf_object("d".to_string()),
        );

        let a_equals_b = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::leaf_object("b".to_string()),
            ],
        );

        let d_equals_b = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("d".to_string()),
                SymbolNode::leaf_object("b".to_string()),
            ],
        );

        let transformed = transformation.transform_at(&a_equals_b, vec![0]);

        assert_eq!(transformed, Ok(d_equals_b));

        let a_equals_b_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![
                        SymbolNode::leaf_object("a".to_string()),
                        SymbolNode::leaf_object("b".to_string()),
                    ],
                ),
                SymbolNode::leaf_object("c".to_string()),
            ],
        );

        let a_equals_d_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![
                        SymbolNode::leaf_object("a".to_string()),
                        SymbolNode::leaf_object("d".to_string()),
                    ],
                ),
                SymbolNode::leaf_object("c".to_string()),
            ],
        );
        let transformed = transformation.transform_at(&a_equals_b_equals_c, vec![0, 1]);

        assert_eq!(transformed, Ok(a_equals_d_equals_c));
    }

    #[test]
    fn test_transformation_reflexivity() {
        let transformation = Transformation::reflexivity(
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
        let transformation = Transformation::symmetry(
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
