use std::{
    collections::{HashMap, HashSet},
    unimplemented,
};

use serde::{Deserialize, Serialize};

use crate::parsing::interpretation::InterpretedType;

use super::{
    symbol_node::{Symbol, SymbolNode},
    transformation::Transformation,
};

pub type TypeName = String;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeHierarchy {
    type_map: HashMap<Type, TypeHierarchyNode>,
}

impl TypeHierarchy {
    pub fn new() -> Self {
        let mut hierarchy = TypeHierarchy {
            type_map: HashMap::new(),
        };
        hierarchy.type_map.insert(
            Type::Object,
            TypeHierarchyNode {
                inner: Type::Object,
                parents: HashSet::new(),
                children: HashSet::new(),
            },
        );
        hierarchy
    }

    pub fn chain(types: Vec<Type>) -> Result<Self, TypeError> {
        let mut hierarchy = Self::new();
        hierarchy.add_chain(types)?;
        Ok(hierarchy)
    }

    pub fn add_chain(&mut self, chain: Vec<Type>) -> Result<(), TypeError> {
        self.add_chain_to_parent(chain, Type::Object)
    }

    pub fn add_chain_to_parent(&mut self, chain: Vec<Type>, parent: Type) -> Result<(), TypeError> {
        let mut next_parent = parent;
        for t in chain {
            next_parent = self.add_child_to_parent(t, next_parent)?;
        }
        Ok(())
    }

    pub fn add_child_to_parent(
        &mut self,
        type_to_add: Type,
        parent_type: Type,
    ) -> Result<Type, TypeError> {
        self.add_child_to_parents(type_to_add, vec![parent_type].into_iter().collect())
    }

    pub fn add_child_to_parents(
        &mut self,
        type_to_add: Type,
        parent_types: HashSet<Type>,
    ) -> Result<Type, TypeError> {
        match self.type_map.get(&type_to_add) {
            Some(_node) => Err(TypeError::TypeHierarchyAlreadyIncludes(type_to_add)),
            None => {
                let node = TypeHierarchyNode {
                    inner: type_to_add.clone(),
                    parents: parent_types.clone(),
                    children: HashSet::new(),
                };
                self.type_map.insert(type_to_add.clone(), node.clone());

                for parent_type in parent_types {
                    match self.type_map.get_mut(&parent_type) {
                        Some(parent_node) => {
                            parent_node.children.insert(type_to_add.clone());
                        }
                        None => return Err(TypeError::ParentNotFound(type_to_add)),
                    }
                }
                Ok(type_to_add)
            }
        }
    }
    pub fn get_parent_child_pairs(&self) -> HashSet<(Type, Type)> {
        let mut parent_child_pairs = HashSet::new();

        for (child, node) in self.type_map.iter() {
            for parent in &node.parents {
                parent_child_pairs.insert((parent.clone(), child.clone()));
            }
        }

        parent_child_pairs
    }

    pub fn generalizes(&self, left: &SymbolNode, right: &SymbolNode) -> Result<bool, TypeError> {
        let root_generalizes = left.get_root_name() == right.get_root_name()
            && self.is_supertype_of(
                &left.get_evaluates_to_type(),
                &right.get_evaluates_to_type(),
            )?
            && left.get_children().len() == right.get_children().len();

        let children_generalize = left
            .get_children()
            .iter()
            .zip(right.get_children().iter())
            .try_fold(true, |acc, (x, y)| {
                self.generalizes(x, y).map(|result| acc && result)
            })?;

        Ok(root_generalizes && children_generalize)
    }

    pub fn is_generalized_by(
        &self,
        left: &SymbolNode,
        right: &SymbolNode,
    ) -> Result<bool, TypeError> {
        self.generalizes(right, left)
    }

    pub fn contains_type(&self, t: &Type) -> bool {
        self.type_map.contains_key(t)
    }

    fn type_exists_or_error(&self, t: &Type) -> Result<(), TypeError> {
        if self.contains_type(t) {
            Ok(())
        } else {
            Err(TypeError::InvalidType(t.clone()))
        }
    }

    pub fn is_subtype_of(&self, child: &Type, parent: &Type) -> Result<bool, TypeError> {
        self.type_exists_or_error(child)?;
        self.type_exists_or_error(parent)?;

        if child == parent {
            return Ok(true);
        }

        let mut visited = HashSet::new();
        let mut queue = vec![child.clone()];

        while let Some(current) = queue.pop() {
            if visited.contains(&current) {
                continue;
            }

            if let Some(node) = self.type_map.get(&current) {
                if node.parents.contains(parent) {
                    return Ok(true);
                }

                for parent_type in &node.parents {
                    if !visited.contains(parent_type) {
                        queue.push(parent_type.clone());
                    }
                }
            }

            visited.insert(current);
        }

        Ok(false)
    }

    pub fn is_supertype_of(&self, parent: &Type, child: &Type) -> Result<bool, TypeError> {
        self.is_subtype_of(child, parent)
    }

    pub fn are_pairwise_subtypes_of(
        &self,
        maybe_parents: Vec<Type>,
        maybe_children: Vec<Type>,
    ) -> Result<bool, TypeError> {
        if maybe_parents.len() != maybe_children.len() {
            return Ok(false);
        }

        maybe_parents
            .iter()
            .zip(maybe_children.iter())
            .try_fold(true, |acc, (parent, child)| {
                self.is_subtype_of(parent, child)
                    .map(|result| acc && result)
            })
    }

    pub fn binds_transformation_or_error(
        &self,
        transformation: &Transformation,
    ) -> Result<(), TypeError> {
        self.binds_statement_or_error(transformation.get_from())?;
        self.binds_statement_or_error(transformation.get_to())
    }

    pub fn binds_statement_or_error(&self, statement: &SymbolNode) -> Result<(), TypeError> {
        let missing_types: HashSet<_> = statement
            .get_types()
            .into_iter()
            .filter(|t| !self.contains_type(t))
            .collect();
        if missing_types.len() == 0 {
            Ok(())
        } else {
            Err(TypeError::StatementIncludesTypesNotInHierarchy(
                missing_types,
            ))
        }
    }

    pub fn get_types(&self) -> HashSet<Type> {
        self.type_map.iter().map(|(t, n)| t).cloned().collect()
    }

    pub fn get_shared_types(&self, other: &Self) -> HashSet<Type> {
        self.get_types()
            .intersection(&other.get_types())
            .cloned()
            .collect()
    }

    pub fn union(&self, other: &TypeHierarchy) -> Result<Self, TypeError> {
        Self::are_compatible_or_error(self, other)?;

        let mut new_hierarchy = self.clone();

        for (other_type, other_node) in other.type_map.iter() {
            if !new_hierarchy.type_map.contains_key(other_type) {
                println!("Unioning in unconflicted {:?}", other_type.clone());
                new_hierarchy
                    .type_map
                    .insert(other_type.clone(), other_node.clone());
            } else {
                let existing_node = new_hierarchy.type_map.get_mut(other_type).unwrap();
                for other_parent in &other_node.parents {
                    // If the parent doesn't exist or isn't already in the lineage of the inner
                    // node, we add it
                    if self.is_subtype_of(&existing_node.inner, other_parent) != Ok(true) {
                        println!(
                            "Inserting parent {:?} into {:?}",
                            other_parent.clone(),
                            existing_node.inner.to_string()
                        );
                        existing_node.parents.insert(other_parent.clone());
                    }
                }

                let existing_node = new_hierarchy.type_map.get_mut(other_type).unwrap();
                let existing_children = &mut existing_node.children;

                for other_child in &other_node.children {
                    // If the child doesn't exist or isn't already in the lineage of the inner
                    // node, we add it
                    if self.is_supertype_of(&existing_node.inner, other_child) != Ok(true) {
                        println!(
                            "Inserting child {:?} into {:?}",
                            other_child.clone(),
                            existing_node.inner.to_string()
                        );
                        existing_children.insert(other_child.clone());
                    }
                }
            }
        }
        new_hierarchy.prune_redundant_relationships();

        Ok(new_hierarchy)
    }

    fn prune_redundant_relationships(&mut self) {
        let mut to_remove = Vec::new();

        for (t, node) in self.type_map.iter() {
            let children = node.children.clone();
            for child in children.iter() {
                for sibling in children.iter() {
                    if child != sibling && self.is_subtype_of(child, sibling).unwrap_or(false) {
                        to_remove.push((t.clone(), child.clone()));
                        break;
                    }
                }
            }
        }

        for (parent, child) in to_remove {
            if let Some(node) = self.type_map.get_mut(&parent) {
                node.children.retain(|c| c != &child);
            }
            if let Some(node) = self.type_map.get_mut(&child) {
                node.parents.retain(|p| p != &parent);
            }
        }
    }

    fn are_compatible_or_error(h1: &TypeHierarchy, h2: &TypeHierarchy) -> Result<(()), TypeError> {
        let left_to_right = Self::is_left_compatible_with_right(h1, h2);
        let right_to_left = Self::is_left_compatible_with_right(h2, h1);

        match (left_to_right, right_to_left) {
            (Ok(()), Ok(())) => Ok(()),
            (
                Err(TypeError::IncompatibleTypeRelationships(e)),
                Err(TypeError::IncompatibleTypeRelationships(mut f)),
            ) => {
                let to_return = e;
                let to_return = to_return.union(&mut f).cloned().collect();
                Err(TypeError::IncompatibleTypeRelationships(to_return))
            }
            (Err(e), _) => Err(e),
            (_, Err(f)) => Err(f),
        }
    }

    fn is_left_compatible_with_right(
        h1: &TypeHierarchy,
        h2: &TypeHierarchy,
    ) -> Result<(), TypeError> {
        let mut conflicts = HashSet::new();

        for (child, node) in h1.type_map.iter() {
            for parent in &node.parents {
                if h2.type_map.contains_key(child) && h2.type_map.contains_key(parent) {
                    let is_subtype_in_h2 = h2.is_subtype_of(child, parent).unwrap_or(false);
                    if !is_subtype_in_h2 {
                        conflicts.insert(child.clone());
                    }
                }
            }
        }

        if conflicts.is_empty() {
            Ok(())
        } else {
            Err(TypeError::IncompatibleTypeRelationships(conflicts))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeHierarchyNode {
    inner: Type,
    parents: HashSet<Type>,
    children: HashSet<Type>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedType {
    condition: GeneratedTypeCondition,
    parents: HashSet<Type>,
}

impl GeneratedType {
    pub fn new(condition: GeneratedTypeCondition, parents: HashSet<Type>) -> Self {
        Self { condition, parents }
    }

    pub fn generate(&self, statement: &SymbolNode) -> Vec<(Type, HashSet<Type>)> {
        let mut to_return: Vec<(Type, HashSet<Type>)> = statement
            .get_children()
            .iter()
            .map(|child| self.generate(child))
            .flatten()
            .collect();
        if self.satisfies_condition(statement.get_symbol()) {
            to_return.push((
                statement.get_symbol().get_name().into(),
                self.parents.clone(),
            ));
        }
        to_return
    }

    pub fn satisfies_condition(&self, symbol: &Symbol) -> bool {
        self.condition.is_satisfied_by(symbol)
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeneratedTypeCondition {
    IsInteger,
    IsNumeric,
}

impl GeneratedTypeCondition {
    pub fn is_satisfied_by(&self, symbol: &Symbol) -> bool {
        match self {
            Self::IsInteger => {
                // Todo: This will fail on big enough numbers
                return symbol.get_name().parse::<i64>().is_ok();
            }
            Self::IsNumeric => {
                // Todo: This will fail on big enough numbers
                return symbol.get_name().parse::<f64>().is_ok();
            }
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Type {
    Object,
    Delimiter,
    NamedType(TypeName),
}

impl Default for Type {
    fn default() -> Type {
        Type::Object
    }
}

impl From<&str> for Type {
    fn from(value: &str) -> Self {
        Self::from(value.to_string())
    }
}

impl From<String> for Type {
    fn from(value: String) -> Self {
        Self::NamedType(value)
    }
}

impl Type {
    pub fn new(name: TypeName) -> Type {
        Type::NamedType(name)
    }

    pub fn to_string(&self) -> String {
        match self {
            Type::Object => "Object".to_string(),
            Type::Delimiter => "Delimiter".to_string(),
            Type::NamedType(name) => name.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeError {
    TypeHierarchyAlreadyIncludes(Type),
    ParentNotFound(Type),
    InvalidType(Type),
    StatementIncludesTypesNotInHierarchy(HashSet<Type>),
    IncompatibleTypeRelationships(HashSet<Type>),
}

#[cfg(test)]
mod test_type {
    use crate::{
        parsing::{interpretation::Interpretation, parser::Parser},
        symbol::symbol_node::Symbol,
    };

    use super::*;

    #[test]
    fn test_type_to_string() {
        assert_eq!(Type::Object.to_string(), "Object".to_string());

        let quaternion = Type::new("Quaternion".to_string());

        assert_eq!(quaternion.to_string(), "Quaternion");
    }

    #[test]
    fn test_generalizes_and_is_generalized_by() {
        let mut type_hierarchy = TypeHierarchy::chain(vec!["Integer".into()]).unwrap();
        type_hierarchy.add_chain(vec!["Boolean".into()]);
        let a_equals_b = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::leaf_object("b".to_string()),
            ],
        );

        assert!(type_hierarchy
            .generalizes(&a_equals_b, &a_equals_b)
            .unwrap());
        assert!(type_hierarchy
            .is_generalized_by(&a_equals_b, &a_equals_b)
            .unwrap());

        let a_equals_b_integers = SymbolNode::new(
            Symbol::new("=".to_string(), Type::new("Boolean".to_string())),
            vec![
                SymbolNode::leaf(Symbol::new(
                    "a".to_string(),
                    Type::new("Integer".to_string()),
                )),
                SymbolNode::leaf(Symbol::new(
                    "b".to_string(),
                    Type::new("Integer".to_string()),
                )),
            ],
        );

        assert!(!type_hierarchy
            .generalizes(&a_equals_b_integers, &a_equals_b)
            .unwrap());
        assert!(type_hierarchy
            .generalizes(&a_equals_b, &a_equals_b_integers)
            .unwrap());
    }
    #[test]
    fn test_type_hierarchy_is_supertype_of() {
        let quaternion = Type::new("Quaternion".to_string());
        let complex = Type::new("Complex".to_string());
        let real = Type::new("Real".to_string());
        let rational = Type::new("Rational".to_string());
        let irrational = Type::new("Irrational".to_string());
        let unary_function = Type::new("UnaryFunction".to_string());
        let binary_function = Type::new("BinaryFunction".to_string());
        let plus = Type::new("Plus".to_string());

        let mut type_hierarchy = TypeHierarchy::chain(vec![
            quaternion.clone(),
            complex.clone(),
            real.clone(),
            rational.clone(),
        ])
        .unwrap();

        type_hierarchy.add_child_to_parent(irrational.clone(), real.clone());

        type_hierarchy.add_chain(vec![unary_function.clone()]);
        type_hierarchy.add_chain(vec![binary_function.clone(), plus.clone()]);

        assert_eq!(
            type_hierarchy
                .is_supertype_of(&quaternion, &quaternion)
                .unwrap(),
            true
        );
        assert_eq!(
            type_hierarchy
                .is_supertype_of(&quaternion, &complex)
                .unwrap(),
            true
        );
        assert_eq!(
            type_hierarchy.is_supertype_of(&quaternion, &real).unwrap(),
            true
        );
        assert_eq!(
            type_hierarchy
                .is_supertype_of(&quaternion, &rational)
                .unwrap(),
            true
        );
        assert_eq!(
            type_hierarchy
                .is_supertype_of(&quaternion, &irrational)
                .unwrap(),
            true
        );

        assert_eq!(
            type_hierarchy
                .is_supertype_of(&Type::Object, &quaternion)
                .unwrap(),
            true
        );

        assert_eq!(
            type_hierarchy
                .is_supertype_of(&quaternion, &Type::Object)
                .unwrap(),
            false
        );
        assert_eq!(
            type_hierarchy
                .is_supertype_of(&quaternion, &unary_function)
                .unwrap(),
            false
        );

        assert_eq!(
            type_hierarchy
                .is_supertype_of(&irrational, &rational)
                .unwrap(),
            false
        );
        assert_eq!(
            type_hierarchy.is_supertype_of(&real, &rational).unwrap(),
            true
        );
        assert_eq!(
            type_hierarchy.is_supertype_of(&real, &irrational).unwrap(),
            true
        );

        assert_eq!(
            type_hierarchy
                .is_supertype_of(&Type::Object, &plus)
                .unwrap(),
            true
        );

        assert_eq!(type_hierarchy.is_supertype_of(&plus, &plus).unwrap(), true);
        assert_eq!(
            type_hierarchy.is_supertype_of(&plus, &quaternion).unwrap(),
            false
        );
    }

    #[test]
    fn test_type_hierarchy_unions() {
        let mut trivial = TypeHierarchy::new();
        assert_eq!(trivial.union(&trivial), Ok(trivial.clone()));

        let mut chain =
            TypeHierarchy::chain(vec!["Real".into(), "Rational".into(), "Integer".into()]).unwrap();

        assert_eq!(trivial.union(&chain), chain.union(&trivial));
        assert_eq!(trivial.union(&chain), Ok(chain.clone()));

        assert_eq!(chain.union(&chain), Ok(chain.clone()));

        let mut chain_with_complex = TypeHierarchy::chain(vec![
            "Complex".into(),
            "Real".into(),
            "Rational".into(),
            "Integer".into(),
        ])
        .unwrap();

        assert_eq!(
            chain_with_complex.union(&chain),
            chain.union(&chain_with_complex)
        );
        assert_eq!(
            chain_with_complex.union(&chain),
            Ok(chain_with_complex.clone())
        );
    }

    #[test]
    fn test_type_hierarchy_are_compatible() {
        let mut trivial = TypeHierarchy::new();
        assert_eq!(
            TypeHierarchy::are_compatible_or_error(&trivial, &trivial),
            Ok(())
        );

        let mut chain =
            TypeHierarchy::chain(vec!["Real".into(), "Rational".into(), "Integer".into()]).unwrap();

        assert_eq!(
            TypeHierarchy::are_compatible_or_error(&chain, &chain),
            Ok(())
        );

        assert_eq!(
            TypeHierarchy::are_compatible_or_error(&trivial, &chain),
            Ok(())
        );

        assert_eq!(
            TypeHierarchy::are_compatible_or_error(&chain, &trivial),
            Ok(())
        );

        let mut chain_with_complex = TypeHierarchy::chain(vec![
            "Complex".into(),
            "Real".into(),
            "Rational".into(),
            "Integer".into(),
        ])
        .unwrap();

        assert_eq!(
            TypeHierarchy::are_compatible_or_error(&chain_with_complex, &chain),
            Ok(())
        );

        assert_eq!(
            TypeHierarchy::are_compatible_or_error(&chain, &chain_with_complex),
            Ok(())
        );

        let mut chain_missing_rational =
            TypeHierarchy::chain(vec!["Real".into(), "Integer".into()]).unwrap();

        assert_eq!(
            TypeHierarchy::are_compatible_or_error(&chain_with_complex, &chain_missing_rational),
            Ok(())
        );

        assert_eq!(
            TypeHierarchy::are_compatible_or_error(&chain_missing_rational, &chain_with_complex),
            Ok(())
        );

        let mut inverted_chain_missing_rational =
            TypeHierarchy::chain(vec!["Integer".into(), "Real".into()]).unwrap();

        assert_eq!(
            TypeHierarchy::are_compatible_or_error(
                &inverted_chain_missing_rational,
                &chain_missing_rational
            ),
            Err(TypeError::IncompatibleTypeRelationships(
                vec!["Integer".into(), "Real".into()].into_iter().collect()
            ))
        );

        assert_eq!(
            TypeHierarchy::are_compatible_or_error(
                &chain_missing_rational,
                &inverted_chain_missing_rational
            ),
            Err(TypeError::IncompatibleTypeRelationships(
                vec!["Integer".into(), "Real".into()].into_iter().collect()
            ))
        );
    }
}
