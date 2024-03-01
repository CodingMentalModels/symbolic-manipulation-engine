use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub type TypeName = String;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeHierarchy {
    // Using a HashMap to quickly access nodes by their Type
    nodes: HashMap<Type, TypeHierarchyNode>,
}

impl TypeHierarchy {
    pub fn new() -> Self {
        let mut hierarchy = TypeHierarchy {
            nodes: HashMap::new(),
        };
        hierarchy.nodes.insert(
            Type::Object,
            TypeHierarchyNode {
                parent: None,
                children: vec![],
            },
        );
        hierarchy
    }

    pub fn chain(types: Vec<Type>) -> Result<Self, TypeError> {
        let mut hierarchy = Self::new();
        hierarchy.add_chain(types);
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
        match self.nodes.get(&type_to_add) {
            Some(node) => Err(TypeError::TypeHierarchyAlreadyIncludes(type_to_add)),
            None => {
                let node = TypeHierarchyNode {
                    parent: Some(parent_type.clone()),
                    children: vec![],
                };
                self.nodes.insert(type_to_add, node.clone());

                match self.nodes.get_mut(&parent_type) {
                    Some(parent_node) => {
                        parent_node.children.push(type_to_add);
                    }
                    None => return Err(TypeError::ParentNotFound(type_to_add)),
                }
                Ok(type_to_add)
            }
        }
    }

    pub fn is_subtype_of(&self, child: &Type, parent: &Type) -> bool {
        let mut current = child;
        while let Some(node) = self.nodes.get(current) {
            if let Some(ref parent_type) = node.parent {
                if parent_type == parent {
                    return true;
                }
                current = parent_type;
            } else {
                break;
            }
        }
        false
    }

    pub fn is_supertype_of(&self, parent: &Type, child: &Type) -> bool {
        self.is_subtype_of(child, parent)
    }

    pub fn are_pairwise_subtypes_of(
        &self,
        maybe_parents: Vec<Type>,
        maybe_children: Vec<Type>,
    ) -> bool {
        if maybe_parents.len() != maybe_children.len() {
            return false;
        }

        maybe_parents
            .iter()
            .zip(maybe_children.iter())
            .all(|(parent, child)| self.is_subtype_of(parent, child))
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeHierarchyNode {
    parent: Option<Type>,
    children: Vec<Type>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Type {
    Object,
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
            Type::NamedType(name) => name.clone(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeError {
    TypeHierarchyAlreadyIncludes(Type),
    ParentNotFound(Type),
}

#[cfg(test)]
mod test_type {
    use super::*;

    #[test]
    fn test_type_to_string() {
        assert_eq!(Type::Object.to_string(), "Object".to_string());

        let quaternion = Type::new("Quaternion".to_string());

        assert_eq!(quaternion.to_string(), "Quaternion");
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

        let mut type_hierarchy =
            TypeHierarchy::chain(vec![quaternion, complex, real, rational]).unwrap();
        type_hierarchy.add_child_to_parent(real, irrational);

        type_hierarchy.add_chain(vec![unary_function]);
        type_hierarchy.add_chain(vec![binary_function, plus]);

        assert_eq!(
            type_hierarchy.is_supertype_of(&quaternion, &quaternion),
            true
        );
        assert_eq!(type_hierarchy.is_supertype_of(&quaternion, &complex), true);
        assert_eq!(type_hierarchy.is_supertype_of(&quaternion, &real), true);
        assert_eq!(type_hierarchy.is_supertype_of(&quaternion, &rational), true);
        assert_eq!(
            type_hierarchy.is_supertype_of(&quaternion, &irrational),
            true
        );

        assert_eq!(
            type_hierarchy.is_supertype_of(&Type::Object, &quaternion),
            true
        );

        assert_eq!(
            type_hierarchy.is_supertype_of(&quaternion, &Type::Object),
            false
        );
        assert_eq!(
            type_hierarchy.is_supertype_of(&quaternion, &unary_function),
            false
        );

        assert_eq!(
            type_hierarchy.is_supertype_of(&irrational, &rational),
            false
        );
        assert_eq!(type_hierarchy.is_supertype_of(&real, &rational), true);
        assert_eq!(type_hierarchy.is_supertype_of(&real, &irrational), true);

        assert_eq!(type_hierarchy.is_supertype_of(&Type::Object, &plus), false);

        assert_eq!(type_hierarchy.is_supertype_of(&plus, &plus), true);
        assert_eq!(type_hierarchy.is_supertype_of(&plus, &quaternion), false);
    }
}
