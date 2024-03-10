use std::{
    collections::{HashMap, HashSet},
    unimplemented,
};

use serde::{Deserialize, Serialize};

use crate::symbol::symbol_type::Type;

pub type SymbolName = String;
pub type SymbolNodeAddress = Vec<usize>;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum SymbolNodeError {
    ConflictingTypes(String, Type, Type),
    ConflictingSymbolArities(Symbol),
    DifferentNumberOfArguments,
    RelabellingNotInjective,
    InvalidAddress,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolNode {
    root: Symbol,
    children: Vec<SymbolNode>,
}

impl SymbolNode {
    pub fn new(root: Symbol, children: Vec<SymbolNode>) -> Self {
        SymbolNode { root, children }
    }

    pub fn leaf(root: Symbol) -> Self {
        Self::new(root, Vec::new())
    }

    pub fn with_single_child(root: Symbol, child: Symbol) -> Self {
        Self::new(root, vec![SymbolNode::leaf(child)])
    }

    pub fn leaf_object(root: String) -> Self {
        Self::leaf(Symbol::new_object(root))
    }

    pub fn object_with_single_child_object(root: String, child: String) -> Self {
        Self::with_single_child(Symbol::new_object(root), Symbol::new_object(child))
    }

    pub fn get_depth(&self) -> usize {
        if self.children.len() == 0 {
            1
        } else {
            1 + self
                .children
                .iter()
                .map(|child| child.get_depth())
                .max()
                .unwrap()
        }
    }

    pub fn get_children(&self) -> &Vec<SymbolNode> {
        &self.children
    }

    pub fn get_n_children(&self) -> usize {
        self.children.len()
    }

    pub fn push_child(&mut self, child: SymbolNode) {
        self.children.push(child)
    }

    pub fn get_symbol(&self) -> &Symbol {
        &self.root
    }

    pub fn get_root_name(&self) -> String {
        self.get_symbol().get_name()
    }

    pub fn get_evaluates_to_type(&self) -> Type {
        self.root.get_evaluates_to_type()
    }

    pub fn has_conflicting_arities(&self) -> bool {
        match self.get_arities() {
            Err(SymbolNodeError::ConflictingSymbolArities(_)) => true,
            _ => false,
        }
    }

    pub fn get_arities(&self) -> Result<HashMap<Symbol, usize>, SymbolNodeError> {
        let mut arities = HashMap::new();
        self.collect_arities(&mut arities)?;
        Ok(arities)
    }

    fn collect_arities(&self, arities: &mut HashMap<Symbol, usize>) -> Result<(), SymbolNodeError> {
        // Check if the symbol already exists with a different arity
        match arities.entry(self.root.clone()) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(self.get_n_children());
            }
            std::collections::hash_map::Entry::Occupied(mut e) => {
                // If the arity (number of children) is different, return an error
                if *e.get() != self.get_n_children() {
                    return Err(SymbolNodeError::ConflictingSymbolArities(self.root.clone()));
                }
            }
        }

        // Recursively collect arities from children
        for child in &self.children {
            child.collect_arities(arities)?;
        }

        Ok(())
    }

    pub fn split_delimiters(&self) -> Self {
        let mut to_return = Self::leaf(self.root.clone());
        for child in self.children.iter() {
            to_return.push_child(child.split_delimiters());
        }
        let splits = to_return.split_if_delimiter();
        if splits.len() == 1 {
            splits[0].clone()
        } else {
            Self::new(self.root.clone(), splits)
        }
    }

    pub fn split_if_delimiter(&self) -> Vec<Self> {
        if self.get_evaluates_to_type() == Type::Delimiter {
            self.split_self()
        } else {
            vec![self.clone()]
        }
    }

    pub fn split_self(&self) -> Vec<Self> {
        self.split(self.get_symbol())
    }

    pub fn split(&self, symbol: &Symbol) -> Vec<Self> {
        if self.get_symbol() == symbol {
            let mut to_return = vec![];
            for child in self.children.iter() {
                to_return.append(&mut child.split(symbol));
            }
            to_return
        } else {
            vec![self.clone()]
        }
    }

    pub fn collapse_delimiters(&self) -> Self {
        let mut new_children = vec![];
        for child in self.children.iter() {
            let new_child = child.collapse_delimiters();
            if new_child.get_evaluates_to_type() == Type::Delimiter {
                new_children.append(&mut new_child.children.clone());
            } else {
                new_children.push(new_child);
            }
        }
        Self::new(self.root.clone(), new_children)
    }

    pub fn get_node(&self, address: SymbolNodeAddress) -> Option<Self> {
        let mut current_node = self.clone();
        for i in address {
            if i < current_node.get_n_children() {
                current_node = current_node.children[i].clone();
            } else {
                return None;
            }
        }
        Some(current_node)
    }

    pub fn replace_node(
        &self,
        address: SymbolNodeAddress,
        new_node: SymbolNode,
    ) -> Result<Self, SymbolNodeError> {
        let mut to_return = self.clone();
        let mut current_node = &mut to_return;
        for i in address {
            if i < current_node.get_n_children() {
                current_node = &mut current_node.children[i];
            } else {
                return Err(SymbolNodeError::InvalidAddress);
            }
        }
        *current_node = new_node;

        Ok(to_return)
    }

    pub fn find_where(&self, condition: &dyn Fn(Self) -> bool) -> HashSet<SymbolNodeAddress> {
        let mut result = HashSet::new();
        if condition(self.clone()) {
            result.insert(Vec::new());
        }
        for (i, child) in self.children.iter().enumerate() {
            let child_result = child.find_where(&condition);
            for address in child_result {
                let mut new_address = address.clone();
                new_address.push(i);
                result.insert(new_address);
            }
        }
        result
    }

    pub fn find_symbol(&self, symbol: Symbol) -> HashSet<SymbolNodeAddress> {
        self.find_where(&|node| node.root == symbol)
    }

    pub fn find_symbol_name(&self, symbol_name: String) -> HashSet<SymbolNodeAddress> {
        self.find_where(&|node| node.root.get_name() == symbol_name)
    }

    pub fn to_string(&self) -> String {
        if self.children.len() == 0 {
            self.root.to_string()
        } else {
            let mut result = format!("{}(", self.root.to_string());
            let arguments = self
                .children
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",");
            result = format!("{}{})", result, arguments);
            result
        }
    }

    pub fn get_symbols(&self) -> HashSet<Symbol> {
        let mut result = vec![self.root.clone()];
        for child in &self.children {
            result.extend(child.get_symbols());
        }
        result.into_iter().collect()
    }

    pub fn to_symbol_string(&self) -> String {
        if self.children.len() == 0 {
            self.root.get_name()
        } else {
            let mut result = format!("{}(", self.root.get_name());
            let arguments = self
                .children
                .iter()
                .map(|x| x.to_symbol_string())
                .collect::<Vec<_>>()
                .join(",");
            result = format!("{}{})", result, arguments);
            result
        }
    }

    pub fn relabel(&self, old_label: String, new_label: String) -> Self {
        self.relabel_and_get_addresses_if(old_label, new_label, Vec::new(), &|x| true)
            .0
    }

    fn relabel_and_get_addresses_if(
        &self,
        old_label: String,
        new_label: String,
        current_address: SymbolNodeAddress,
        condition: &dyn Fn((Self, SymbolNodeAddress)) -> bool,
    ) -> (Self, HashSet<SymbolNodeAddress>) {
        let (new_children, children_addresses) = self
            .children
            .iter()
            .enumerate()
            .map(|(i, child)| {
                let mut child_address = current_address.clone();
                child_address.push(i);
                child.relabel_and_get_addresses_if(
                    old_label.clone(),
                    new_label.clone(),
                    child_address,
                    condition,
                )
            })
            .fold((Vec::new(), HashSet::new()), |acc, elt| {
                let (new_children, new_children_addresses) = acc;
                let (new_child, new_child_addresses) = elt;
                (
                    new_children.into_iter().chain(vec![new_child]).collect(),
                    new_children_addresses
                        .into_iter()
                        .chain(new_child_addresses)
                        .collect(),
                )
            });
        if self.root.get_name() == old_label && condition((self.clone(), current_address.clone())) {
            let mut addresses = children_addresses;
            addresses.insert(current_address);
            (
                Self::new(
                    Symbol::new(new_label, self.root.get_evaluates_to_type()),
                    new_children,
                ),
                addresses,
            )
        } else {
            (
                Self::new(self.root.clone(), new_children),
                children_addresses,
            )
        }
    }

    pub fn relabel_all(&self, relabelling: HashSet<(String, String)>) -> Self {
        relabelling
            .into_iter()
            .fold(
                (self.clone(), HashSet::new()),
                |acc, (old_label, new_label)| {
                    let (new_tree, addresses) = acc;
                    new_tree.relabel_and_get_addresses_if(old_label, new_label, Vec::new(), &|x| {
                        !addresses.contains(&x.1)
                    })
                },
            )
            .0
    }

    pub fn get_relabelling(
        &self,
        other: &Self,
    ) -> Result<HashMap<String, String>, SymbolNodeError> {
        if self.children.len() != other.children.len() {
            return Err(SymbolNodeError::DifferentNumberOfArguments);
        }
        let mut to_return = vec![(self.root.get_name(), other.root.get_name())];
        for (i, (child, other_child)) in self.children.iter().zip(other.children.iter()).enumerate()
        {
            let child_relabelling = child.get_relabelling(other_child)?;
            for (old_label, new_label) in child_relabelling {
                to_return.push((old_label, new_label));
            }
        }

        if to_return
            .iter()
            .map(|x| x.0.clone())
            .collect::<HashSet<_>>()
            .len()
            != to_return.len()
        {
            return Err(SymbolNodeError::RelabellingNotInjective);
        }

        Ok(to_return.into_iter().collect())
    }

    pub fn validate(&self) -> Result<(), SymbolNodeError> {
        self.get_type_map()?;
        self.get_arities()?;
        Ok(())
    }

    pub fn get_type_map(&self) -> Result<HashMap<String, Type>, SymbolNodeError> {
        let children_type_map =
            self.children
                .iter()
                .try_fold(HashMap::new(), |mut acc, child| {
                    let child_types = child.get_type_map()?;
                    for (name, t) in child_types.iter() {
                        match acc.get(name) {
                            None => {
                                acc.insert(name.clone(), t.clone());
                            }
                            Some(prior_type) => {
                                if prior_type != t {
                                    return Err(SymbolNodeError::ConflictingTypes(
                                        name.clone(),
                                        t.clone(),
                                        prior_type.clone(),
                                    ));
                                }
                            }
                        }
                    }
                    Ok(acc)
                })?;
        let mut to_return = children_type_map.clone();
        match children_type_map.get(&self.get_root_name()) {
            Some(prior_type) => {
                if &self.get_evaluates_to_type() != prior_type {
                    return Err(SymbolNodeError::ConflictingTypes(
                        self.get_root_name(),
                        self.get_evaluates_to_type(),
                        prior_type.clone(),
                    ));
                }
            }
            None => {
                to_return.insert(self.get_root_name(), self.get_evaluates_to_type());
            }
        }
        Ok(to_return)
    }
}

#[derive(Clone, Debug, Default, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symbol {
    name: SymbolName,
    evaluates_to_type: Type,
}

impl From<&str> for Symbol {
    fn from(value: &str) -> Self {
        Self::from(value.to_string())
    }
}

impl From<String> for Symbol {
    fn from(value: String) -> Self {
        Self::new(value, Type::Object)
    }
}

impl Symbol {
    pub fn new(name: SymbolName, symbol_type: Type) -> Self {
        Self {
            name,
            evaluates_to_type: symbol_type,
        }
    }

    pub fn new_object(name: SymbolName) -> Self {
        Self::new(name, Type::default())
    }

    pub fn delimiter(name: SymbolName) -> Self {
        Self::new(name, Type::Delimiter)
    }

    pub fn get_name(&self) -> SymbolName {
        self.name.clone()
    }

    pub fn get_evaluates_to_type(&self) -> Type {
        self.evaluates_to_type.clone()
    }

    pub fn to_string(&self) -> String {
        format!("{}: {}", self.name, self.evaluates_to_type.to_string())
    }
}

#[cfg(test)]
mod test_statement {
    use super::*;

    #[test]
    fn test_symbol_node_initializes() {
        let a_equals_b_plus_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new(
                    "+".into(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("c".to_string()),
                    ],
                ),
            ],
        );

        assert_eq!(a_equals_b_plus_c.get_depth(), 3);
        assert_eq!(a_equals_b_plus_c.get_n_children(), 2);
        assert_eq!(a_equals_b_plus_c.to_symbol_string(), "=(a,+(b,c))");

        let n_factorial =
            SymbolNode::object_with_single_child_object("!".to_string(), "n".to_string());

        // prod_{i = 1}^{n} i
        // Point: Expand 5! to 5 * 4 * 3 * 2 * 1, which is going to require a transformation like:
        // 5! => prod_{i = 1}^{5} i => 5 * 4 * 3 * 2 * 1
        let n_factorial_definition = SymbolNode::new(
            "Prod".into(),
            vec![
                SymbolNode::leaf_object("i".to_string()), // i is the index variable
                SymbolNode::leaf_object("1".to_string()), // 1 is the lower bound
                SymbolNode::leaf_object("n".to_string()), // n is the upper bound
                SymbolNode::leaf_object("i".to_string()), // i is the expression to be multiplied
            ],
        );

        let factorial_definition =
            SymbolNode::new("=".into(), vec![n_factorial, n_factorial_definition]);

        assert_eq!(factorial_definition.get_depth(), 3);
    }

    #[test]
    fn test_symbol_nodes_relabel() {
        let a_equals_b_plus_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new(
                    "+".into(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("c".to_string()),
                    ],
                ),
            ],
        );

        let x_equals_b_plus_c = a_equals_b_plus_c.relabel("a".to_string(), "x".to_string());
        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("x".to_string()),
                SymbolNode::new(
                    "+".into(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("c".to_string()),
                    ],
                ),
            ],
        );
        assert_eq!(x_equals_b_plus_c, expected);

        let x_equals_y_plus_y = x_equals_b_plus_c
            .relabel("b".to_string(), "y".to_string())
            .relabel("c".to_string(), "y".to_string());
        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("x".to_string()),
                SymbolNode::new(
                    "+".into(),
                    vec![
                        SymbolNode::leaf_object("y".to_string()),
                        SymbolNode::leaf_object("y".to_string()),
                    ],
                ),
            ],
        );
        assert_eq!(x_equals_y_plus_y, expected);

        let x_equals_x_plus_x = x_equals_y_plus_y.relabel("y".to_string(), "x".to_string());
        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("x".to_string()),
                SymbolNode::new(
                    "+".into(),
                    vec![
                        SymbolNode::leaf_object("x".to_string()),
                        SymbolNode::leaf_object("x".to_string()),
                    ],
                ),
            ],
        );
        assert_eq!(x_equals_x_plus_x, expected);

        let also_x_equals_x_plus_x = a_equals_b_plus_c.relabel_all(
            vec![
                ("b".to_string(), "x".to_string()),
                ("c".to_string(), "x".to_string()),
                ("a".to_string(), "x".to_string()),
            ]
            .into_iter()
            .collect(),
        );

        assert_eq!(x_equals_x_plus_x, also_x_equals_x_plus_x);

        let a_equals_c_plus_b = a_equals_b_plus_c.relabel_all(
            vec![
                ("b".to_string(), "c".to_string()),
                ("c".to_string(), "b".to_string()),
            ]
            .into_iter()
            .collect(),
        );

        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new(
                    "+".into(),
                    vec![
                        SymbolNode::leaf_object("c".to_string()),
                        SymbolNode::leaf_object("b".to_string()),
                    ],
                ),
            ],
        );

        assert_eq!(a_equals_c_plus_b, expected);
    }

    #[test]
    fn test_symbol_node_splits_delimiters() {
        let trivial_node = SymbolNode::leaf(Symbol::new_object("x".to_string()));
        assert_eq!(trivial_node, trivial_node.split_delimiters());

        let no_delimiters = SymbolNode::new(
            "+".into(),
            vec![
                SymbolNode::leaf(Symbol::delimiter("a".to_string())),
                SymbolNode::new(
                    "+".into(),
                    vec![
                        SymbolNode::leaf(Symbol::delimiter("b".to_string())),
                        SymbolNode::leaf(Symbol::delimiter("c".to_string())),
                    ],
                ),
            ],
        );

        assert_eq!(no_delimiters.split_delimiters(), no_delimiters);

        let a_plus_b_plus_c = SymbolNode::new(
            Symbol::delimiter("+".to_string()),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new(
                    Symbol::delimiter("+".to_string()),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("c".to_string()),
                    ],
                ),
            ],
        );

        let expected = SymbolNode::new(
            Symbol::delimiter("+".to_string()),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::leaf_object("b".to_string()),
                SymbolNode::leaf_object("c".to_string()),
            ],
        );

        assert_eq!(a_plus_b_plus_c.split_delimiters(), expected);

        let x_plus_y = SymbolNode::new(
            "+".into(),
            vec![
                SymbolNode::leaf(Symbol::new_object("x".to_string())),
                SymbolNode::leaf(Symbol::new_object("y".to_string())),
            ],
        );

        let x_plus_y_equals_a_plus_b_plus_c =
            SymbolNode::new("=".into(), vec![x_plus_y.clone(), a_plus_b_plus_c]);

        let expected = SymbolNode::new("=".into(), vec![x_plus_y, expected]);

        assert_eq!(x_plus_y_equals_a_plus_b_plus_c.split_delimiters(), expected);
    }

    #[test]
    fn test_symbol_node_splits() {
        let a_plus_b_plus_c = SymbolNode::new(
            "+".into(),
            vec![
                SymbolNode::leaf(Symbol::new_object("a".to_string())),
                SymbolNode::new(
                    "+".into(),
                    vec![
                        SymbolNode::leaf(Symbol::new_object("b".to_string())),
                        SymbolNode::leaf(Symbol::new_object("c".to_string())),
                    ],
                ),
            ],
        );

        let expected = vec![
            SymbolNode::leaf(Symbol::new_object("a".to_string())),
            SymbolNode::leaf(Symbol::new_object("b".to_string())),
            SymbolNode::leaf(Symbol::new_object("c".to_string())),
        ];

        assert_eq!(
            a_plus_b_plus_c.split(&Symbol::new_object("+".to_string())),
            expected
        );
    }

    #[test]
    fn test_symbol_node_collapses_delimiters() {
        let trivial_node = SymbolNode::leaf(Symbol::new_object("x".to_string()));
        assert_eq!(trivial_node, trivial_node.collapse_delimiters());

        let no_delimiters = SymbolNode::new(
            "+".into(),
            vec![
                SymbolNode::leaf(Symbol::new_object("a".to_string())),
                SymbolNode::new(
                    "+".into(),
                    vec![
                        SymbolNode::leaf(Symbol::new_object("b".to_string())),
                        SymbolNode::leaf(Symbol::new_object("c".to_string())),
                    ],
                ),
            ],
        );

        assert_eq!(no_delimiters.collapse_delimiters(), no_delimiters);

        let a_plus_b_plus_c = SymbolNode::new(
            Symbol::delimiter("+".to_string()),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new(
                    Symbol::delimiter("+".to_string()),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("c".to_string()),
                    ],
                ),
            ],
        );

        let expected = SymbolNode::new(
            Symbol::delimiter("+".to_string()),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::leaf_object("b".to_string()),
                SymbolNode::leaf_object("c".to_string()),
            ],
        );

        assert_eq!(a_plus_b_plus_c.collapse_delimiters(), expected);

        let x_plus_y = SymbolNode::new(
            Symbol::delimiter("+".to_string()),
            vec![
                SymbolNode::leaf(Symbol::new_object("x".to_string())),
                SymbolNode::leaf(Symbol::new_object("y".to_string())),
            ],
        );

        let x_plus_y_equals_a_plus_b_plus_c =
            SymbolNode::new("=".into(), vec![x_plus_y.clone(), a_plus_b_plus_c]);

        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf(Symbol::new_object("x".to_string())),
                SymbolNode::leaf(Symbol::new_object("y".to_string())),
                SymbolNode::leaf(Symbol::new_object("a".to_string())),
                SymbolNode::leaf(Symbol::new_object("b".to_string())),
                SymbolNode::leaf(Symbol::new_object("c".to_string())),
            ],
        );

        assert_eq!(
            x_plus_y_equals_a_plus_b_plus_c.collapse_delimiters(),
            expected
        );
    }

    #[test]
    fn test_symbol_nodes_detect_conflicting_arities() {
        let trivial = SymbolNode::leaf_object("x".to_string());
        let trivial_arities = trivial.get_arities();
        assert_eq!(trivial_arities.clone().unwrap().len(), 1);
        assert_eq!(
            trivial_arities
                .unwrap()
                .get(&Symbol::new_object("x".to_string())),
            Some(&0)
        );
        assert!(!trivial.has_conflicting_arities());

        let function = SymbolNode::new(
            Symbol::new_object("f".to_string()),
            vec![
                SymbolNode::leaf_object("x".to_string()),
                SymbolNode::leaf_object("y".to_string()),
                SymbolNode::leaf_object("z".to_string()),
            ],
        );
        let function_arities = function.get_arities();
        assert_eq!(function_arities.clone().unwrap().len(), 4);
        assert_eq!(function_arities.clone().unwrap().get(&"f".into()), Some(&3));
        assert_eq!(function_arities.clone().unwrap().get(&"x".into()), Some(&0));
        assert_eq!(function_arities.clone().unwrap().get(&"y".into()), Some(&0));
        assert_eq!(function_arities.clone().unwrap().get(&"z".into()), Some(&0));
        assert!(!function.has_conflicting_arities());

        let a_plus_b_plus_c = SymbolNode::new(
            Symbol::new_object("+".to_string()),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new(
                    Symbol::new_object("+".to_string()),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("c".to_string()),
                    ],
                ),
            ],
        );

        let x_plus_y = SymbolNode::new(
            Symbol::new_object("+".to_string()),
            vec![
                SymbolNode::leaf(Symbol::new_object("x".to_string())),
                SymbolNode::leaf(Symbol::new_object("y".to_string())),
            ],
        );

        let x_plus_y_equals_a_plus_b_plus_c =
            SymbolNode::new("=".into(), vec![x_plus_y.clone(), a_plus_b_plus_c]);

        let x_plus_y_equals_a_plus_b_plus_c_arities = x_plus_y_equals_a_plus_b_plus_c.get_arities();
        assert_eq!(
            x_plus_y_equals_a_plus_b_plus_c_arities
                .clone()
                .unwrap()
                .len(),
            7
        );
        assert_eq!(
            x_plus_y_equals_a_plus_b_plus_c_arities
                .clone()
                .unwrap()
                .get(&"=".into()),
            Some(&2)
        );
        assert_eq!(
            x_plus_y_equals_a_plus_b_plus_c_arities
                .clone()
                .unwrap()
                .get(&"+".into()),
            Some(&2)
        );
        assert_eq!(
            x_plus_y_equals_a_plus_b_plus_c_arities
                .clone()
                .unwrap()
                .get(&"a".into()),
            Some(&0)
        );
        assert_eq!(
            x_plus_y_equals_a_plus_b_plus_c_arities
                .clone()
                .unwrap()
                .get(&"b".into()),
            Some(&0)
        );
        assert_eq!(
            x_plus_y_equals_a_plus_b_plus_c_arities
                .clone()
                .unwrap()
                .get(&"c".into()),
            Some(&0)
        );
        assert_eq!(
            x_plus_y_equals_a_plus_b_plus_c_arities
                .clone()
                .unwrap()
                .get(&"x".into()),
            Some(&0)
        );
        assert_eq!(
            x_plus_y_equals_a_plus_b_plus_c_arities
                .clone()
                .unwrap()
                .get(&"y".into()),
            Some(&0)
        );
        assert!(!x_plus_y_equals_a_plus_b_plus_c.has_conflicting_arities());
    }

    #[test]
    fn test_symbol_nodes_detect_conflicting_types() {
        let trivial = SymbolNode::leaf_object("x".to_string());
        let trivial_types = trivial.get_type_map();
        assert_eq!(trivial_types.clone().unwrap().len(), 1);
        assert_eq!(
            trivial_types.clone().unwrap().get(&"x".to_string()),
            Some(&Type::Object)
        );

        let a_plus_b_plus_c = SymbolNode::new(
            Symbol::new("+".to_string(), "Operator".into()),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new(
                    Symbol::new("+".to_string(), "Operator".into()),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("c".to_string()),
                    ],
                ),
            ],
        );

        let x_plus_y = SymbolNode::new(
            Symbol::new("+".to_string(), "Operator".into()),
            vec![
                SymbolNode::leaf(Symbol::new_object("x".to_string())),
                SymbolNode::leaf(Symbol::new_object("y".to_string())),
            ],
        );

        let x_plus_y_equals_a_plus_b_plus_c = SymbolNode::new(
            Symbol::new("=".to_string(), "Operator".into()),
            vec![x_plus_y.clone(), a_plus_b_plus_c],
        );

        let expected = vec![
            ("+".to_string(), "Operator".into()),
            ("=".to_string(), "Operator".into()),
            ("a".to_string(), Type::Object),
            ("b".to_string(), Type::Object),
            ("c".to_string(), Type::Object),
            ("x".to_string(), Type::Object),
            ("y".to_string(), Type::Object),
        ]
        .into_iter()
        .collect();
        assert_eq!(x_plus_y_equals_a_plus_b_plus_c.get_type_map(), Ok(expected));
    }
}
