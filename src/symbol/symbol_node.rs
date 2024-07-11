use core::fmt;
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
};

use serde::{Deserialize, Serialize};

use crate::{
    parsing::{
        interpretation::{ExpressionType, Interpretation},
        tokenizer::Token,
    },
    symbol::symbol_type::Type,
};

use super::symbol_type::TypeHierarchy;

pub type SymbolName = String;
pub type SymbolNodeAddress = Vec<usize>;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum SymbolNodeError {
    ConflictingTypes(String, Type, Type),
    ConflictingSymbolArities(SymbolNode),
    ChildIndexOutOfRange,
    DifferentNumberOfArguments(SymbolNode, SymbolNode),
    RelabellingNotInjective(Vec<(String, String)>),
    InvalidFunctionCalledOn(SymbolNodeRoot),
    InvalidAddress,
}

#[derive(Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolNode {
    root: SymbolNodeRoot,
    children: Vec<SymbolNode>,
}

impl From<SymbolNodeRoot> for SymbolNode {
    fn from(value: SymbolNodeRoot) -> Self {
        Self {
            root: value,
            children: vec![],
        }
    }
}

impl From<Symbol> for SymbolNode {
    fn from(value: Symbol) -> Self {
        SymbolNodeRoot::Symbol(value).into()
    }
}

impl Debug for SymbolNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string_representation = self.to_string();
        let line = "--------".to_string();

        write!(
            f,
            "\nSymbolNode.to_string(): {}\nIndented:\n{}\n",
            string_representation, line,
        )?;

        fn format_children(
            node: &SymbolNode,
            f: &mut fmt::Formatter<'_>,
            indentation_level: usize,
        ) -> fmt::Result {
            // Indent children
            let indentation = " ".repeat(indentation_level);
            write!(f, "{}{:?}\n", indentation, node.root)?;

            for child in &node.children {
                // Recursively format children
                format_children(child, f, indentation_level + 4)?;
            }

            Ok(())
        }

        let to_return = format_children(self, f, 0);
        write!(f, "{}\n", line)?;

        to_return
    }
}

impl SymbolNode {
    pub fn new(root: SymbolNodeRoot, children: Vec<SymbolNode>) -> Self {
        SymbolNode { root, children }
    }

    pub fn new_from_symbol(root: Symbol, children: Vec<SymbolNode>) -> Self {
        Self::new(root.into(), children)
    }

    pub fn singleton(s: &str) -> Self {
        Self::leaf(Symbol::new(s.to_string(), s.into()).into())
    }

    pub fn leaf(root: Symbol) -> Self {
        Self::new(root.into(), Vec::new())
    }

    pub fn with_single_child(root: Symbol, child: Symbol) -> Self {
        Self::new(root.into(), vec![SymbolNode::leaf(child)])
    }

    pub fn leaf_object(root: &str) -> Self {
        Self::leaf(Symbol::new_object(root.to_string()).into())
    }

    pub fn object_with_single_child_object(root: &str, child: &str) -> Self {
        Self::with_single_child(
            Symbol::new_object(root.to_string()).into(),
            Symbol::new_object(child.to_string()).into(),
        )
    }

    pub fn arbitrary(child: SymbolNode, evaluates_to_type: Type) -> Self {
        Self::new(
            SymbolNodeRoot::ArbitraryReturning(evaluates_to_type),
            vec![child],
        )
    }

    pub fn contains_arbitrary_nodes(&self) -> bool {
        if let SymbolNodeRoot::ArbitraryReturning(_) = self.root {
            return true;
        }

        self.children
            .iter()
            .any(|child| child.contains_arbitrary_nodes())
    }

    pub fn get_arbitrary_nodes(&self) -> HashSet<Self> {
        if let SymbolNodeRoot::ArbitraryReturning(_) = self.get_root() {
            return vec![self.clone()].into_iter().collect();
        }

        let mut to_return = HashSet::new();
        for child in self.children.iter() {
            to_return = to_return
                .union(&mut child.get_arbitrary_nodes())
                .cloned()
                .collect();
        }

        to_return
    }

    pub fn get_arbitrary_node_instantiations(
        &self,
        statements: &HashSet<SymbolNode>,
    ) -> HashSet<SymbolNode> {
        statements
            .iter()
            .map(|s| self.get_arbitrary_node_instantiation(s))
            .flatten()
            .collect()
    }

    pub fn get_arbitrary_node_instantiation(&self, statement: &SymbolNode) -> HashSet<SymbolNode> {
        // TODO checking Arbitrary Returning doesn't have to happen recursively
        let mut to_return = HashSet::new();
        match self.get_root() {
            SymbolNodeRoot::ArbitraryReturning(return_type) => {
                let mut children_instantiations = self.get_arbitrary_node_instantiations(
                    &statement.get_children().into_iter().cloned().collect(),
                );
                to_return = to_return
                    .union(&mut children_instantiations)
                    .cloned()
                    .collect();
                if &statement.get_evaluates_to_type() == return_type {
                    to_return.insert(statement.clone());
                }
            }
            _ => { // Do nothing
            }
        }
        to_return
    }

    pub fn is_join(&self) -> bool {
        self.root.is_join()
    }

    pub fn join(self, other: Self) -> Self {
        Self::new(SymbolNodeRoot::Join, vec![self, other])
    }

    pub fn has_same_value_as_type(&self) -> bool {
        !self.is_join() && self.get_root_as_string() == self.get_evaluates_to_type().to_string()
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

    pub fn has_children(&self) -> bool {
        self.children.len() > 0
    }

    pub fn push_child(&mut self, child: SymbolNode) {
        self.children.push(child)
    }

    pub fn with_child_replaced(
        mut self,
        i: usize,
        new_child: SymbolNode,
    ) -> Result<Self, SymbolNodeError> {
        if i >= self.get_n_children() {
            return Err(SymbolNodeError::ChildIndexOutOfRange);
        }

        self.children[i] = new_child;
        Ok(self)
    }

    pub fn get_root(&self) -> &SymbolNodeRoot {
        &self.root
    }

    pub fn get_symbol(&self) -> Result<&Symbol, SymbolNodeError> {
        // TODO Could we be missing symbols here because of Arbitrary Root?
        match &self.root {
            SymbolNodeRoot::Join => {
                Err(SymbolNodeError::InvalidFunctionCalledOn(self.root.clone()))
            }
            SymbolNodeRoot::Symbol(s) => Ok(&s),
            SymbolNodeRoot::ArbitraryReturning(r) => {
                Err(SymbolNodeError::InvalidFunctionCalledOn(self.root.clone()))
            }
        }
    }

    pub fn get_root_as_string(&self) -> String {
        match self.get_root() {
            SymbolNodeRoot::Symbol(s) => s.get_name(),
            SymbolNodeRoot::Join => ", ".to_string(),
            SymbolNodeRoot::ArbitraryReturning(r) => r.to_string(),
        }
    }

    pub fn get_evaluates_to_type(&self) -> Type {
        match &self.root {
            SymbolNodeRoot::Join => Type::Join,
            SymbolNodeRoot::Symbol(s) => s.get_evaluates_to_type(),
            SymbolNodeRoot::ArbitraryReturning(t) => t.clone(),
        }
    }

    pub fn has_conflicting_arities(&self) -> bool {
        match self.get_arities() {
            Err(SymbolNodeError::ConflictingSymbolArities(_)) => true,
            _ => false,
        }
    }

    pub fn get_arities(&self) -> Result<HashMap<SymbolNodeRoot, usize>, SymbolNodeError> {
        let mut arities = HashMap::new();
        self.collect_arities(&mut arities)?;
        Ok(arities)
    }

    fn collect_arities(
        &self,
        arities: &mut HashMap<SymbolNodeRoot, usize>,
    ) -> Result<(), SymbolNodeError> {
        // Check if the symbol already exists with a different arity
        match arities.entry(self.root.clone()) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(self.get_n_children());
            }
            std::collections::hash_map::Entry::Occupied(e) => {
                // If the arity (number of children) is different, return an error
                if *e.get() != self.get_n_children() {
                    return Err(SymbolNodeError::ConflictingSymbolArities(self.clone()));
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
        let mut to_return: Self = self.root.clone().into();
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
        self.split(self.get_root())
    }

    pub fn split(&self, symbol: &SymbolNodeRoot) -> Vec<Self> {
        if self.get_root() == symbol {
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

    pub fn replace_by_name(&self, from: &str, to: &Self) -> Result<Self, SymbolNodeError> {
        if !self.is_join() && (self.get_root_as_string() == from) {
            return Ok(to.clone());
        };
        let new_children = self
            .children
            .iter()
            .try_fold(Vec::new(), |mut acc, child| {
                child.replace_by_name(from, to).map(|c| acc.push(c))?;
                Ok(acc)
            })?;
        Ok(Self::new(self.get_root().clone(), new_children))
    }

    pub fn replace_name(&self, from: &str, to: &str) -> Result<Self, SymbolNodeError> {
        let new_root = if !self.is_join() && (self.get_root_as_string() == from) {
            Symbol::new(from.to_string(), self.get_evaluates_to_type()).into()
        } else {
            self.root.clone()
        };
        let new_children = self
            .children
            .iter()
            .try_fold(Vec::new(), |mut acc, child| {
                child.replace_name(from, to).map(|c| acc.push(c))?;
                Ok(acc)
            })?;
        Ok(Self::new(new_root, new_children))
    }

    pub fn replace_symbol(&self, from: &Symbol, to: &Symbol) -> Result<Self, SymbolNodeError> {
        let new_root = if !self.is_join() && (self.get_symbol()? == from) {
            to.clone().into()
        } else {
            self.root.clone().into()
        };
        let new_children = self
            .children
            .iter()
            .try_fold(Vec::new(), |mut acc, child| {
                child.replace_symbol(from, to).map(|c| acc.push(c))?;
                Ok(acc)
            })?;
        Ok(Self::new(new_root, new_children)).into()
    }

    pub fn replace_all(&self, from: &Symbol, to: &SymbolNode) -> Result<Self, SymbolNodeError> {
        if !self.is_join() && (self.get_symbol()? == from) {
            Ok(to.clone())
        } else {
            let children = self.get_children().iter().try_fold(
                Vec::new(),
                |mut acc: Vec<SymbolNode>, child: &SymbolNode| {
                    child
                        .replace_all(from, to)
                        .map(|new_child: SymbolNode| acc.push(new_child))?;
                    Ok(acc)
                },
            )?;
            Ok(Self::new(self.get_root().clone(), children))
        }
    }

    pub fn replace_node(
        &self,
        address: &SymbolNodeAddress,
        new_node: &SymbolNode,
    ) -> Result<Self, SymbolNodeError> {
        let mut to_return = self.clone();
        let mut current_node = &mut to_return;
        for i in address {
            if *i < current_node.get_n_children() {
                current_node = &mut current_node.children[*i];
            } else {
                return Err(SymbolNodeError::InvalidAddress);
            }
        }
        *current_node = new_node.clone();

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
                let mut new_address = vec![i];
                new_address.append(&mut address.clone());
                result.insert(new_address);
            }
        }
        result
    }

    pub fn find_symbol(&self, symbol: &Symbol) -> HashSet<SymbolNodeAddress> {
        self.find_where(&|node| &node.root == &SymbolNodeRoot::Symbol(symbol.clone()))
    }

    pub fn find_symbol_name(&self, symbol_name: &str) -> HashSet<SymbolNodeAddress> {
        self.find_where(&|node| node.root.get_name() == symbol_name)
    }

    pub fn to_interpreted_string_and_type_map(
        &self,
        interpretations: &Vec<Interpretation>,
    ) -> Result<(String, Vec<(String, Type)>), SymbolNodeError> {
        self.get_sorted_type_map()
            .map(|type_map| (self.to_interpreted_string(interpretations), type_map))
    }

    pub fn to_interpreted_string(&self, interpretations: &Vec<Interpretation>) -> String {
        if self.has_children() {
            let interpreted_children = self
                .get_children()
                .iter()
                .map(|child| child.to_interpreted_string(interpretations))
                .collect::<Vec<_>>();
            let functional_string = format!(
                "{}({})",
                self.get_root_as_string(),
                interpreted_children.join(", ")
            )
            .to_string();
            match interpretations.iter().find(|i| i.could_produce(self)) {
                Some(interpretation) => match interpretation.get_expression_type() {
                    ExpressionType::Singleton => self.get_root_as_string(),
                    ExpressionType::Prefix => {
                        format!("{}{}", self.get_root_as_string(), interpreted_children[0])
                            .to_string()
                    }
                    ExpressionType::Infix => format!(
                        "({}{}{})",
                        interpreted_children[0],
                        self.get_root_as_string(),
                        interpreted_children[1]
                    )
                    .to_string(),
                    ExpressionType::Postfix => {
                        format!("{}{}", interpreted_children[0], self.get_root_as_string())
                            .to_string()
                    }
                    ExpressionType::Outfix(right) => match right {
                        Token::Object(right_string) => format!(
                            "{}{}{}",
                            self.get_root_as_string(),
                            interpreted_children[0],
                            right_string,
                        )
                        .to_string(),
                        _ => functional_string,
                    },
                    ExpressionType::Functional => functional_string,
                },
                None => functional_string,
            }
        } else {
            self.get_root_as_string()
        }
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
        let mut result = match self.get_root() {
            SymbolNodeRoot::Symbol(symbol) => vec![symbol.clone()],
            SymbolNodeRoot::Join => Vec::new(),
            SymbolNodeRoot::ArbitraryReturning(_) => Vec::new(),
        };
        for child in &self.children {
            result.extend(child.get_symbols());
        }
        result.into_iter().collect()
    }

    pub fn get_types(&self) -> HashSet<Type> {
        let mut result = vec![self.get_evaluates_to_type().clone()];
        for child in &self.children {
            result.extend(child.get_types());
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

    pub fn relabel(&self, old_label: &str, new_label: &str) -> Self {
        self.relabel_and_get_addresses_if(old_label, new_label, Vec::new(), &|_, _| true)
            .0
    }

    fn relabel_and_get_addresses_if(
        &self,
        old_label: &str,
        new_label: &str,
        current_address: SymbolNodeAddress,
        condition: &dyn Fn(&Self, &SymbolNodeAddress) -> bool,
    ) -> (Self, HashSet<SymbolNodeAddress>) {
        let (new_children, children_addresses) = self
            .children
            .iter()
            .enumerate()
            .map(|(i, child)| {
                let mut child_address = current_address.clone();
                child_address.push(i);
                child.relabel_and_get_addresses_if(old_label, new_label, child_address, condition)
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
        if !self.root.is_join()
            && self.root.get_name() == old_label
            && condition(self, &current_address)
        {
            let mut addresses = children_addresses;
            addresses.insert(current_address);
            (
                Self::new(
                    Symbol::new(new_label.to_string(), self.root.get_evaluates_to_type()).into(),
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

    pub fn relabel_all(&self, relabelling: &HashSet<(String, String)>) -> Self {
        relabelling
            .into_iter()
            .fold(
                (self.clone(), HashSet::new()),
                |acc, (old_label, new_label)| {
                    let (new_tree, addresses) = acc;
                    new_tree.relabel_and_get_addresses_if(
                        old_label,
                        new_label,
                        Vec::new(),
                        &|_node, address| !addresses.contains(address),
                    )
                },
            )
            .0
    }

    pub fn get_typed_relabelling(
        &self,
        hierarchy: &TypeHierarchy,
        other: &Self,
    ) -> Result<Substitution, SymbolNodeError> {
        if !(hierarchy.generalizes(self, other).is_ok()) {
            return Err(SymbolNodeError::ConflictingTypes(
                self.get_root_as_string(),
                self.get_evaluates_to_type(),
                other.get_evaluates_to_type(),
            ));
        }
        return if !self.has_children() {
            let substitutions = vec![(self.get_root_as_string().clone(), other.clone())]
                .into_iter()
                .collect();
            Ok(Substitution::new(substitutions))
        } else {
            if self.get_n_children() != other.get_n_children() {
                return Err(SymbolNodeError::DifferentNumberOfArguments(
                    self.clone(),
                    other.clone(),
                ));
            }
            let new_inner = self
                .get_children()
                .iter()
                .zip(other.get_children())
                .try_fold(HashMap::new(), |mut acc, (i, j)| {
                    i.get_typed_relabelling(hierarchy, j).map(|new_subs| {
                        new_subs.get_inner().iter().for_each(|(s, n)| {
                            acc.insert(s.clone(), n.clone());
                        });
                        acc
                    })
                })?;
            Ok(Substitution::new(new_inner))
        };
    }

    pub fn get_relabelling(
        &self,
        other: &Self,
    ) -> Result<HashMap<String, String>, SymbolNodeError> {
        if self.children.len() != other.children.len() {
            return Err(SymbolNodeError::DifferentNumberOfArguments(
                self.clone(),
                other.clone(),
            ));
        }
        let mut to_return = if self.root.is_join() {
            vec![]
        } else {
            vec![(self.root.get_name(), other.root.get_name())]
        };
        for (_i, (child, other_child)) in
            self.children.iter().zip(other.children.iter()).enumerate()
        {
            let child_relabelling = child.get_relabelling(other_child)?;
            for (old_label, new_label) in child_relabelling {
                to_return.push((old_label, new_label));
            }
        }

        let to_return_keys = to_return
            .iter()
            .map(|x| x.0.clone())
            .collect::<HashSet<_>>();

        let to_return_key_values = to_return
            .iter()
            .map(|x| (x.0.clone(), x.1.clone()))
            .collect::<HashSet<_>>();

        if to_return_keys.len() != to_return_key_values.len() {
            return Err(SymbolNodeError::RelabellingNotInjective(
                to_return_key_values.into_iter().collect(),
            ));
        }

        Ok(to_return.into_iter().collect())
    }

    pub fn validate(&self) -> Result<(), SymbolNodeError> {
        self.get_type_map()?;
        self.get_arities()?;
        Ok(())
    }

    pub fn get_sorted_type_map(&self) -> Result<Vec<(String, Type)>, SymbolNodeError> {
        let mut to_return: Vec<(String, Type)> =
            self.get_type_map().map(|m| m.into_iter().collect())?;
        to_return.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(to_return)
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
        if self.root.is_join() {
            return Ok(to_return);
        }
        match children_type_map.get(&self.get_root_as_string()) {
            Some(prior_type) => {
                if &self.get_evaluates_to_type() != prior_type {
                    return Err(SymbolNodeError::ConflictingTypes(
                        self.get_root_as_string(),
                        self.get_evaluates_to_type(),
                        prior_type.clone(),
                    ));
                }
            }
            None => {
                to_return.insert(self.get_root_as_string(), self.get_evaluates_to_type());
            }
        }
        Ok(to_return)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Substitution {
    substitution: HashMap<String, SymbolNode>,
}

impl Substitution {
    pub fn new(substitution: HashMap<String, SymbolNode>) -> Self {
        Self { substitution }
    }

    pub fn get_inner(&self) -> &HashMap<String, SymbolNode> {
        &self.substitution
    }

    pub fn substitute(&self, statement: &SymbolNode) -> SymbolNode {
        self.substitute_and_get_addresses_if(statement, &|_, _| true)
            .0
    }

    pub fn substitute_and_get_addresses_if(
        &self,
        statement: &SymbolNode,
        condition: &dyn Fn(&SymbolNode, &SymbolNodeAddress) -> bool,
    ) -> (SymbolNode, HashSet<SymbolNodeAddress>) {
        let mut addresses_to_subs = self
            .substitution
            .iter()
            .map(|(from, to)| {
                statement
                    .find_symbol_name(from)
                    .into_iter()
                    .filter(|address| condition(statement, address))
                    .map(|x| (x, to))
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<_>>();

        // Sort decreasing in order to capture the deepest addresses first so they don't stomp each
        // other
        addresses_to_subs.sort_by(|a, b| a.0.len().cmp(&b.0.len()).reverse());

        let mut to_return = statement.clone();
        addresses_to_subs.iter().for_each(|(x, s)| {
            to_return = to_return
                .replace_node(x, s)
                .expect("The address is guaranteed to be valid.");
        });
        (
            to_return,
            addresses_to_subs.into_iter().map(|x| x.0).collect(),
        )
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolNodeRoot {
    Symbol(Symbol),
    Join,
    ArbitraryReturning(Type),
}

impl Default for SymbolNodeRoot {
    fn default() -> Self {
        Symbol::default().into()
    }
}

impl From<Symbol> for SymbolNodeRoot {
    fn from(value: Symbol) -> Self {
        Self::Symbol(value)
    }
}

impl From<String> for SymbolNodeRoot {
    fn from(value: String) -> Self {
        Self::Symbol(value.into())
    }
}

impl From<&str> for SymbolNodeRoot {
    fn from(value: &str) -> Self {
        value.to_string().into()
    }
}

impl SymbolNodeRoot {
    pub fn is_join(&self) -> bool {
        self == &Self::Join
    }

    pub fn to_string(&self) -> String {
        match self {
            Self::Join => "Join".to_string(),
            Self::Symbol(s) => s.to_string(),
            Self::ArbitraryReturning(t) => {
                format!("ArbitraryReturning({})", t.to_string()).to_string()
            }
        }
    }

    pub fn get_name(&self) -> String {
        // TODO Ensure that this isn't happening for non symbol
        match self {
            Self::Join => "Join".to_string(),
            Self::Symbol(s) => s.get_name(),
            Self::ArbitraryReturning(t) => {
                format!("ArbitraryReturning({})", t.to_string()).to_string()
            }
        }
    }

    pub fn get_evaluates_to_type(&self) -> Type {
        match self {
            Self::Join => Type::Join,
            Self::Symbol(s) => s.get_evaluates_to_type(),
            Self::ArbitraryReturning(t) => t.clone(),
        }
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

    pub fn new_with_same_type_as_value(name: &str) -> Self {
        Self::new(name.to_string(), name.into())
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
    use crate::{
        parsing::{interpretation::Interpretation, parser::Parser},
        symbol::symbol_type::GeneratedTypeCondition,
    };

    use super::*;

    #[test]
    fn test_symbol_node_gets_arbitrary_node_instantiations() {
        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "Boolean".into()),
            Interpretation::infix_operator("&".into(), 2, "Boolean".into()),
            Interpretation::outfix_operator(("|".into(), "|".into()), 2, "Integer".into()),
            Interpretation::postfix_operator("!".into(), 3, "Integer".into()),
            Interpretation::prefix_operator("-".into(), 4, "Integer".into()),
            Interpretation::singleton("p", "Boolean".into()),
            Interpretation::singleton("q", "Boolean".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
            Interpretation::arbitrary_functional("Any".into(), 99, "Boolean".into()),
        ];

        let parser = Parser::new(interpretations.clone());

        let custom_tokens = vec![
            "=".to_string(),
            "&".to_string(),
            "|".to_string(),
            "!".to_string(),
            "-".to_string(),
        ];

        let parse = |s: &str| parser.parse_from_string(custom_tokens.clone(), s).unwrap();

        let non_arbitrary = parse("x");
        assert_eq!(
            non_arbitrary.get_arbitrary_node_instantiation(&non_arbitrary.clone()),
            HashSet::new()
        );

        let wrong_type = non_arbitrary.clone();
        let single_arbitrary = parser
            .parse_from_string(custom_tokens.clone(), "Any(x)")
            .unwrap();
        assert_eq!(
            single_arbitrary.get_arbitrary_node_instantiation(&wrong_type),
            HashSet::new()
        );

        let p = parse("p");
        let single_arbitrary = parse("Any(p)");
        assert_eq!(
            single_arbitrary.get_arbitrary_node_instantiation(&p.clone()),
            vec![p.clone()].into_iter().collect()
        );

        let p_and_q = parse("p&q");
        assert_eq!(
            single_arbitrary.get_arbitrary_node_instantiation(&p_and_q.clone()),
            vec![p, parse("q"), p_and_q].into_iter().collect()
        );

        let x_equals_y_and_q = parse("(x=y)&q");
        assert_eq!(
            single_arbitrary.get_arbitrary_node_instantiation(&x_equals_y_and_q.clone()),
            vec![parse("x=y"), parse("q"), x_equals_y_and_q]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn test_symbol_node_to_interpreted_string() {
        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "Integer".into()),
            Interpretation::outfix_operator(("|".into(), "|".into()), 2, "Integer".into()),
            Interpretation::postfix_operator("!".into(), 3, "Integer".into()),
            Interpretation::prefix_operator("-".into(), 4, "Integer".into()),
            Interpretation::function("f".into(), 99),
            Interpretation::singleton("a", "Integer".into()),
            Interpretation::singleton("b", "Integer".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
            Interpretation::parentheses_like(
                Token::Object("{".to_string()),
                Token::Object("}".to_string()),
            ),
            Interpretation::function("\\frac".into(), 99),
            Interpretation::singleton("\\alpha", "Integer".into()),
            Interpretation::singleton("\\beta", "Integer".into()),
            Interpretation::singleton("\\gamma", "Integer".into()),
        ];

        let parser = Parser::new(interpretations.clone());

        let custom_tokens = vec!["=".to_string(), "|".to_string()];

        let trivial = SymbolNode::leaf_object("d");
        assert_eq!(
            trivial.to_interpreted_string(&interpretations),
            "d".to_string()
        );

        let trivial = SymbolNode::leaf_object("x");
        assert_eq!(
            trivial.to_interpreted_string(&interpretations),
            "x".to_string()
        );

        let x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "x=y")
            .unwrap();
        assert_eq!(x_equals_y.to_interpreted_string(&interpretations), "(x=y)");

        let abs_x = parser
            .parse_from_string(custom_tokens.clone(), "|x|")
            .unwrap();
        assert_eq!(
            abs_x.to_interpreted_string(&interpretations),
            "|x|".to_string()
        );

        let neg_x = parser
            .parse_from_string(custom_tokens.clone(), "-x")
            .unwrap();
        assert_eq!(
            neg_x.to_interpreted_string(&interpretations),
            "-x".to_string()
        );

        let x_factorial = parser
            .parse_from_string(custom_tokens.clone(), "x!")
            .unwrap();
        assert_eq!(
            x_factorial.to_interpreted_string(&interpretations),
            "x!".to_string()
        );

        let f_of_x = parser
            .parse_from_string(custom_tokens.clone(), "f(x)")
            .unwrap();
        assert_eq!(
            f_of_x.to_interpreted_string(&interpretations),
            "f(x)".to_string()
        );

        let f_of_x_y_z = parser
            .parse_from_string(custom_tokens.clone(), "f(x, y, z)")
            .unwrap();
        assert_eq!(
            f_of_x_y_z.to_interpreted_string(&interpretations),
            "f(x, y, z)".to_string()
        );

        let f_of_alpha_beta_gamma = parser
            .parse_from_string(custom_tokens.clone(), "f(\\alpha, \\beta, \\gamma)")
            .unwrap();
        assert_eq!(
            f_of_alpha_beta_gamma.to_interpreted_string(&interpretations),
            "f(\\alpha, \\beta, \\gamma)".to_string()
        );

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "Integer".into()),
            Interpretation::postfix_operator("=".into(), 1, "Integer".into()),
            Interpretation::function("=".into(), 1),
        ];

        let parser = Parser::new(interpretations.clone());

        let custom_tokens = vec!["=".to_string(), "|".to_string()];

        let x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "x=y")
            .unwrap();
        assert_eq!(x_equals_y.to_interpreted_string(&interpretations), "(x=y)");

        let plus = Interpretation::infix_operator("+".into(), 1, "+".into());
        let integer_condition = GeneratedTypeCondition::IsInteger;

        let integer = Interpretation::generated_type(integer_condition);
        assert_eq!(integer, integer);

        let interpretations = vec![plus, integer];
        let parser = Parser::new(interpretations.clone());
        let two_plus_two = parser
            .parse_from_string(vec!["+".to_string()], "(2+2)")
            .unwrap();

        assert_eq!(
            two_plus_two.to_interpreted_string(&interpretations),
            "(2+2)".to_string()
        );

        let one_plus_two_then_plus_three = parser
            .parse_from_string(vec!["+".to_string()], "(1+2)+3")
            .unwrap();
        let then_one_plus_two_plus_three = parser
            .parse_from_string(vec!["+".to_string()], "1+(2+3)")
            .unwrap();
        assert_eq!(
            one_plus_two_then_plus_three.to_interpreted_string(&interpretations),
            "((1+2)+3)"
        );
        assert_eq!(
            then_one_plus_two_plus_three.to_interpreted_string(&interpretations),
            "(1+(2+3))"
        );
    }

    #[test]
    fn test_symbol_node_finds() {
        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "Integer".into()),
            Interpretation::singleton("a", "Integer".into()),
            Interpretation::singleton("b", "Integer".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
        ];
        let parser = Parser::new(interpretations);

        let custom_tokens = vec!["=".to_string()];
        let b = parser
            .parse_from_string(custom_tokens.clone(), "b")
            .unwrap();
        let x = parser
            .parse_from_string(custom_tokens.clone(), "x")
            .unwrap();
        let y = parser
            .parse_from_string(custom_tokens.clone(), "y")
            .unwrap();
        let z = parser
            .parse_from_string(custom_tokens.clone(), "z")
            .unwrap();
        let a_equals_b = parser
            .parse_from_string(custom_tokens.clone(), "a=b")
            .unwrap();
        let x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "x=y")
            .unwrap();
        let x_equals_y_equals_b = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=b")
            .unwrap();
        let x_equals_y_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=y")
            .unwrap();
        let x_equals_y_equals_x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=(x=y)")
            .unwrap();
        let x_equals_y_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=z")
            .unwrap();
        assert_eq!(b.find_symbol_name("x"), HashSet::new());
        assert_eq!(b.find_symbol_name("b"), vec![vec![]].into_iter().collect());
        assert_eq!(
            b.find_symbol(&Symbol::new("b".to_string(), "Integer".into())),
            vec![vec![]].into_iter().collect()
        );
        assert_eq!(
            b.find_symbol(&Symbol::new_object("b".to_string())),
            HashSet::new()
        );
        assert_eq!(
            a_equals_b.find_symbol_name("b"),
            vec![vec![1]].into_iter().collect()
        );
        assert_eq!(
            x_equals_y_equals_z.find_symbol_name("x"),
            vec![vec![0, 0]].into_iter().collect()
        );
        assert_eq!(
            x_equals_y_equals_z.find_symbol_name("y"),
            vec![vec![0, 1]].into_iter().collect()
        );
        assert_eq!(
            x_equals_y_equals_z.find_symbol_name("z"),
            vec![vec![1]].into_iter().collect()
        );
        assert_eq!(
            x_equals_y_equals_y.find_symbol_name("x"),
            vec![vec![0, 0]].into_iter().collect()
        );
        assert_eq!(
            x_equals_y_equals_y.find_symbol_name("y"),
            vec![vec![0, 1], vec![1]].into_iter().collect()
        );
    }

    #[test]
    fn test_substitution_substitutes() {
        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "Integer".into()),
            Interpretation::singleton("a", "Integer".into()),
            Interpretation::singleton("b", "Integer".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
        ];
        let parser = Parser::new(interpretations);

        let custom_tokens = vec!["=".to_string()];
        let b = parser
            .parse_from_string(custom_tokens.clone(), "b")
            .unwrap();
        let x = parser
            .parse_from_string(custom_tokens.clone(), "x")
            .unwrap();
        let y = parser
            .parse_from_string(custom_tokens.clone(), "y")
            .unwrap();
        let z = parser
            .parse_from_string(custom_tokens.clone(), "z")
            .unwrap();
        let a_equals_b = parser
            .parse_from_string(custom_tokens.clone(), "a=b")
            .unwrap();
        let x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "x=y")
            .unwrap();
        let x_equals_y_equals_b = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=b")
            .unwrap();
        let x_equals_y_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=y")
            .unwrap();
        let x_equals_y_equals_x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=(x=y)")
            .unwrap();
        let x_equals_y_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=z")
            .unwrap();

        let substitution = Substitution::new(
            vec![("a".to_string(), x_equals_y.clone()), ("b".to_string(), b)]
                .into_iter()
                .collect(),
        );
        assert_eq!(
            substitution.substitute(&a_equals_b.clone()),
            x_equals_y_equals_b.clone()
        );

        let substitution = Substitution::new(
            vec![
                ("a".to_string(), x_equals_y.clone()),
                ("b".to_string(), x_equals_y.clone()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(
            substitution.substitute(&a_equals_b),
            x_equals_y_equals_x_equals_y
        );

        let substitution = Substitution::new(
            vec![
                ("x".to_string(), x_equals_y.clone()),
                ("y".to_string(), y.clone()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(
            substitution.substitute(&x_equals_y),
            x_equals_y_equals_y.clone()
        );

        let substitution = Substitution::new(
            vec![
                ("z".to_string(), z.clone()),
                ("y".to_string(), y.clone()),
                ("x".to_string(), x.clone()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(
            substitution.substitute(&x_equals_y_equals_z),
            x_equals_y_equals_z.clone()
        );
    }

    #[test]
    fn test_symbol_node_initializes() {
        let a_equals_b_plus_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a"),
                SymbolNode::new(
                    "+".into(),
                    vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("c")],
                ),
            ],
        );

        assert_eq!(a_equals_b_plus_c.get_depth(), 3);
        assert_eq!(a_equals_b_plus_c.get_n_children(), 2);
        assert_eq!(a_equals_b_plus_c.to_symbol_string(), "=(a,+(b,c))");

        let n_factorial = SymbolNode::object_with_single_child_object("!", "n");

        // prod_{i = 1}^{n} i
        // Point: Expand 5! to 5 * 4 * 3 * 2 * 1, which is going to require a transformation like:
        // 5! => prod_{i = 1}^{5} i => 5 * 4 * 3 * 2 * 1
        let n_factorial_definition = SymbolNode::new(
            "Prod".into(),
            vec![
                SymbolNode::leaf_object("i"), // i is the index variable
                SymbolNode::leaf_object("1"), // 1 is the lower bound
                SymbolNode::leaf_object("n"), // n is the upper bound
                SymbolNode::leaf_object("i"), // i is the expression to be multiplied
            ],
        );

        let factorial_definition =
            SymbolNode::new("=".into(), vec![n_factorial, n_factorial_definition]);

        assert_eq!(factorial_definition.get_depth(), 3);
    }

    #[test]
    fn test_symbol_node_with_child_replaced() {
        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "Integer".into()),
            Interpretation::singleton("a", "Integer".into()),
            Interpretation::singleton("b", "Integer".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
        ];
        let parser = Parser::new(interpretations);

        let custom_tokens = vec!["=".to_string()];
        let a_equals_b = parser
            .parse_from_string(custom_tokens.clone(), "a=b")
            .unwrap();
        let q_equals_b = parser
            .parse_from_string(custom_tokens.clone(), "q=b")
            .unwrap();
        let a_equals_q = parser
            .parse_from_string(custom_tokens.clone(), "a=q")
            .unwrap();
        let x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "x=y")
            .unwrap();
        let a_equals_x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "a=(x=y)")
            .unwrap();

        assert_eq!(
            a_equals_b
                .clone()
                .with_child_replaced(0, Symbol::new_object("q".to_string()).into())
                .unwrap(),
            q_equals_b
        );
        assert_eq!(
            a_equals_b
                .clone()
                .with_child_replaced(1, Symbol::new_object("q".to_string()).into())
                .unwrap(),
            a_equals_q
        );
        assert_eq!(
            a_equals_b
                .clone()
                .with_child_replaced(2, Symbol::new_object("q".to_string()).into()),
            Err(SymbolNodeError::ChildIndexOutOfRange),
        );

        assert_eq!(
            a_equals_b
                .clone()
                .with_child_replaced(1, x_equals_y)
                .unwrap(),
            a_equals_x_equals_y,
        );
    }

    #[test]
    fn test_symbol_nodes_relabel() {
        let a_equals_b_plus_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a"),
                SymbolNode::new(
                    "+".into(),
                    vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("c")],
                ),
            ],
        );

        let x_equals_b_plus_c = a_equals_b_plus_c.relabel("a", "x");
        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("x"),
                SymbolNode::new(
                    "+".into(),
                    vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("c")],
                ),
            ],
        );
        assert_eq!(x_equals_b_plus_c, expected);

        let x_equals_y_plus_y = x_equals_b_plus_c.relabel("b", "y").relabel("c", "y");
        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("x"),
                SymbolNode::new(
                    "+".into(),
                    vec![SymbolNode::leaf_object("y"), SymbolNode::leaf_object("y")],
                ),
            ],
        );
        assert_eq!(x_equals_y_plus_y, expected);

        let x_equals_x_plus_x = x_equals_y_plus_y.relabel("y", "x");
        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("x"),
                SymbolNode::new(
                    "+".into(),
                    vec![SymbolNode::leaf_object("x"), SymbolNode::leaf_object("x")],
                ),
            ],
        );
        assert_eq!(x_equals_x_plus_x, expected);

        let also_x_equals_x_plus_x = a_equals_b_plus_c.relabel_all(
            &vec![
                ("b".to_string(), "x".to_string()),
                ("c".to_string(), "x".to_string()),
                ("a".to_string(), "x".to_string()),
            ]
            .into_iter()
            .collect(),
        );

        assert_eq!(x_equals_x_plus_x, also_x_equals_x_plus_x);

        let a_equals_c_plus_b = a_equals_b_plus_c.relabel_all(
            &vec![
                ("b".to_string(), "c".to_string()),
                ("c".to_string(), "b".to_string()),
            ]
            .into_iter()
            .collect(),
        );

        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::leaf_object("a"),
                SymbolNode::new(
                    "+".into(),
                    vec![SymbolNode::leaf_object("c"), SymbolNode::leaf_object("b")],
                ),
            ],
        );

        assert_eq!(a_equals_c_plus_b, expected);

        let p_joined_q = SymbolNode::new(
            SymbolNodeRoot::Join,
            vec![SymbolNode::leaf_object("p"), SymbolNode::leaf_object("q")],
        );

        let a_joined_q = p_joined_q.relabel("p", "a");
        let expected = SymbolNode::new(
            SymbolNodeRoot::Join,
            vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("q")],
        );
        assert_eq!(a_joined_q, expected);
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
            Symbol::delimiter("+".to_string()).into(),
            vec![
                SymbolNode::leaf_object("a"),
                SymbolNode::new(
                    Symbol::delimiter("+".to_string()).into(),
                    vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("c")],
                ),
            ],
        );

        let expected = SymbolNode::new(
            Symbol::delimiter("+".to_string()).into(),
            vec![
                SymbolNode::leaf_object("a"),
                SymbolNode::leaf_object("b"),
                SymbolNode::leaf_object("c"),
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
            a_plus_b_plus_c.split(&Symbol::new_object("+".to_string()).into()),
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
            Symbol::delimiter("+".to_string()).into(),
            vec![
                SymbolNode::leaf_object("a"),
                SymbolNode::new(
                    Symbol::delimiter("+".to_string()).into(),
                    vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("c")],
                ),
            ],
        );

        let expected = SymbolNode::new(
            Symbol::delimiter("+".to_string()).into(),
            vec![
                SymbolNode::leaf_object("a"),
                SymbolNode::leaf_object("b"),
                SymbolNode::leaf_object("c"),
            ],
        );

        assert_eq!(a_plus_b_plus_c.collapse_delimiters(), expected);

        let x_plus_y = SymbolNode::new(
            Symbol::delimiter("+".to_string()).into(),
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
        let trivial = SymbolNode::leaf_object("x");
        let trivial_arities = trivial.get_arities();
        assert_eq!(trivial_arities.clone().unwrap().len(), 1);
        assert_eq!(
            trivial_arities
                .unwrap()
                .get(&Symbol::new_object("x".to_string()).into()),
            Some(&0)
        );
        assert!(!trivial.has_conflicting_arities());

        let function = SymbolNode::new(
            Symbol::new_object("f".to_string()).into(),
            vec![
                SymbolNode::leaf_object("x"),
                SymbolNode::leaf_object("y"),
                SymbolNode::leaf_object("z"),
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
            Symbol::new_object("+".to_string()).into(),
            vec![
                SymbolNode::leaf_object("a"),
                SymbolNode::new(
                    Symbol::new_object("+".to_string()).into(),
                    vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("c")],
                ),
            ],
        );

        let x_plus_y = SymbolNode::new(
            Symbol::new_object("+".to_string()).into(),
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
        let trivial = SymbolNode::leaf_object("x");
        let trivial_types = trivial.get_type_map();
        assert_eq!(trivial_types.clone().unwrap().len(), 1);
        assert_eq!(
            trivial_types.clone().unwrap().get(&"x".to_string()),
            Some(&Type::Object)
        );

        let a_plus_b_plus_c = SymbolNode::new(
            Symbol::new("+".to_string(), "Operator".into()).into(),
            vec![
                SymbolNode::leaf_object("a"),
                SymbolNode::new(
                    Symbol::new("+".to_string(), "Operator".into()).into(),
                    vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("c")],
                ),
            ],
        );

        let x_plus_y = SymbolNode::new(
            Symbol::new("+".to_string(), "Operator".into()).into(),
            vec![
                SymbolNode::leaf(Symbol::new_object("x".to_string())),
                SymbolNode::leaf(Symbol::new_object("y".to_string())),
            ],
        );

        let x_plus_y_equals_a_plus_b_plus_c = SymbolNode::new(
            Symbol::new("=".to_string(), "Operator".into()).into(),
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
