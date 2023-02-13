use std::collections::{HashMap, HashSet};

use crate::symbol::symbol_type::Type;

pub type SymbolName = String;
pub type SymbolNodeAddress = Vec<usize>;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum SymbolNodeError {
    IncorrectArgumentTypes(Vec<String>),
    TypeConflicts(Vec<String>),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SymbolNode {
    root: Symbol,
    children: Vec<SymbolNode>,
}

impl SymbolNode {

    pub fn new(root: Symbol, children: Vec<SymbolNode>) -> Self {
        SymbolNode {
            root,
            children,
        }
    }

    pub fn leaf(root: Symbol) -> Self {
        Self::new(root, Vec::new())
    }

    pub fn with_single_child(root: Symbol, child: Symbol) -> Self {
        Self::new(root, vec![SymbolNode::leaf(child)])
    }

    pub fn new_generic(root: String, children: Vec<Self>) -> Self {
        Self::new(Symbol::new(root, Type::new_generic_function_with_arguments(children.len())), children)
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
            1 + self.children.iter().map(|child| child.get_depth()).max().unwrap()
        }
    }

    pub fn get_children(&self) -> &Vec<SymbolNode> {
        &self.children
    }

    pub fn get_n_children(&self) -> usize {
        self.children.len()
    }

    pub fn get_symbol(&self) -> &Symbol {
        &self.root
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
            let arguments = self.children.iter().map(|x| x.to_symbol_string()).collect::<Vec<_>>().join(",");
            result = format!("{}{})", result, arguments);
            result
        }
    }

    pub fn relabel(&self, old_label: String, new_label: String) -> Self {
        self.relabel_and_get_addresses_if(old_label, new_label, Vec::new(), &|x| true).0
    }

    fn relabel_and_get_addresses_if(&self, old_label: String, new_label: String, current_address: SymbolNodeAddress, condition: &dyn Fn((Self, SymbolNodeAddress)) -> bool) -> (Self, HashSet<SymbolNodeAddress>) {
        let (new_children, children_addresses) = self.children.iter().enumerate().map(|(i, child)| {
            let mut child_address = current_address.clone();
            child_address.push(i);
            child.relabel_and_get_addresses_if(old_label.clone(), new_label.clone(), child_address, condition)
            }
        ).fold((Vec::new(), HashSet::new()), |acc, elt| {
            let (new_children, new_children_addresses) = acc;
            let (new_child, new_child_addresses) = elt;
            (new_children.into_iter().chain(vec![new_child]).collect(), new_children_addresses.into_iter().chain(new_child_addresses).collect())
        });
        if self.root.get_name() == old_label && condition((self.clone(), current_address.clone())) {
            let mut addresses  = children_addresses;
            addresses.insert(current_address);
            (Self::new(Symbol::new(new_label, self.root.get_type()), new_children), addresses)
        } else {
            (Self::new(self.root.clone(), new_children), children_addresses)
        }
    }

    pub fn relabel_all(&self, relabelling: HashSet<(String, String)>) -> Self {
        relabelling.into_iter().fold(
            (self.clone(), HashSet::new()),
            |acc, (old_label, new_label)| {
                let (new_tree, addresses) = acc;
                new_tree.relabel_and_get_addresses_if(old_label, new_label, Vec::new(), &|x| !addresses.contains(&x.1))
            }
        ).0
    }

    pub fn validate(&self) -> Result<(), SymbolNodeError> {
        let type_conflicts = self.get_type_conflicts();
        let incorrect_argument_types = self.get_incorrect_argument_types();

        if type_conflicts.len() > 0 {
            return Err(SymbolNodeError::TypeConflicts(type_conflicts.into_iter().map(|x| x.to_string()).collect()));
        }
        if incorrect_argument_types.len() > 0 {
            return Err(SymbolNodeError::IncorrectArgumentTypes(incorrect_argument_types.into_iter().map(|x| x.to_string()).collect()));
        }

        Ok(())
    }

    pub fn get_type_conflicts(&self) -> HashSet<SymbolName> {
        let mut to_return: HashSet<_> = Vec::new().into_iter().collect();
        let mut type_map = HashMap::new();
        for symbol in self.get_symbols() {
            if type_map.contains_key(&symbol.get_name()) {
                if type_map.get(&symbol.get_name()).unwrap() != &symbol.get_type() {
                    to_return.insert(symbol.get_name());
                }
            } else {
                type_map.insert(symbol.get_name(), symbol.get_type());
            }
        }

        return to_return;
    }

    pub fn get_incorrect_argument_types(&self) -> HashSet<SymbolName> {
        let mut to_return: HashSet<_> = Vec::new().into_iter().collect();

        let argument_types_from_type = self.root.get_type().get_argument_types();
        let argument_types_from_children = self.children.iter().map(|child| child.root.get_type()).collect::<Vec<_>>();

        if !Type::are_pairwise_allowed_to_take(argument_types_from_type, argument_types_from_children) {
            to_return.insert(self.root.get_name());
        }

        for child in &self.children {
            to_return.extend(child.get_incorrect_argument_types());
        }

        return to_return;
    }

}

#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct Symbol {
    name: SymbolName,
    symbol_type: Type,
}

impl Symbol {
    
        pub fn new(name: SymbolName, symbol_type: Type) -> Self {
            Self {
                name,
                symbol_type,
            }
        }

        pub fn new_object(name: SymbolName) -> Self {
            Self::new(name, Type::default())
        }
    
        pub fn get_name(&self) -> SymbolName {
            self.name.clone()
        }
    
        pub fn get_type(&self) -> Type {
            self.symbol_type.clone()
        }
    
}


#[cfg(test)]
mod test_statement {
    use super::*;


    #[test]
    fn test_symbol_node_initializes() {

        let a_equals_b_plus_c = SymbolNode::new_generic(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new_generic(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("c".to_string())
                    ]
                )
            ]
        );

        assert_eq!(a_equals_b_plus_c.get_depth(), 3);
        assert_eq!(a_equals_b_plus_c.get_n_children(), 2);
        assert_eq!(a_equals_b_plus_c.to_symbol_string(), "=(a,+(b,c))");

        let n_factorial = SymbolNode::object_with_single_child_object("!".to_string(), "n".to_string());
        
        // prod_{i = 1}^{n} i
        // Point: Expand 5! to 5 * 4 * 3 * 2 * 1, which is going to require a transformation like:
        // 5! => prod_{i = 1}^{5} i => 5 * 4 * 3 * 2 * 1
        let n_factorial_definition = SymbolNode::new_generic(
            "Prod".to_string(),
            vec![
                SymbolNode::leaf_object("i".to_string()), // i is the index variable
                SymbolNode::leaf_object("1".to_string()), // 1 is the lower bound
                SymbolNode::leaf_object("n".to_string()), // n is the upper bound
                SymbolNode::leaf_object("i".to_string()), // i is the expression to be multiplied
            ]
        );

        let factorial_definition = SymbolNode::new_generic(
            "=".to_string(),
            vec![
                n_factorial,
                n_factorial_definition
            ]
        );

        assert_eq!(factorial_definition.get_depth(), 3);
        
    }

    #[test]
    fn test_symbol_nodes_relabel() {
        
        let a_equals_b_plus_c = SymbolNode::new_generic(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new_generic(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("c".to_string())
                    ]
                )
            ]
        );

        let x_equals_b_plus_c = a_equals_b_plus_c.relabel("a".to_string(), "x".to_string());
        let expected = SymbolNode::new_generic(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("x".to_string()),
                SymbolNode::new_generic(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("c".to_string())
                    ]
                )
            ]
        );
        assert_eq!(x_equals_b_plus_c, expected);

        let x_equals_y_plus_y = x_equals_b_plus_c
            .relabel("b".to_string(), "y".to_string())
            .relabel("c".to_string(), "y".to_string());
        let expected = SymbolNode::new_generic(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("x".to_string()),
                SymbolNode::new_generic(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("y".to_string()),
                        SymbolNode::leaf_object("y".to_string())
                    ]
                )
            ]
        );
        assert_eq!(x_equals_y_plus_y, expected);

        let x_equals_x_plus_x = x_equals_y_plus_y
            .relabel("y".to_string(), "x".to_string());
        let expected = SymbolNode::new_generic(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("x".to_string()),
                SymbolNode::new_generic(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("x".to_string()),
                        SymbolNode::leaf_object("x".to_string())
                    ]
                )
            ]
        );
        assert_eq!(x_equals_x_plus_x, expected);

        let also_x_equals_x_plus_x = a_equals_b_plus_c
            .relabel_all(vec![
                ("b".to_string(), "x".to_string()),
                ("c".to_string(), "x".to_string()),
                ("a".to_string(), "x".to_string()),
            ].into_iter().collect()
        );

        assert_eq!(x_equals_x_plus_x, also_x_equals_x_plus_x);

        let a_equals_c_plus_b = a_equals_b_plus_c.relabel_all(vec![
            ("b".to_string(), "c".to_string()),
            ("c".to_string(), "b".to_string()),
        ].into_iter().collect());

        let expected = SymbolNode::new_generic(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new_generic(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("c".to_string()),
                        SymbolNode::leaf_object("b".to_string())
                    ]
                )
            ]
        );

        assert_eq!(a_equals_c_plus_b, expected);

    }

    #[test]
    fn test_symbol_node_identifies_conflicts() {

        let a_equals_b_plus_c = SymbolNode::new_generic(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new_generic(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf(Symbol::new("c".to_string(), Type::new_from_object("Variable".to_string())))
                    ]
                )
            ]
        );
        assert_eq!(a_equals_b_plus_c.get_type_conflicts(), vec![].into_iter().collect());

        let a_equals_b_plus_a = SymbolNode::new_generic(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new_generic(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("a".to_string())
                    ]
                )
            ]
        );
        assert_eq!(a_equals_b_plus_a.get_type_conflicts(), vec![].into_iter().collect());

        let a_equals_b_plus_a_conflicting = SymbolNode::new_generic(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new_generic(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf(Symbol::new("a".to_string(), Type::new_from_object("Variable".to_string())))
                    ]
                )
            ]
        );
        assert_eq!(a_equals_b_plus_a_conflicting.get_type_conflicts(), vec!["a".to_string()].into_iter().collect());
        
    }

    #[test]
    fn test_symbol_node_identifies_incorrect_argument_types() {

        assert_eq!(SymbolNode::leaf_object("a".to_string()).get_incorrect_argument_types(), vec![].into_iter().collect());
        assert_eq!(SymbolNode::leaf(Symbol::new("a".to_string(), Type::new_from_object("Variable".to_string()))).get_incorrect_argument_types(), vec![].into_iter().collect());
        assert_eq!(
            SymbolNode::new(
                Symbol::new_object("a".to_string()),
                vec![
                    SymbolNode::leaf_object("b".to_string()),
                    SymbolNode::leaf_object("c".to_string()),
                ]
            ).get_incorrect_argument_types(), vec!["a".to_string()].into_iter().collect());

        assert_eq!(
            SymbolNode::new(
                Symbol::new("a".to_string(), Type::new_generic_function_with_arguments(3)),
                vec![
                    SymbolNode::leaf_object("b".to_string()),
                    SymbolNode::leaf_object("c".to_string()),
                ]
         ).get_incorrect_argument_types(), vec!["a".to_string()].into_iter().collect());

        let a_equals_b_plus_c = SymbolNode::new_generic(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new_generic(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf(Symbol::new("c".to_string(), Type::new_from_object("Variable".to_string())))
                    ]
                )
            ]
        );

        assert_eq!(a_equals_b_plus_c.get_incorrect_argument_types(), vec![].into_iter().collect());

        let a_equals_b_plus_c_valid_equals = SymbolNode::new(
            Symbol::new(
                "=".to_string(), 
                Type::new(
                    "=".to_string(),
                    Type::new_generic_function_with_arguments(2)
                )
            ),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new_generic(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf(Symbol::new("c".to_string(), Type::new_from_object("Variable".to_string())))
                    ]
                )
            ],
        );
        assert_eq!(a_equals_b_plus_c_valid_equals.get_incorrect_argument_types(), vec![].into_iter().collect());

        let a_equals_b_plus_c_invalid_return_type = SymbolNode::new(
            Symbol::new("=".to_string(), Type::new_generic_function(vec![Type::new_from_object("Specific".to_string()), Type::new_from_object("Specific".to_string())], Type::new_from_object("Specific".to_string()))),
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new(
                    Symbol::new("+".to_string(), Type::new_generic_function(vec![Type::default(), Type::default()], Type::new_from_object("Different Specific".to_string()))),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf(Symbol::new("c".to_string(), Type::new_from_object("Variable".to_string())))
                    ]
                )
            ]
        );

        assert_eq!(a_equals_b_plus_c_invalid_return_type.get_incorrect_argument_types(), vec!["=".to_string()].into_iter().collect());
    }
}