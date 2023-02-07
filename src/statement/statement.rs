use std::collections::HashMap;

use crate::statement::symbol_type::Type;

#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct Symbol {
    name: String,
    symbol_type: Type,
}

impl Symbol {
    
        pub fn new(name: String, symbol_type: Type) -> Self {
            Self {
                name,
                symbol_type,
            }
        }

        pub fn new_object(name: String) -> Self {
            Self::new(name, Type::default())
        }
    
        pub fn get_name(&self) -> String {
            self.name.clone()
        }
    
        pub fn get_type(&self) -> Type {
            self.symbol_type.clone()
        }
    
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

    pub fn new_object(root: String, children: Vec<Self>) -> Self {
        Self::new(Symbol::new_object(root), children)
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

    pub fn get_n_children(&self) -> usize {
        self.children.len()
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
        let new_children = self.children.iter().map(|child| child.relabel(old_label.clone(), new_label.clone())).collect();
        if self.root.get_name() == old_label {
            Self::new(Symbol::new(new_label, self.root.get_type()), new_children)
        } else {
            Self::new(self.root.clone(), new_children)
        }
    }

    pub fn relabel_all(&self, relabelling: Vec<(String, String)>) -> Self {
        relabelling.into_iter().fold(self.clone(), |acc, (old_label, new_label)| acc.relabel(old_label, new_label))
    }

}


#[cfg(test)]
mod test_statement {
    use super::*;


    #[test]
    fn test_symbol_node_initializes() {

        let a_equals_b_plus_c = SymbolNode::new_object(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new_object(
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
        let n_factorial_definition = SymbolNode::new_object(
            "Prod".to_string(),
            vec![
                SymbolNode::leaf_object("i".to_string()), // i is the index variable
                SymbolNode::leaf_object("1".to_string()), // 1 is the lower bound
                SymbolNode::leaf_object("n".to_string()), // n is the upper bound
                SymbolNode::leaf_object("i".to_string()), // i is the expression to be multiplied
            ]
        );

        let factorial_definition = SymbolNode::new_object(
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
        
        let a_equals_b_plus_c = SymbolNode::new_object(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("a".to_string()),
                SymbolNode::new_object(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf_object("b".to_string()),
                        SymbolNode::leaf_object("c".to_string())
                    ]
                )
            ]
        );

        let x_equals_b_plus_c = a_equals_b_plus_c.relabel("a".to_string(), "x".to_string());
        let expected = SymbolNode::new_object(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("x".to_string()),
                SymbolNode::new_object(
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
        let expected = SymbolNode::new_object(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("x".to_string()),
                SymbolNode::new_object(
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
        let expected = SymbolNode::new_object(
            "=".to_string(), 
            vec![
                SymbolNode::leaf_object("x".to_string()),
                SymbolNode::new_object(
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
            ]
        );

        assert_eq!(x_equals_x_plus_x, also_x_equals_x_plus_x);

    }
}