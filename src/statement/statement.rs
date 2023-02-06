use std::collections::HashMap;



#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SymbolNode {
    root: String,
    children: Vec<SymbolNode>,
}

impl SymbolNode {

    pub fn new(root: String, children: Vec<SymbolNode>) -> SymbolNode {
        SymbolNode {
            root,
            children,
        }
    }

    pub fn leaf(root: String) -> SymbolNode {
        SymbolNode {
            root,
            children: Vec::new(),
        }
    }

    pub fn single_child(root: String, child: String) -> SymbolNode {
        SymbolNode {
            root,
            children: vec![SymbolNode::leaf(child)],
        }
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

    pub fn to_string(&self) -> String {
        if self.children.len() == 0 {
            self.root.clone()
        } else {
            let mut result = format!("{}(", self.root);
            let arguments = self.children.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
            result = format!("{}{})", result, arguments);
            result
        }
    }

    pub fn relabel(&self, old_label: String, new_label: String) -> Self {
        let new_children = self.children.iter().map(|child| child.relabel(old_label.clone(), new_label.clone())).collect();
        if self.root == old_label {
            Self::new(new_label, new_children)
        } else {
            Self::new( self.root.clone(), new_children)
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

        let a_equals_b_plus_c = SymbolNode::new(
            "=".to_string(), 
            vec![
                SymbolNode::leaf("a".to_string()),
                SymbolNode::new(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf("b".to_string()),
                        SymbolNode::leaf("c".to_string())
                    ]
                )
            ]
        );

        assert_eq!(a_equals_b_plus_c.get_depth(), 3);
        assert_eq!(a_equals_b_plus_c.get_n_children(), 2);
        assert_eq!(a_equals_b_plus_c.to_string(), "=(a,+(b,c))");

        let n_factorial = SymbolNode::single_child("!".to_string(), "n".to_string());
        
        // prod_{i = 1}^{n} i
        // Point: Expand 5! to 5 * 4 * 3 * 2 * 1, which is going to require a transformation like:
        // 5! => prod_{i = 1}^{5} i => 5 * 4 * 3 * 2 * 1
        let n_factorial_definition = SymbolNode::new(
            "Prod".to_string(),
            vec![
                SymbolNode::leaf("i".to_string()), // i is the index variable
                SymbolNode::leaf("1".to_string()), // 1 is the lower bound
                SymbolNode::leaf("n".to_string()), // n is the upper bound
                SymbolNode::leaf("i".to_string()), // i is the expression to be multiplied
            ]
        );

        let factorial_definition = SymbolNode::new(
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
        
        let a_equals_b_plus_c = SymbolNode::new(
            "=".to_string(), 
            vec![
                SymbolNode::leaf("a".to_string()),
                SymbolNode::new(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf("b".to_string()),
                        SymbolNode::leaf("c".to_string())
                    ]
                )
            ]
        );

        let x_equals_b_plus_c = a_equals_b_plus_c.relabel("a".to_string(), "x".to_string());
        let expected = SymbolNode::new(
            "=".to_string(), 
            vec![
                SymbolNode::leaf("x".to_string()),
                SymbolNode::new(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf("b".to_string()),
                        SymbolNode::leaf("c".to_string())
                    ]
                )
            ]
        );
        assert_eq!(x_equals_b_plus_c, expected);

        let x_equals_y_plus_y = x_equals_b_plus_c
            .relabel("b".to_string(), "y".to_string())
            .relabel("c".to_string(), "y".to_string());
        let expected = SymbolNode::new(
            "=".to_string(), 
            vec![
                SymbolNode::leaf("x".to_string()),
                SymbolNode::new(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf("y".to_string()),
                        SymbolNode::leaf("y".to_string())
                    ]
                )
            ]
        );
        assert_eq!(x_equals_y_plus_y, expected);

        let x_equals_x_plus_x = x_equals_y_plus_y
            .relabel("y".to_string(), "x".to_string());
        let expected = SymbolNode::new(
            "=".to_string(), 
            vec![
                SymbolNode::leaf("x".to_string()),
                SymbolNode::new(
                    "+".to_string(),
                    vec![
                        SymbolNode::leaf("x".to_string()),
                        SymbolNode::leaf("x".to_string())
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