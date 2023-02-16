use serde::{Serialize, Deserialize};


pub type TypeName = String;

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum Type {
    Object,
    Generic(Vec<Type>, Box<Type>),
    Type(TypeName, Box<Type>),
}

impl Default for Type {

    fn default() -> Type {
        Type::Object
    }

}

impl Type {

    pub fn new(name: TypeName, parent: Type) -> Type {
        Type::Type(name, Box::new(parent))
    }

    pub fn new_from_object(name: TypeName) -> Type {
        Type::Type(name, Box::new(Type::default()))
    }

    pub fn new_generic_function(args: Vec<Self>, return_type: Self) -> Self {
        Type::Generic(args, Box::new(return_type))
    }

    pub fn new_generic_function_with_arguments(n_arguments: usize) -> Self {
        Type::Generic((0..n_arguments).map(|_| Type::default()).collect(), Box::new(Type::default()))
    }

    pub fn get_parent(&self) -> Option<Type> {
        match self {
            Type::Object => None,
            Type::Generic(args, return_type) => None,
            Type::Type(_, parent) => Some(*parent.clone()),
        }
    }

    pub fn get_argument_types(&self) -> Vec<Type> {
        match self {
            Type::Object => Vec::new(),
            Type::Generic(args, _return_type) => args.clone(),
            Type::Type(_, parent) => parent.get_argument_types(),
        }
    }

    pub fn get_return_type(&self) -> Option<Type> {
        match self {
            Type::Object => None,
            Type::Generic(_args, return_type) => Some(*return_type.clone()),
            Type::Type(_, parent) => parent.get_return_type(),
        }
    }

    pub fn to_string(&self) -> String {
        let args = self.get_argument_types();
        let args_string = args.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", ");
        let return_type = self.get_return_type();
        match self {
            Type::Object => "Object".to_string(),
            Type::Generic(args, return_type) => {
                format!("Generic({}) -> {}", args_string, return_type.to_string()).to_string()
            },
            Type::Type(name, _parent) => {
                if args.len() == 0 {
                    name.clone()
                } else {
                    match return_type {
                        None => name.to_string(),
                        Some(t) => format!("{}({}) -> {}", name, args_string, t.to_string()).to_string()
                    }
                }
            },
        }
    }

    pub fn are_pairwise_supertypes_of(maybe_parents: Vec<Self>, maybe_children: Vec<Self>) -> bool {

        if maybe_parents.len() != maybe_children.len() {
            return false;
        }

        maybe_parents.iter().zip(maybe_children.iter()).all(|(parent, child)| parent.is_supertype_of(child))

    }

    pub fn are_pairwise_allowed_to_take(maybe_parents: Vec<Self>, maybe_children: Vec<Self>) -> bool {

        if maybe_parents.len() != maybe_children.len() {
            return false;
        }

        maybe_parents.iter().zip(maybe_children.iter()).all(|(parent, child)| parent.is_allowed_to_take(child))

    }

    pub fn is_supertype_of(&self, other: &Self) -> bool {
        if self == other {
            return true;
        }
        match other {
            Type::Object => false,
            Type::Generic(args, return_type) => {
                if let Type::Generic(self_args, self_return_type) = self {
                    if self_args.len() != args.len() {
                        return false;
                    }
                    self_args.iter().zip(args.iter()).all(
                            |(self_arg, arg)| self_arg.is_supertype_of(arg)
                        ) && self_return_type.is_supertype_of(return_type)
                } else {
                    false
                }
            },
            Type::Type(_, parent) => {
                if self == &**parent {
                    true
                } else {
                    self.is_supertype_of(parent)
                }
            },
        }
    }

    pub fn is_allowed_to_take(&self, other: &Self) -> bool {
        other.get_return_type()
            .map(|other_return_type| self.is_supertype_of(&other_return_type))
            .unwrap_or(self.is_supertype_of(other))
    }
    
}


#[cfg(test)]
mod test_type {
    use super::*;

    #[test]
    fn test_type_instantiates() {
        
        let quaternion = Type::new_from_object("Quaternion".to_string());
        let complex = Type::new("Complex".to_string(), quaternion.clone());
        let real = Type::new("Real".to_string(), complex.clone());
        let rational = Type::new("Rational".to_string(), real.clone());
        let irrational = Type::new("Irrational".to_string(), real.clone());

        assert_eq!(quaternion.get_parent(), Some(Type::Object));
        assert_eq!(complex.get_parent(), Some(quaternion));
        assert_eq!(real.get_parent(), Some(complex));
        assert_eq!(rational.get_parent(), Some(real.clone()));
        assert_eq!(irrational.get_parent(), Some(real));

    }

    #[test]
    fn test_type_to_string() {

        let quaternion = Type::new_from_object("Quaternion".to_string());
        let unary_function = Type::new("UnaryFunction".to_string(), Type::new_generic_function_with_arguments(1));
        let binary_function = Type::new("BinaryFunction".to_string(), Type::new_generic_function_with_arguments(2));
        let plus = Type::new("Plus".to_string(), Type::Generic(vec![quaternion.clone(), quaternion.clone()], Box::new(quaternion.clone())));

        assert_eq!(quaternion.to_string(), "Quaternion");
        assert_eq!(unary_function.to_string(), "UnaryFunction(Object) -> Object");
        assert_eq!(binary_function.to_string(), "BinaryFunction(Object, Object) -> Object");
        assert_eq!(plus.to_string(), "Plus(Quaternion, Quaternion) -> Quaternion");
        
    }

    #[test]
    fn test_type_is_supertype_of() {
        
        let quaternion = Type::new_from_object("Quaternion".to_string());
        let complex = Type::new("Complex".to_string(), quaternion.clone());
        let real = Type::new("Real".to_string(), complex.clone());
        let rational = Type::new("Rational".to_string(), real.clone());
        let irrational = Type::new("Irrational".to_string(), real.clone());
        let unary_function = Type::new("UnaryFunction".to_string(), Type::new_generic_function_with_arguments(1));
        let binary_function = Type::new("BinaryFunction".to_string(), Type::new_generic_function_with_arguments(2));
        let plus = Type::new("Plus".to_string(), Type::Generic(vec![quaternion.clone(), quaternion.clone()], Box::new(quaternion.clone())));

        assert_eq!(quaternion.is_supertype_of(&quaternion), true);
        assert_eq!(quaternion.is_supertype_of(&complex), true);
        assert_eq!(quaternion.is_supertype_of(&real), true);
        assert_eq!(quaternion.is_supertype_of(&rational), true);
        assert_eq!(quaternion.is_supertype_of(&irrational), true);

        assert_eq!(Type::Object.is_supertype_of(&quaternion), true);
        assert_eq!(Type::new_generic_function_with_arguments(1).is_supertype_of(&quaternion), false);
        assert_eq!(quaternion.is_supertype_of(&Type::Object), false);
        assert_eq!(quaternion.is_supertype_of(&unary_function), false);

        assert_eq!(irrational.is_supertype_of(&rational), false);
        assert_eq!(real.is_supertype_of(&rational), true);
        assert_eq!(real.is_supertype_of(&irrational), true);
        

        assert_eq!(Type::Object.is_supertype_of(&plus), false);
        assert_eq!(Type::new_generic_function_with_arguments(2).is_supertype_of(&plus), true);
        assert_eq!(Type::Generic(vec![quaternion.clone(), quaternion.clone()], Box::new(quaternion.clone())).is_supertype_of(&plus), true);
        assert_eq!(plus.is_supertype_of(&plus), true);
        assert_eq!(plus.is_supertype_of(&quaternion), false);

    }
}