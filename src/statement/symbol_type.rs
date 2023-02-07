
pub type TypeName = String;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum Type {
    ObjectWithArguments(Vec<Type>),
    Type(TypeName, Box<Type>),
}

impl Default for Type {

    fn default() -> Type {
        Type::ObjectWithArguments(Vec::new())
    }

}

impl Type {

    pub fn new(name: TypeName, parent: Type) -> Type {
        Type::Type(name, Box::new(parent))
    }

    pub fn new_from_object(name: TypeName) -> Type {
        Type::Type(name, Box::new(Type::ObjectWithArguments(Vec::new())))
    }

    pub fn get_parent(&self) -> Type {
        match self {
            Type::ObjectWithArguments(args) => Type::ObjectWithArguments(args.clone()),
            Type::Type(_, parent) => *parent.clone(),
        }
    }

    pub fn get_argument_types(&self) -> Vec<Type> {
        match self {
            Type::ObjectWithArguments(args) => args.clone(),
            Type::Type(_, parent) => parent.get_argument_types(),
        }
    }

    pub fn to_string(&self) -> String {
        let args = self.get_argument_types();
        let args_string = args.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", ");
        match self {
            Type::ObjectWithArguments(args) => {
                if args.len() == 0 {
                    "Object".to_string()
                } else {
                    format!("Object({})", args_string).to_string()
                }
            },
            Type::Type(name, _parent) => {
                if args.len() == 0 {
                    name.clone()
                } else {
                    format!("{}({})", name, args_string).to_string()
                }
            },
        }
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

        assert_eq!(quaternion.get_parent(), Type::ObjectWithArguments(Vec::new()));
        assert_eq!(complex.get_parent(), quaternion);
        assert_eq!(real.get_parent(), complex);
        assert_eq!(rational.get_parent(), real);
        assert_eq!(irrational.get_parent(), real);

    }

    #[test]
    fn test_type_to_string() {

        let quaternion = Type::new_from_object("Quaternion".to_string());
        let unary_function = Type::new("UnaryFunction".to_string(), Type::ObjectWithArguments(vec![Type::default()]));
        let binary_function = Type::new("BinaryFunction".to_string(), Type::ObjectWithArguments(vec![Type::default(), Type::default()]));
        let plus = Type::new("Plus".to_string(), Type::ObjectWithArguments(vec![quaternion.clone(), quaternion.clone()]));

        assert_eq!(quaternion.to_string(), "Quaternion");
        assert_eq!(unary_function.to_string(), "UnaryFunction(Object)");
        assert_eq!(binary_function.to_string(), "BinaryFunction(Object, Object)");
        assert_eq!(plus.to_string(), "Plus(Quaternion, Quaternion)");
        
    }
}