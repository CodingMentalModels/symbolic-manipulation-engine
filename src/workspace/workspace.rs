use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::to_string;
use ts_rs::TS;

use crate::{
    context::context::Context,
    parsing::{
        interpretation::{DisplayInterpretation, Interpretation, InterpretedType},
        parser::{Parser, ParserError},
    },
    symbol::{
        symbol_node::{SymbolName, SymbolNode, SymbolNodeAddress, SymbolNodeError},
        symbol_type::{
            DisplayGeneratedType, DisplayTypeHierarchyNode, GeneratedType, Type, TypeError,
            TypeHierarchy, TypeName,
        },
        transformation::{ExplicitTransformation, Transformation, TransformationError},
    },
};

type SymbolNodeString = String;
type ProvenanceString = String;
type DisplayTransformation = String;
type StatementIndex = usize;
type TransformationIndex = usize;

#[derive(Clone, Debug, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DisplayWorkspace {
    types: Vec<DisplayTypeHierarchyNode>,
    generated_types: Vec<DisplayGeneratedType>,
    interpretations: Vec<DisplayInterpretation>,
    statements: Vec<DisplaySymbolNode>,
    transformations: Vec<DisplayTransformation>,
    provenance: Vec<DisplayProvenance>,
}

impl DisplayWorkspace {
    fn from_workspace(workspace: &Workspace) -> Result<Self, WorkspaceError> {
        Ok(DisplayWorkspace {
            types: Vec::<DisplayTypeHierarchyNode>::from(&workspace.types),
            generated_types: workspace
                .generated_types
                .iter()
                .map(|g| DisplayGeneratedType::from(g))
                .collect(),
            interpretations: workspace
                .interpretations
                .iter()
                .map(|i| DisplayInterpretation::from(i))
                .collect(),
            statements: workspace.get_display_symbol_nodes()?,
            transformations: workspace
                .transformations
                .iter()
                .map(|t| t.to_interpreted_string(&workspace.interpretations))
                .collect(),
            provenance: workspace.get_display_provenances(),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Workspace {
    types: TypeHierarchy,
    generated_types: Vec<GeneratedType>,
    interpretations: Vec<Interpretation>,
    statements: Vec<SymbolNode>,
    transformations: Vec<Transformation>,
    provenance: Vec<Provenance>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new(TypeHierarchy::new(), vec![], vec![])
    }
}

impl Workspace {
    pub fn new(
        types: TypeHierarchy,
        generated_types: Vec<GeneratedType>,
        interpretations: Vec<Interpretation>,
    ) -> Workspace {
        Self {
            types,
            generated_types,
            interpretations,
            statements: vec![],
            transformations: vec![],
            provenance: vec![],
        }
    }

    pub fn get_types(&self) -> &TypeHierarchy {
        &self.types
    }

    pub fn get_interpretations(&self) -> &Vec<Interpretation> {
        &self.interpretations
    }

    pub fn try_import_context(&mut self, context: Context) -> Result<(), WorkspaceError> {
        let mut shared_types = self.types.get_shared_types(context.get_types());
        shared_types.remove(&Type::Object);
        shared_types.remove(&Type::Join);
        if shared_types.len() > 0 {
            return Err(WorkspaceError::AttemptedToImportAmbiguousTypes(
                shared_types,
            ));
        }
        let new_types = self
            .types
            .union(context.get_types())
            .map_err(|e| WorkspaceError::from(e))?;

        if context.get_generated_types().len() > 0 {
            if self.generated_types.len() > 0
                && &self.generated_types != context.get_generated_types()
            {
                return Err(WorkspaceError::UnsupportedOperation("For simplicity, we don't support importing Contexts with different generated types than the workspace.".to_string()));
            }
        }

        let mut new_generated_types = self.generated_types.clone();
        new_generated_types.append(&mut context.get_generated_types().clone());

        if context.get_interpretations().len() > 0 {
            if self.interpretations.len() > 0
                && &self.interpretations != context.get_interpretations()
            {
                return Err(WorkspaceError::UnsupportedOperation("For simplicity, we don't support importing Contexts with different interpretations than the workspace.".to_string()));
            }
        }

        let mut new_interpretations = self.interpretations.clone();
        new_interpretations.append(&mut context.get_interpretations().clone());

        self.types = new_types;
        self.generated_types = new_generated_types;
        self.interpretations = new_interpretations;
        self.transformations = context.get_transformations().clone();
        Ok(())
    }

    pub fn add_parsed_hypothesis(&mut self, s: &str) -> Result<SymbolNode, WorkspaceError> {
        let parsed = self.parse_from_string(s)?;
        self.generate_types(&parsed)?;
        self.add_hypothesis(parsed.clone())?;
        Ok(parsed)
    }

    pub fn add_parsed_joint_transformation(
        &mut self,
        left_from: &str,
        right_from: &str,
        to: &str,
    ) -> Result<Transformation, WorkspaceError> {
        let parsed_from =
            (self.parse_from_string(left_from)?).join(self.parse_from_string(right_from)?);
        self.generate_types(&parsed_from)?;
        let parsed_to = self.parse_from_string(to)?;
        self.generate_types(&parsed_to)?;
        let transformation: Transformation =
            ExplicitTransformation::new(parsed_from, parsed_to).into();
        self.add_transformation(transformation.clone())?;
        Ok(transformation)
    }

    pub fn add_parsed_transformation(
        &mut self,
        from: &str,
        to: &str,
    ) -> Result<Transformation, WorkspaceError> {
        let parsed_from = self.parse_from_string(from)?;
        self.generate_types(&parsed_from)?;
        let parsed_to = self.parse_from_string(to)?;
        self.generate_types(&parsed_to)?;
        let transformation: Transformation =
            ExplicitTransformation::new(parsed_from, parsed_to).into();
        self.add_transformation(transformation.clone())?;
        Ok(transformation)
    }

    fn parse_from_string(&self, s: &str) -> Result<SymbolNode, WorkspaceError> {
        let parser = Parser::new(self.interpretations.clone());
        parser
            .parse_from_string(parser.get_interpretation_custom_tokens(), s)
            .map_err(|e| e.into())
    }

    pub fn get_statements(&self) -> &Vec<SymbolNode> {
        &self.statements
    }

    pub fn get_statement_pairs(&self) -> Vec<SymbolNode> {
        // Returns pairs of statements for use in joint transforms
        // Returns both orders as well as duplicates since those are valid joint transform args
        let left = self.get_statements();
        let right = self.get_statements();
        let mut to_return = Vec::new();
        for i in 0..left.len() {
            for j in 0..right.len() {
                to_return.push(left[i].clone().join(right[j].clone()));
            }
        }
        to_return
    }

    pub fn get_statement(&self, index: StatementIndex) -> Result<SymbolNode, WorkspaceError> {
        if self.statement_index_is_invalid(index) {
            Err(WorkspaceError::InvalidStatementIndex)
        } else {
            Ok(self.statements[index].clone())
        }
    }

    pub fn add_interpretation(
        &mut self,
        interpretation: Interpretation,
    ) -> Result<(), WorkspaceError> {
        match interpretation.get_output_type() {
            InterpretedType::Type(t) => {
                if !self.types.contains_type(&t) {
                    return Err(WorkspaceError::InvalidType(t));
                }
            }
            _ => {}
        }
        self.interpretations.push(interpretation);
        Ok(())
    }

    pub fn remove_interpretation(
        &mut self,
        interpretation_index: usize,
    ) -> Result<Interpretation, WorkspaceError> {
        if interpretation_index >= self.interpretations.len() {
            return Err(WorkspaceError::InvalidInterpretationIndex);
        }
        Ok(self.interpretations.remove(interpretation_index))
    }

    pub fn add_type_to_parent(&mut self, t: Type, parent: Type) -> Result<Type, WorkspaceError> {
        self.types
            .add_child_to_parent(t, parent)
            .map_err(|e| e.into())
    }

    pub fn add_hypothesis(&mut self, statement: SymbolNode) -> Result<(), WorkspaceError> {
        self.types
            .binds_statement_or_error(&statement)
            .map_err(|x| WorkspaceError::from(x))?;
        self.generate_types_in_bulk(vec![statement.clone()].into_iter().collect())?;
        self.statements.push(statement);
        self.provenance.push(Provenance::Hypothesis);
        Ok(())
    }

    fn add_derived_statement(&mut self, statement: SymbolNode, provenance: Provenance) {
        self.statements.push(statement);
        self.provenance.push(provenance);
    }

    pub fn get_transformations(&self) -> &Vec<Transformation> {
        &self.transformations
    }

    pub fn get_transformation(
        &self,
        index: TransformationIndex,
    ) -> Result<Transformation, WorkspaceError> {
        if self.transformation_index_is_invalid(index) {
            Err(WorkspaceError::InvalidTransformationIndex)
        } else {
            Ok(self.transformations[index].clone())
        }
    }

    pub fn add_transformation(
        &mut self,
        transformation: Transformation,
    ) -> Result<(), WorkspaceError> {
        self.types.binds_transformation_or_error(&transformation)?;
        match &transformation {
            Transformation::ExplicitTransformation(t) => {
                self.generate_types_in_bulk(
                    vec![t.get_from().clone(), t.get_to().clone()]
                        .into_iter()
                        .collect(),
                )?;
            }
            Transformation::AdditionAlgorithm(_) => {
                // Nothing to generate
            }
            Transformation::ApplyToBothSidesTransformation(t) => {
                let transformation = t.get_transformation();
                self.generate_types_in_bulk(
                    vec![
                        transformation.get_from().clone(),
                        transformation.get_to().clone(),
                    ]
                    .into_iter()
                    .collect(),
                )?;
            }
        }
        self.transformations.push(transformation);
        Ok(())
    }

    pub fn get_valid_transformations(&self, partial_statement: &str) -> Vec<SymbolNode> {
        // TODO Try to complete partial statements
        let desired = match self.parse_from_string(partial_statement) {
            Err(_) => {
                return vec![];
            }
            Ok(s) => s,
        };
        for transformation in self.get_transformations() {
            let statements = if transformation.is_joint_transform() {
                self.get_statement_pairs()
            } else {
                self.get_statements().clone()
            };
            for statement in statements {
                let valid_transformations =
                    transformation.get_valid_transformations(self.get_types(), &statement);
                if valid_transformations.contains(&desired)
                    && !self.get_statements().contains(&desired)
                {
                    return vec![desired.clone()];
                }
            }
        }
        return vec![];
    }

    pub fn try_transform_into_parsed(
        &mut self,
        desired: &str,
    ) -> Result<SymbolNode, WorkspaceError> {
        let parsed = self.parse_from_string(desired)?;
        self.generate_types(&parsed)?;
        self.try_transform_into(parsed)
    }

    pub fn try_transform_into(
        &mut self,
        desired: SymbolNode,
    ) -> Result<SymbolNode, WorkspaceError> {
        if self.statements.contains(&desired) {
            return Err(WorkspaceError::StatementsAlreadyInclude(desired.clone()));
        }
        for (transform_idx, transform) in self.transformations.iter().enumerate() {
            let statements = if transform.is_joint_transform() {
                self.get_statement_pairs()
            } else {
                self.get_statements().clone()
            };
            for (statement_idx, statement) in statements.iter().enumerate() {
                match transform.try_transform_into(self.get_types(), &statement, &desired) {
                    Ok(output) => {
                        // TODO Derive the appropriate transform addresses
                        let provenance =
                            Provenance::Derived((statement_idx, transform_idx, vec![]));
                        self.add_derived_statement(output.clone(), provenance);
                        return Ok(output);
                    }
                    Err(_) => { // Do nothing, keep trying transformations
                    }
                }
            }
        }
        return Err(WorkspaceError::NoTransformationsPossible);
    }

    pub fn get_provenance_lineage(
        &self,
        index: StatementIndex,
    ) -> Result<Vec<Provenance>, WorkspaceError> {
        let mut provenance = Vec::new();
        let mut current_index = index;
        loop {
            let current_provenance = self.get_provenance(current_index)?;
            provenance.push(current_provenance.clone());
            match current_provenance {
                Provenance::Hypothesis => break,
                Provenance::Derived((parent_index, _, _)) => current_index = parent_index,
            }
        }
        Ok(provenance)
    }

    pub fn get_provenance(&self, index: StatementIndex) -> Result<Provenance, WorkspaceError> {
        if self.statement_index_is_invalid(index) {
            return Err(WorkspaceError::InvalidStatementIndex);
        }

        Ok(self.provenance[index].clone())
    }

    pub fn get_display_provenances(&self) -> Vec<DisplayProvenance> {
        let mut to_return = Vec::new();
        for i in 0..self.statements.len() {
            // TODO Ensure that this expectation is always valid
            let provenance = self
                .get_display_provenance(i)
                .expect("We control the statements and provenance generation.");
            to_return.push(provenance);
        }
        to_return
    }

    pub fn get_display_provenance(
        &self,
        index: StatementIndex,
    ) -> Result<DisplayProvenance, WorkspaceError> {
        let provenance = self.get_provenance(index)?;
        match provenance {
            Provenance::Hypothesis => Ok(DisplayProvenance::Hypothesis),
            Provenance::Derived((s, t, indices)) => {
                let statement = self.get_statement(s)?;
                let transformation = self.get_transformation(t)?;
                Ok(DisplayProvenance::Derived(DerivedDisplayProvenance::new(
                    statement.to_interpreted_string(&self.interpretations),
                    transformation.to_interpreted_string(&self.interpretations),
                    indices,
                )))
            }
        }
    }

    pub fn get_display_symbol_nodes(&self) -> Result<Vec<DisplaySymbolNode>, WorkspaceError> {
        let mut to_return = Vec::new();
        for i in 0..self.statements.len() {
            let node = self.get_display_symbol_node(i)?;
            to_return.push(node);
        }
        Ok(to_return)
    }

    pub fn get_display_symbol_node(
        &self,
        index: usize,
    ) -> Result<DisplaySymbolNode, WorkspaceError> {
        let statement = self.get_statement(index)?;
        let (interpreted_string, type_map) = statement
            .to_interpreted_string_and_type_map(&self.interpretations)
            .map_err(|e| Into::<WorkspaceError>::into(e))?;
        let provenance = self.get_display_provenance(index)?;
        Ok(DisplaySymbolNode::new(
            interpreted_string,
            type_map
                .into_iter()
                .map(|(name, t)| (name, t.to_string()))
                .collect(),
            provenance.get_from_statement(),
            provenance.get_from_transformation(),
        ))
    }

    pub fn get_generated_types(&self) -> &Vec<GeneratedType> {
        &self.generated_types
    }

    pub fn add_generated_type(&mut self, generated_type: GeneratedType) {
        self.generated_types.push(generated_type);
    }

    pub fn get_instantiated_transformations(
        &self,
    ) -> Result<HashSet<Transformation>, WorkspaceError> {
        let substatements = self
            .get_statements()
            .iter()
            .map(|statement| statement.get_substatements())
            .flatten()
            .collect::<HashSet<_>>();

        let mut to_return = vec![].into_iter().collect::<HashSet<_>>();
        for transform in self.get_arbitrary_transformations() {
            to_return = to_return
                .union(
                    &mut transform
                        .instantiate_arbitrary_nodes(&substatements)
                        .map_err(|e| Into::<WorkspaceError>::into(e))?,
                )
                .cloned()
                .collect();
        }

        to_return = to_return
            .union(
                &self
                    .get_non_arbitrary_transformations()
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .cloned()
            .collect();

        return Ok(to_return);
    }

    fn get_arbitrary_transformations(&self) -> HashSet<Transformation> {
        self.transformations
            .iter()
            .filter(|t| t.contains_arbitrary_nodes())
            .cloned()
            .collect()
    }

    fn get_non_arbitrary_transformations(&self) -> HashSet<Transformation> {
        self.transformations
            .iter()
            .filter(|t| !t.contains_arbitrary_nodes())
            .cloned()
            .collect()
    }

    pub fn to_json(&self) -> Result<String, WorkspaceError> {
        let display_workspace = DisplayWorkspace::from_workspace(self)?;
        to_string(&display_workspace).map_err(|e| WorkspaceError::UnableToSerialize(e.to_string()))
    }

    pub fn serialize(&self) -> Result<String, WorkspaceError> {
        toml::to_string(self).map_err(|e| WorkspaceError::UnableToSerialize(e.to_string()))
    }

    pub fn deserialize(serialized: &str) -> Result<Workspace, toml::de::Error> {
        toml::from_str(serialized)
    }

    fn statement_index_is_invalid(&self, index: StatementIndex) -> bool {
        index >= self.statements.len() || index >= self.provenance.len()
    }

    fn transformation_index_is_invalid(&self, index: TransformationIndex) -> bool {
        index >= self.transformations.len()
    }

    fn add_type_to_parents(
        &mut self,
        t: Type,
        parents: &HashSet<Type>,
    ) -> Result<Type, WorkspaceError> {
        self.types
            .add_child_to_parents(t, parents)
            .map_err(|e| e.into())
    }

    fn get_parent_types(&self, t: &Type) -> Result<HashSet<Type>, WorkspaceError> {
        self.types.get_parents(t).map_err(|e| e.into())
    }

    fn generate_types_in_bulk(
        &mut self,
        statements: HashSet<SymbolNode>,
    ) -> Result<(), WorkspaceError> {
        for statement in statements {
            match self.generate_types(&statement) {
                Ok(()) => {}
                Err(e) => {
                    return Err(e.into());
                }
            };
        }
        Ok(())
    }

    fn generate_types(&mut self, statement: &SymbolNode) -> Result<(), WorkspaceError> {
        for generated_type in self.get_generated_types().clone() {
            let result: Option<WorkspaceError> = generated_type
                .generate(statement)
                .into_iter()
                .map(|(t, parents)| self.add_type_to_parents(t, &parents))
                .filter(|r| match r {
                    Err(WorkspaceError::TypeHierarchyAlreadyIncludes(prior_type)) => {
                        // TODO Check that prior_type parents == parents
                        false
                    }
                    Err(_) => true,
                    _ => false,
                })
                .map(|r| r.expect_err("We just checked that it's an error."))
                .next();
            match result {
                Some(e) => return Err(e.into()),
                None => {}
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(tag = "kind", rename_all = "camelCase")]
#[ts(export)]
pub enum DisplayProvenance {
    Hypothesis,
    Derived(DerivedDisplayProvenance),
}

impl DisplayProvenance {
    pub fn get_from_statement(&self) -> Option<String> {
        match self {
            Self::Hypothesis => None,
            Self::Derived(derived) => Some(derived.get_from_statement().clone()),
        }
    }

    pub fn get_from_transformation(&self) -> Option<String> {
        match self {
            Self::Hypothesis => None,
            Self::Derived(derived) => Some(derived.get_from_transformation().clone()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DerivedDisplayProvenance {
    from_transformation: DisplayTransformation,
    from_statement: SymbolNodeString,
    applied_addresses: Vec<SymbolNodeAddress>,
}

impl DerivedDisplayProvenance {
    pub fn new(
        from_transformation: DisplayTransformation,
        from_statement: SymbolNodeString,
        applied_addresses: Vec<SymbolNodeAddress>,
    ) -> Self {
        Self {
            from_transformation,
            from_statement,
            applied_addresses,
        }
    }

    pub fn get_from_statement(&self) -> &String {
        &self.from_statement
    }

    pub fn get_from_transformation(&self) -> &String {
        &self.from_transformation
    }

    pub fn get_applied_addresses(&self) -> &Vec<SymbolNodeAddress> {
        &self.applied_addresses
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Provenance {
    Hypothesis,
    Derived((TransformationIndex, StatementIndex, Vec<SymbolNodeAddress>)),
}

#[derive(Clone, Debug, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DisplaySymbolNode {
    interpreted_string: String,
    types: Vec<(SymbolName, TypeName)>,
    from_statement: Option<String>,
    from_transformation: Option<String>,
}

impl DisplaySymbolNode {
    pub fn new(
        symbol_node_string: String,
        types: Vec<(SymbolName, TypeName)>,
        from_statement: Option<String>,
        from_transformation: Option<String>,
    ) -> Self {
        Self {
            interpreted_string: symbol_node_string,
            types,
            from_statement,
            from_transformation,
        }
    }

    pub fn get_interpreted_string(&self) -> &String {
        &self.interpreted_string
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceError {
    InvalidStatementIndex,
    InvalidTransformationIndex,
    InvalidInterpretationIndex,
    InvalidTransformationAddress,
    CantContainArbitraryNode,
    ParserError(ParserError),
    UnableToSerialize(String),
    TransformationError(TransformationError),
    StatementContainsTypesNotInHierarchy(HashSet<Type>),
    IncompatibleTypeRelationships(HashSet<Type>),
    ConflictingTypes(String, Type, Type),
    StatementsAlreadyInclude(SymbolNode),
    TypeHierarchyAlreadyIncludes(Type),
    InvalidType(Type),
    ParentTypeNotFound(Type),
    UnsupportedOperation(String),
    ArbitraryNodeHasNonOneChildren,
    NoTransformationsPossible,
    InvalidTypeError(TypeError),
    InvalidSymbolNodeError(SymbolNodeError),
    InvalidTransformationError(TransformationError),
    AttemptedToImportAmbiguousTypes(HashSet<Type>),
}

impl From<ParserError> for WorkspaceError {
    fn from(value: ParserError) -> Self {
        Self::ParserError(value)
    }
}

impl From<TransformationError> for WorkspaceError {
    fn from(value: TransformationError) -> Self {
        match value {
            TransformationError::ArbitraryNodeHasNonOneChildren => {
                Self::ArbitraryNodeHasNonOneChildren
            }
            e => Self::InvalidTransformationError(e),
        }
    }
}

impl From<SymbolNodeError> for WorkspaceError {
    fn from(value: SymbolNodeError) -> Self {
        match value {
            SymbolNodeError::ConflictingTypes(name, t_0, t_1) => {
                WorkspaceError::ConflictingTypes(name, t_0, t_1)
            }
            e => WorkspaceError::InvalidSymbolNodeError(e),
        }
    }
}

impl From<TypeError> for WorkspaceError {
    fn from(value: TypeError) -> Self {
        match value {
            TypeError::InvalidType(t) => WorkspaceError::InvalidType(t),
            TypeError::TypeHierarchyAlreadyIncludes(t) => {
                WorkspaceError::TypeHierarchyAlreadyIncludes(t)
            }
            TypeError::StatementIncludesTypesNotInHierarchy(types) => {
                Self::StatementContainsTypesNotInHierarchy(types)
            }
            TypeError::IncompatibleTypeRelationships(types) => {
                Self::IncompatibleTypeRelationships(types)
            }
            TypeError::ParentNotFound(t) => WorkspaceError::ParentTypeNotFound(t),
            e => Self::InvalidTypeError(e),
        }
    }
}

#[cfg(test)]
mod test_workspace {

    use crate::{
        context::context::Context,
        parsing::{interpretation::Interpretation, parser::Parser},
        symbol::{
            symbol_node::Symbol,
            symbol_type::{GeneratedType, GeneratedTypeCondition, TypeHierarchy},
        },
    };

    use super::*;

    #[test]
    fn test_workspace_try_transform_into_with_arbitrary() {
        let mut types = TypeHierarchy::chain(vec!["Boolean".into(), "=".into()]).unwrap();
        types
            .add_child_to_parent("^".into(), "Boolean".into())
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("^".into(), 1, "^".into()),
            Interpretation::singleton("p".into(), "Boolean".into()),
            Interpretation::singleton("q".into(), "Boolean".into()),
            Interpretation::singleton("r".into(), "Boolean".into()),
            Interpretation::singleton("s".into(), "Boolean".into()),
            Interpretation::singleton("s".into(), "Boolean".into()),
            Interpretation::arbitrary_functional("Any".into(), 99, "Boolean".into()),
        ];

        let mut workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        workspace
            .add_parsed_transformation("p=q", "Any(p)=Any(q)")
            .unwrap();

        workspace.add_parsed_hypothesis("p=q").unwrap();

        let expected = workspace.parse_from_string("p^s=q^s").unwrap();
        assert_eq!(
            workspace.try_transform_into_parsed("p^s=q^s").unwrap(),
            expected
        );
        assert!(workspace.get_statements().contains(&expected));
    }

    #[test]
    fn test_workspace_instantiates_arbitrary_transforms() {
        let mut types = TypeHierarchy::chain(vec!["Boolean".into(), "=".into()]).unwrap();
        types
            .add_child_to_parent("^".into(), "Boolean".into())
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("^".into(), 1, "^".into()),
            Interpretation::singleton("p".into(), "Boolean".into()),
            Interpretation::singleton("q".into(), "Boolean".into()),
            Interpretation::singleton("r".into(), "Boolean".into()),
            Interpretation::singleton("s".into(), "Boolean".into()),
            Interpretation::singleton("s".into(), "Boolean".into()),
            Interpretation::arbitrary_functional("Any".into(), 99, "Boolean".into()),
        ];

        let mut workspace = Workspace::new(types.clone(), vec![], interpretations.clone());

        workspace
            .add_parsed_transformation("p=q", "Any(p)=Any(q)")
            .unwrap();
        workspace.add_parsed_hypothesis("p=q").unwrap();
        workspace.add_parsed_hypothesis("p^s").unwrap();

        let instantiate = |s: &str| {
            ExplicitTransformation::new(
                workspace.parse_from_string("p=q").unwrap(),
                workspace.parse_from_string(s).unwrap(),
            )
            .into()
        };

        let expected = vec![
            instantiate("p=q"),
            instantiate("(p=q)=(q=q)"),
            instantiate("(p=p)=(p=q)"),
            instantiate("(p^s)=(q^s)"),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            workspace.get_instantiated_transformations().unwrap(),
            expected
        );

        //        workspace.add_parsed_hypothesis("t^p").unwrap();
        //        let t_and_equal = workspace.parse_from_string("t^p=t^q").unwrap();
        //        let expected = vec![and_s_equal, t_and_equal].into_iter().collect();
        //        assert_eq!(workspace.get_instantiated_transformations(), expected);
    }

    #[test]
    fn test_workspace_disallows_arbitrary_statements() {
        let mut types = TypeHierarchy::chain(vec!["Boolean".into(), "=".into()]).unwrap();
        types
            .add_child_to_parent("^".into(), "Boolean".into())
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("^".into(), 1, "^".into()),
            Interpretation::singleton("p".into(), "Boolean".into()),
            Interpretation::singleton("q".into(), "Boolean".into()),
            Interpretation::singleton("r".into(), "Boolean".into()),
            Interpretation::singleton("s".into(), "Boolean".into()),
            Interpretation::singleton("s".into(), "Boolean".into()),
            Interpretation::arbitrary_functional("Any".into(), 99, "Boolean".into()),
        ];

        let mut workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        assert_eq!(
            workspace.add_parsed_hypothesis("Any(p)"),
            Err(WorkspaceError::CantContainArbitraryNode)
        );
        assert_eq!(
            workspace.add_parsed_hypothesis("Any(p)=q"),
            Err(WorkspaceError::CantContainArbitraryNode)
        );

        workspace.add_parsed_hypothesis("p=q").unwrap();
        assert_eq!(
            workspace.try_transform_into_parsed("Any(p)=Any(q)"),
            Err(WorkspaceError::CantContainArbitraryNode)
        );
    }

    #[test]
    fn test_workspace_try_transform_into() {
        let mut types = TypeHierarchy::chain(vec!["Real".into(), "Integer".into()]).unwrap();
        types
            .add_child_to_parent("=".into(), "Real".into())
            .unwrap();
        types
            .add_child_to_parent("+".into(), "Real".into())
            .unwrap();
        types
            .add_chain(vec!["Proposition".into(), "^".into()])
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("+".into(), 6, "+".into()),
            Interpretation::singleton("x".into(), "Real".into()),
            Interpretation::singleton("y".into(), "Real".into()),
            Interpretation::singleton("j".into(), "Integer".into()),
            Interpretation::singleton("k".into(), "Integer".into()),
            Interpretation::singleton("a".into(), "Integer".into()),
            Interpretation::singleton("b".into(), "Integer".into()),
            Interpretation::singleton("c".into(), "Integer".into()),
            Interpretation::infix_operator("^".into(), 1, "^".into()),
            Interpretation::singleton("p".into(), "Proposition".into()),
            Interpretation::singleton("q".into(), "Proposition".into()),
            Interpretation::singleton("r".into(), "Proposition".into()),
            Interpretation::singleton("s".into(), "Proposition".into()),
            Interpretation::singleton("s".into(), "Proposition".into()),
        ];
        let mut workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        workspace
            .add_transformation(
                ExplicitTransformation::symmetry(
                    "+".to_string(),
                    "+".into(),
                    ("a".to_string(), "b".to_string()),
                    "Real".into(),
                )
                .into(),
            )
            .unwrap();

        workspace.add_parsed_hypothesis("x+y").unwrap();
        let expected = workspace.parse_from_string("y+x").unwrap();
        assert_eq!(
            workspace.try_transform_into_parsed("y+x").unwrap(),
            expected
        );
        assert!(workspace.get_statements().contains(&expected));

        workspace.add_parsed_hypothesis("j+k").unwrap();
        let expected = workspace.parse_from_string("k+j").unwrap();
        assert_eq!(
            workspace.try_transform_into_parsed("k+j").unwrap(),
            expected
        );
        assert!(workspace.get_statements().contains(&expected));

        workspace.add_parsed_hypothesis("a+(b+c)").unwrap();
        let expected = workspace.parse_from_string("(b+c)+a").unwrap();
        assert_eq!(
            workspace.try_transform_into_parsed("(b+c)+a").unwrap(),
            expected
        );
        assert!(workspace.get_statements().contains(&expected));

        let mut workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        workspace
            .add_transformation(
                ExplicitTransformation::symmetry(
                    "+".to_string(),
                    "+".into(),
                    ("a".to_string(), "b".to_string()),
                    "Real".into(),
                )
                .into(),
            )
            .unwrap();

        workspace.add_parsed_hypothesis("a+(b+c)").unwrap();
        let expected = workspace.parse_from_string("a+(c+b)").unwrap();
        assert_eq!(
            workspace.try_transform_into_parsed("a+(c+b)").unwrap(),
            expected
        );
        assert!(workspace.get_statements().contains(&expected));

        let mut workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        workspace
            .add_parsed_joint_transformation("p", "q", "p^q")
            .unwrap();

        workspace.add_parsed_hypothesis("r").unwrap();
        workspace.add_parsed_hypothesis("s").unwrap();

        let expected = workspace.parse_from_string("r^s").unwrap();
        assert_eq!(
            workspace.try_transform_into_parsed("r^s").unwrap(),
            expected
        );
        assert!(workspace.get_statements().contains(&expected));

        let expected = workspace.parse_from_string("s^r").unwrap();
        assert_eq!(
            workspace.try_transform_into_parsed("s^r").unwrap(),
            expected
        );
        assert!(workspace.get_statements().contains(&expected));
    }

    #[test]
    fn test_workspace_gets_valid_transformations() {
        let mut types = TypeHierarchy::chain(vec!["Real".into(), "Integer".into()]).unwrap();
        types
            .add_child_to_parent("=".into(), "Real".into())
            .unwrap();
        types
            .add_child_to_parent("+".into(), "Real".into())
            .unwrap();
        types
            .add_chain(vec!["Proposition".into(), "^".into()])
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("+".into(), 6, "+".into()),
            Interpretation::singleton("x".into(), "Real".into()),
            Interpretation::singleton("y".into(), "Real".into()),
            Interpretation::singleton("j".into(), "Integer".into()),
            Interpretation::singleton("k".into(), "Integer".into()),
            Interpretation::singleton("a".into(), "Integer".into()),
            Interpretation::singleton("b".into(), "Integer".into()),
            Interpretation::singleton("c".into(), "Integer".into()),
            Interpretation::infix_operator("^".into(), 1, "^".into()),
            Interpretation::singleton("p".into(), "Proposition".into()),
            Interpretation::singleton("q".into(), "Proposition".into()),
            Interpretation::singleton("r".into(), "Proposition".into()),
            Interpretation::singleton("s".into(), "Proposition".into()),
        ];
        let mut workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        workspace
            .add_transformation(
                ExplicitTransformation::symmetry(
                    "+".to_string(),
                    "+".into(),
                    ("a".to_string(), "b".to_string()),
                    "Real".into(),
                )
                .into(),
            )
            .unwrap();

        workspace.add_parsed_hypothesis("x+y").unwrap();
        let expected = vec![workspace.parse_from_string("y+x").unwrap()];
        assert_eq!(workspace.get_valid_transformations("y+x"), expected);

        workspace.add_parsed_hypothesis("j+k").unwrap();
        assert_eq!(workspace.get_statements().len(), 2);
        let expected = vec![workspace.parse_from_string("k+j").unwrap()];
        assert_eq!(workspace.get_valid_transformations("k+j"), expected);

        workspace.add_parsed_hypothesis("a+(b+c)").unwrap();
        let expected = workspace.parse_from_string("(b+c)+a").unwrap();
        assert_eq!(
            workspace.get_valid_transformations("(b+c)+a"),
            vec![expected]
        );

        let mut workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        workspace
            .add_transformation(
                ExplicitTransformation::symmetry(
                    "+".to_string(),
                    "+".into(),
                    ("a".to_string(), "b".to_string()),
                    "Real".into(),
                )
                .into(),
            )
            .unwrap();

        workspace.add_parsed_hypothesis("a+(b+c)").unwrap();
        let expected = workspace.parse_from_string("a+(c+b)").unwrap();
        assert_eq!(
            workspace.get_valid_transformations("a+(c+b)"),
            vec![expected]
        );

        let mut workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        workspace
            .add_parsed_joint_transformation("p", "q", "p^q")
            .unwrap();

        workspace.add_parsed_hypothesis("r").unwrap();
        workspace.add_parsed_hypothesis("s").unwrap();

        let expected = workspace.parse_from_string("r^s").unwrap();
        assert_eq!(workspace.get_valid_transformations("r^s"), vec![expected]);

        let expected = workspace.parse_from_string("s^r").unwrap();
        assert_eq!(workspace.get_valid_transformations("s^r"), vec![expected]);
    }

    #[test]
    fn test_workspace_adds_hypotheses() {
        let types = TypeHierarchy::new();
        let mut workspace = Workspace::new(types, vec![], vec![]);
        let statement = SymbolNode::leaf_object("a");
        workspace.add_hypothesis(statement).unwrap();
        assert_eq!(workspace.statements.len(), 1);

        let statement = SymbolNode::leaf_object("b");
        workspace.add_hypothesis(statement).unwrap();

        assert_eq!(workspace.statements.len(), 2);
    }

    #[test]
    fn test_workspace_adds_statement_with_generated_type() {
        let plus = Interpretation::infix_operator("+".into(), 1, "Integer".into());
        let integer_interpretation =
            Interpretation::generated_type(GeneratedTypeCondition::IsInteger);
        let integer_generated_type = GeneratedType::new(
            GeneratedTypeCondition::IsInteger,
            vec!["Integer".into()].into_iter().collect(),
        );

        let mut types = TypeHierarchy::chain(vec!["Real".into(), "Integer".into()]).unwrap();
        types.add_chain(vec!["+".into()]).unwrap();
        let mut workspace = Workspace::new(
            types,
            vec![integer_generated_type],
            vec![plus, integer_interpretation],
        );

        assert!(workspace.add_parsed_hypothesis("2+2").is_ok());
        let mut expected =
            TypeHierarchy::chain(vec!["Real".into(), "Integer".into(), "2".into()]).unwrap();
        expected.add_chain(vec!["+".into()]).unwrap();
        assert_eq!(workspace.types, expected);

        let expected = SymbolNode::new_from_symbol(
            Symbol::new("+".to_string(), "Integer".into()),
            vec![SymbolNode::singleton("2"), SymbolNode::singleton("2")],
        );
        assert_eq!(workspace.statements, vec![expected]);
    }

    #[test]
    fn test_workspace_transforms_statement_and_maintains_provenance() {
        let types = TypeHierarchy::new();
        let mut workspace = Workspace::new(types, vec![], vec![]);
        let statement = SymbolNode::leaf_object("a");
        assert_eq!(workspace.add_hypothesis(statement), Ok(()));
        assert_eq!(workspace.statements.len(), 1);

        let transformation =
            ExplicitTransformation::new(SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b"));
        workspace.add_transformation(transformation.into()).unwrap();
        assert_eq!(workspace.transformations.len(), 1);

        let _transformed = workspace.try_transform_into_parsed("b").unwrap();
        assert_eq!(workspace.statements.len(), 2);
        assert_eq!(
            workspace.statements,
            vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")]
        );

        assert_eq!(workspace.get_provenance(0), Ok(Provenance::Hypothesis));
        // TODO Currently we don't derive the transformation addresses but this assertion should
        // fail
        assert_eq!(
            workspace.get_provenance(1),
            Ok(Provenance::Derived((0, 0, vec![])))
        );

        assert_eq!(
            workspace.get_provenance_lineage(0),
            Ok(vec![Provenance::Hypothesis])
        );
    }

    #[test]
    fn test_workspace_imports_context() {
        let types = TypeHierarchy::new();
        let mut workspace = Workspace::new(types, vec![], vec![]);

        let mut types =
            TypeHierarchy::chain(vec!["Real".into(), "Rational".into(), "Integer".into()]).unwrap();
        types
            .add_chain(vec!["Operator".into(), "=".into()])
            .unwrap();
        types
            .add_chain_to_parent(vec!["+".into()], "Operator".into())
            .unwrap();

        let equality_reflexivity = ExplicitTransformation::reflexivity(
            "=".to_string(),
            "=".into(),
            "x".to_string(),
            "Real".into(),
        );

        let equality_symmetry = ExplicitTransformation::symmetry(
            "=".to_string(),
            "=".into(),
            ("x".to_string(), "y".to_string()),
            "Real".into(),
        );

        let context = Context::new(
            types.clone(),
            vec![],
            vec![],
            vec![
                equality_reflexivity.clone().into(),
                equality_symmetry.clone().into(),
            ],
        )
        .unwrap();

        assert_eq!(workspace.try_import_context(context.clone()), Ok(()));
        assert_eq!(workspace.types, types);
        assert_eq!(workspace.transformations.len(), 2);

        let mut complex_types = TypeHierarchy::chain(vec![
            "Complex".into(),
            "Real".into(),
            "Rational".into(),
            "Integer".into(),
        ])
        .unwrap();

        complex_types
            .add_chain(vec!["Operator".into(), "=".into()])
            .unwrap();
        complex_types
            .add_chain_to_parent(vec!["+".into()], "Operator".into())
            .unwrap();
        let ambiguous_context = Context::new(
            complex_types,
            vec![],
            vec![],
            vec![equality_reflexivity.into(), equality_symmetry.into()],
        )
        .unwrap();

        assert_eq!(
            workspace.try_import_context(ambiguous_context),
            Err(WorkspaceError::AttemptedToImportAmbiguousTypes(
                vec![
                    "Real".into(),
                    "Rational".into(),
                    "Integer".into(),
                    "=".into(),
                    "+".into(),
                    "Operator".into()
                ]
                .into_iter()
                .collect()
            ))
        );

        assert_eq!(workspace.transformations.len(), 2);

        let inverted_types = TypeHierarchy::chain(vec!["Rational".into(), "Real".into()]).unwrap();
        let conflicting_context = Context::new(inverted_types, vec![], vec![], vec![]).unwrap();

        assert_eq!(
            workspace.try_import_context(conflicting_context),
            Err(WorkspaceError::AttemptedToImportAmbiguousTypes(
                vec!["Real".into(), "Rational".into(),]
                    .into_iter()
                    .collect()
            ))
        );
    }
}
