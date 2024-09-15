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
        algorithm::AlgorithmType,
        symbol_node::{Symbol, SymbolName, SymbolNode, SymbolNodeAddress, SymbolNodeError},
        symbol_type::{
            DisplayGeneratedType, DisplayTypeHierarchyNode, GeneratedType, GeneratedTypeCondition,
            Type, TypeError, TypeHierarchy, TypeName,
        },
        transformation::{
            AlgorithmTransformation, ExplicitTransformation, Transformation, TransformationError,
        },
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

    pub fn get_types_mut(&mut self) -> &mut TypeHierarchy {
        &mut self.types
    }

    pub fn get_type_from_name(&self, name: &str) -> Result<Type, WorkspaceError> {
        let types = self
            .get_types()
            .get_types()
            .into_iter()
            .filter(|t| t.to_string() == name)
            .collect::<HashSet<_>>();
        if types.len() == 0 {
            Err(WorkspaceError::NoSuchType(name.to_string()))
        } else if types.len() > 1 {
            Err(WorkspaceError::AmbiguousTypeName(name.to_string()))
        } else {
            Ok(types
                .iter()
                .next()
                .expect("We just checked that there's at least one.")
                .clone())
        }
    }

    pub fn get_generated_type_from_parent_name(
        &self,
        name: &str,
    ) -> Result<GeneratedType, WorkspaceError> {
        let matches = self
            .get_generated_types()
            .into_iter()
            .filter(|t| {
                t.get_parents()
                    .into_iter()
                    .map(|parent| parent.to_string())
                    .collect::<HashSet<_>>()
                    .contains(name)
            })
            .cloned()
            .collect::<Vec<_>>();
        if matches.len() == 0 {
            Err(WorkspaceError::NoSuchGeneratedTypeParent(name.to_string()))
        } else if matches.len() > 1 {
            Err(WorkspaceError::AmbiguousGeneratedTypeParent(
                name.to_string(),
            ))
        } else {
            Ok(matches
                .iter()
                .next()
                .expect("We just checked that there's at least one.")
                .clone())
        }
    }

    pub fn get_interpretations(&self) -> &Vec<Interpretation> {
        &self.interpretations
    }

    fn try_import_context(&mut self, context: Context) -> Result<(), WorkspaceError> {
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
        self.get_statement_pairs_including(None)
    }

    pub fn get_statement_pairs_including(&self, statement: Option<&SymbolNode>) -> Vec<SymbolNode> {
        // Returns pairs of statements for use in joint transforms
        // Returns both orders as well as duplicates since those are valid joint transform args
        let left = self.get_statements();
        let right = self.get_statements();
        let mut to_return = Vec::new();
        for i in 0..left.len() {
            for j in 0..right.len() {
                if statement.is_none()
                    || (Some(&left[i]) == statement)
                    || (Some(&right[i]) == statement)
                {
                    to_return.push(left[i].clone().join(right[j].clone()));
                }
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

    fn add_type_to_parent(&mut self, t: Type, parent: Type) -> Result<Type, WorkspaceError> {
        self.types
            .add_child_to_parent(t, parent)
            .map_err(|e| e.into())
    }

    fn add_hypothesis(&mut self, statement: SymbolNode) -> Result<(), WorkspaceError> {
        self.types
            .binds_statement_or_error(&statement)
            .map_err(|x| WorkspaceError::from(x))?;
        if statement.get_arbitrary_nodes().len() > 0 {
            return Err(WorkspaceError::ContainsArbitraryNode);
        }
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

    pub fn get_transformations_with_indices(&self) -> Vec<(Transformation, TransformationIndex)> {
        self.transformations
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, t)| (t, i))
            .collect()
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

    fn add_transformation(&mut self, transformation: Transformation) -> Result<(), WorkspaceError> {
        self.types.binds_transformation_or_error(&transformation)?;
        self.transformations.push(transformation);
        Ok(())
    }

    pub fn get_valid_transformations(
        &self,
        partial_statement: &str,
    ) -> Result<Vec<SymbolNode>, WorkspaceError> {
        // TODO Try to complete partial statements
        let desired = match self.parse_from_string(partial_statement) {
            Err(_) => {
                return Ok(vec![]);
            }
            Ok(s) => s,
        };
        let instantiated_transformations =
            self.get_instantiated_transformations_with_indices(Some(desired.clone()))?;
        for (transformation, _) in instantiated_transformations {
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
                    return Ok(vec![desired.clone()]);
                }
            }
        }
        return Ok(vec![]);
    }

    pub fn get_valid_transformations_from(
        &mut self,
        statement_index: StatementIndex,
    ) -> Result<Vec<SymbolNode>, WorkspaceError> {
        let from_statement = self.get_statement(statement_index)?;
        let instantiated_transformations =
            self.get_instantiated_transformations_with_indices(None)?;
        let mut to_return = HashSet::new();
        for (transformation, _) in instantiated_transformations {
            let statements = if transformation.is_joint_transform() {
                self.get_statement_pairs_including(Some(&from_statement))
            } else {
                vec![from_statement.clone()]
            };
            for statement in statements {
                let valid_transformations =
                    transformation.get_valid_transformations(self.get_types(), &statement);
                to_return.extend(valid_transformations);
            }
        }
        return Ok(to_return
            .into_iter()
            .filter(|s| !self.statements.contains(s))
            .collect());
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

    fn add_generated_type(&mut self, generated_type: GeneratedType) {
        self.generated_types.push(generated_type);
    }

    pub fn get_instantiated_transformations_with_indices(
        &self,
        maybe_desired: Option<SymbolNode>,
    ) -> Result<HashSet<(Transformation, TransformationIndex)>, WorkspaceError> {
        // TODO Desired should probably wind up at the top of what we test, but this vec
        // doesn't stay ordered.  If we pass it down the dependency chain, we could ensure that
        // it's checked first
        let statements = if let Some(desired) = maybe_desired {
            let mut statements = vec![desired];
            statements.append(&mut self.get_statements().clone());
            statements
        } else {
            self.get_statements().clone()
        };
        let substatements = statements
            .iter()
            .map(|statement| statement.get_substatements())
            .flatten()
            .collect::<HashSet<_>>();

        let mut to_return = HashSet::new();
        for (transform, transform_idx) in self.get_arbitrary_transformations_with_indices() {
            to_return = to_return
                .union(
                    &transform
                        .instantiate_arbitrary_nodes(self.get_types(), &substatements)
                        .map_err(|e| Into::<WorkspaceError>::into(e))?
                        .into_iter()
                        .map(|t| (t, transform_idx))
                        .collect(),
                )
                .cloned()
                .collect();
        }

        to_return = to_return
            .union(
                &self
                    .get_non_arbitrary_transformations_with_indices()
                    .into_iter()
                    .collect::<HashSet<_>>(),
            )
            .cloned()
            .collect();

        return Ok(to_return);
    }

    fn get_arbitrary_transformations_with_indices(
        &self,
    ) -> HashSet<(Transformation, TransformationIndex)> {
        self.get_transformations_with_indices()
            .iter()
            .filter(|(t, _)| t.contains_arbitrary_nodes())
            .cloned()
            .collect()
    }

    fn get_non_arbitrary_transformations_with_indices(
        &self,
    ) -> HashSet<(Transformation, TransformationIndex)> {
        self.get_transformations_with_indices()
            .iter()
            .filter(|(t, _)| !t.contains_arbitrary_nodes())
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceTransactionStore {
    transactions: Vec<WorkspaceTransaction>,
    next_index: usize,
}

impl Default for WorkspaceTransactionStore {
    fn default() -> Self {
        Self::empty()
    }
}

impl WorkspaceTransactionStore {
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn snapshot(workspace: Workspace) -> Self {
        Self::new(vec![WorkspaceTransactionItem::Snapshot(workspace).into()])
    }

    pub fn new(transactions: Vec<WorkspaceTransaction>) -> Self {
        let next_index = transactions.len();
        Self {
            transactions,
            next_index,
        }
    }

    pub fn serialize(&self) -> Result<String, WorkspaceError> {
        toml::to_string(self).map_err(|e| WorkspaceError::UnableToSerialize(e.to_string()))
    }

    pub fn deserialize(serialized: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(serialized)
    }

    pub fn len(&self) -> usize {
        self.transactions.len()
    }

    pub fn add(&mut self, transaction: WorkspaceTransaction) -> Result<(), WorkspaceError> {
        // Ensure that the transaction compiles before adding it
        Self::apply_transaction(&mut self.compile(), transaction.clone())?;

        self.clear_undos();
        self.transactions.push(transaction);
        self.next_index += 1;
        Ok(())
    }

    pub fn clear_undos(&mut self) {
        self.transactions.truncate(self.next_index)
    }

    pub fn get_live_transactions(&self) -> Vec<WorkspaceTransaction> {
        self.transactions[0..self.next_index].to_vec()
    }

    pub fn compile(&self) -> Workspace {
        let mut workspace = Workspace::default();
        for transaction in self.get_live_transactions() {
            // Does not check for errors because they've been checked on adding them
            let _ = Self::apply_transaction(&mut workspace, transaction);
        }

        workspace
    }

    fn apply_transaction(
        workspace: &mut Workspace,
        transaction: WorkspaceTransaction,
    ) -> Result<(), WorkspaceError> {
        for item in transaction.items {
            Self::apply_transaction_item(workspace, item)?;
        }
        Ok(())
    }

    fn apply_transaction_item(
        workspace: &mut Workspace,
        transaction: WorkspaceTransactionItem,
    ) -> Result<(), WorkspaceError> {
        match transaction {
            WorkspaceTransactionItem::Snapshot(snapshot) => {
                *workspace = snapshot.clone();
            }
            WorkspaceTransactionItem::AddType((t, parent)) => {
                workspace.add_type_to_parent(t, parent)?;
            }
            WorkspaceTransactionItem::AddGeneratedType(t) => {
                workspace.add_generated_type(t);
            }
            WorkspaceTransactionItem::AddInterpretation(interpretation) => {
                workspace.add_interpretation(interpretation)?;
            }
            WorkspaceTransactionItem::RemoveInterpretation(index) => {
                workspace.remove_interpretation(index)?;
            }
            WorkspaceTransactionItem::AddHypothesis(hypothesis) => {
                workspace.add_hypothesis(hypothesis)?;
            }
            WorkspaceTransactionItem::AddTransformation(transformation) => {
                workspace.add_transformation(transformation)?;
            }
            WorkspaceTransactionItem::Derive(statement, provenance) => {
                workspace.add_derived_statement(statement, provenance);
            }
        };
        Ok(())
    }

    pub fn truncate(&mut self, n_incremental_to_keep: usize) {
        let n_to_compile = self.len().saturating_sub(n_incremental_to_keep);
        if n_to_compile == 0 {
            return;
        }

        self.compile_first_n_to_snapshot(n_to_compile);
    }

    fn compile_to_snapshot(&mut self) {
        // Forgets anything beyond the pointer
        let snapshot = Self::new(self.transactions[0..self.next_index].to_vec()).compile();
        self.transactions = vec![WorkspaceTransactionItem::Snapshot(snapshot).into()];
        self.next_index = self.len();
    }

    fn compile_first_n_to_snapshot(&mut self, n: usize) {
        let to_snapshot = self.transactions[0..n].to_vec();
        let mut remaining = self.transactions[n..].to_vec();

        let next_index_pointed_at_compiled = self.next_index < n;
        if next_index_pointed_at_compiled {
            // If the next index is somewhere within the compilation
            // we need to just compile it to a snapshot and forget the rest
            self.compile_to_snapshot();
            return;
        }

        let snapshot = Self::new(to_snapshot).compile();
        self.transactions = vec![WorkspaceTransactionItem::Snapshot(snapshot).into()];
        self.transactions.append(&mut remaining);

        self.next_index -= n - 1;
    }

    pub fn undo(&mut self) -> Option<WorkspaceTransaction> {
        if self.next_index > 0 {
            self.next_index -= 1;
            Some(self.transactions[self.next_index].clone())
        } else {
            None
        }
    }

    pub fn redo(&mut self) -> Option<WorkspaceTransaction> {
        if self.next_index < self.transactions.len() {
            self.next_index += 1;
            Some(self.transactions[self.next_index - 1].clone())
        } else {
            None
        }
    }

    pub fn add_type_to_parent(&mut self, t: Type, parent: Type) -> Result<(), WorkspaceError> {
        self.add(WorkspaceTransactionItem::AddType((t, parent)).into())
    }

    pub fn import_context(&mut self, context: Context) -> Result<(), WorkspaceError> {
        let mut workspace = self.compile();
        workspace.try_import_context(context)?;
        *self = Self::snapshot(workspace);
        Ok(())
    }

    pub fn add_parsed_hypothesis(&mut self, s: &str) -> Result<SymbolNode, WorkspaceError> {
        let parsed = self.compile().parse_from_string(s)?;
        let mut transaction = self.get_generated_types_transaction(&parsed)?;
        println!("Generated types transaction: {:?}", transaction);
        transaction.add(WorkspaceTransactionItem::AddHypothesis(parsed.clone()));
        self.add(transaction)?;
        Ok(parsed)
    }

    pub fn add_parsed_joint_transformation(
        &mut self,
        left_from: &str,
        right_from: &str,
        to: &str,
    ) -> Result<Transformation, WorkspaceError> {
        let workspace = self.compile();
        let parsed_from = (workspace.parse_from_string(left_from)?)
            .join(workspace.parse_from_string(right_from)?);
        let from_transaction = self.get_generated_types_transaction(&parsed_from)?;
        let parsed_to = workspace.parse_from_string(to)?;
        let to_transaction = self.get_generated_types_transaction(&parsed_to)?;
        let mut transaction = from_transaction.combine(to_transaction);
        // TODO Do we need to handle the case where from and to transactions try to add the same
        // type?
        let transformation: Transformation =
            ExplicitTransformation::new(parsed_from, parsed_to).into();
        transaction.add(WorkspaceTransactionItem::AddTransformation(
            transformation.clone(),
        ));
        self.add(transaction)?;
        Ok(transformation)
    }

    pub fn add_parsed_transformation(
        &mut self,
        is_equivalence: bool,
        from: &str,
        to: &str,
    ) -> Result<Transformation, WorkspaceError> {
        let workspace = self.compile();
        let parsed_from = workspace.parse_from_string(from)?;
        let from_transaction = self.get_generated_types_transaction(&parsed_from)?;
        let parsed_to = workspace.parse_from_string(to)?;
        let to_transaction = self.get_generated_types_transaction(&parsed_to)?;
        let mut transaction = from_transaction.combine(to_transaction);
        // TODO Similar to above, do we need to get rid of duplicative AddType transactions between
        // from and to?
        let transformation: Transformation =
            ExplicitTransformation::new(parsed_from.clone(), parsed_to.clone()).into();
        transaction.add(WorkspaceTransactionItem::AddTransformation(
            transformation.clone(),
        ));
        if is_equivalence {
            let transformation: Transformation =
                ExplicitTransformation::new(parsed_to, parsed_from).into();
            transaction.add(WorkspaceTransactionItem::AddTransformation(
                transformation.clone(),
            ));
        }
        self.add(transaction)?;
        Ok(transformation)
    }

    pub fn add_algorithm(
        &mut self,
        algorithm_type: &AlgorithmType,
        operator_name: &str,
        input_type_name: &str,
    ) -> Result<Transformation, WorkspaceError> {
        let workspace = self.compile();
        let operator_type = workspace.get_type_from_name(operator_name)?;
        let operator = Symbol::new(operator_name.to_string(), operator_type);
        let input_type = workspace.get_generated_type_from_parent_name(input_type_name)?;
        let transformation: Transformation =
            AlgorithmTransformation::new(algorithm_type.clone(), operator, input_type).into();
        let transaction =
            WorkspaceTransactionItem::AddTransformation(transformation.clone()).into();
        self.add(transaction)?;
        Ok(transformation)
    }

    pub fn try_transform_into_parsed(
        &mut self,
        desired: &str,
    ) -> Result<SymbolNode, WorkspaceError> {
        let parsed = self.compile().parse_from_string(desired)?;
        self.try_transform_into(parsed)
    }

    pub fn try_transform_into(
        &mut self,
        desired: SymbolNode,
    ) -> Result<SymbolNode, WorkspaceError> {
        let mut transaction = self.get_generated_types_transaction(&desired)?;
        let workspace = self.compile();
        if workspace.statements.contains(&desired) {
            return Err(WorkspaceError::StatementsAlreadyInclude(desired.clone()));
        }
        if desired.get_arbitrary_nodes().len() > 0 {
            return Err(WorkspaceError::ContainsArbitraryNode);
        }
        let instantiated_transformations =
            workspace.get_instantiated_transformations_with_indices(Some(desired.clone()))?;
        for (transform, transform_idx) in instantiated_transformations {
            let statements = if transform.is_joint_transform() {
                workspace.get_statement_pairs()
            } else {
                workspace.get_statements().clone()
            };
            for (statement_idx, statement) in statements.iter().enumerate() {
                match transform.try_transform_into(workspace.get_types(), &statement, &desired) {
                    Ok(output) => {
                        // TODO Derive the appropriate transform addresses
                        let provenance =
                            Provenance::Derived((statement_idx, transform_idx, vec![]));

                        transaction
                            .add(WorkspaceTransactionItem::Derive(output.clone(), provenance));
                        self.add(transaction)?;
                        return Ok(output);
                    }
                    Err(_) => {
                        // Do nothing, keep trying transformations
                    }
                }
            }
        }
        return Err(WorkspaceError::NoTransformationsPossible);
    }

    fn get_generated_types_transaction(
        &mut self,
        statement: &SymbolNode,
    ) -> Result<WorkspaceTransaction, WorkspaceError> {
        let workspace = self.compile();
        let mut items = Vec::new();
        let mut already_added = HashSet::new();
        for generated_type in workspace.get_generated_types().clone() {
            println!("generated_type: {:?}", generated_type);
            for (t, parents) in generated_type.generate(statement) {
                if !already_added.contains(&t) && !workspace.get_types().contains_type(&t) {
                    println!("Adding {:?} to {:?}", t, parents);
                    items.extend(self.get_add_type_to_parents_items(t.clone(), &parents));
                    already_added.insert(t);
                }
            }
        }
        Ok(WorkspaceTransaction::new(items))
    }

    fn get_generated_types_in_bulk_transaction(
        &mut self,
        statements: HashSet<SymbolNode>,
    ) -> Result<WorkspaceTransaction, WorkspaceError> {
        let mut items = Vec::new();
        for statement in statements {
            items.extend(self.get_generated_types_transaction(&statement)?.items);
        }
        Ok(WorkspaceTransaction::new(items))
    }

    fn get_add_type_to_parents_items(
        &mut self,
        t: Type,
        parents: &HashSet<Type>,
    ) -> Vec<WorkspaceTransactionItem> {
        let items = parents
            .iter()
            .map(|parent| WorkspaceTransactionItem::AddType((t.clone(), parent.clone())))
            .collect();
        items
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceTransaction {
    pub items: Vec<WorkspaceTransactionItem>,
}

impl From<WorkspaceTransactionItem> for WorkspaceTransaction {
    fn from(value: WorkspaceTransactionItem) -> Self {
        Self::new(vec![value])
    }
}

impl WorkspaceTransaction {
    pub fn new(items: Vec<WorkspaceTransactionItem>) -> Self {
        Self { items }
    }

    pub fn add(&mut self, item: WorkspaceTransactionItem) {
        self.items.push(item);
    }

    pub fn combine(self, other: Self) -> Self {
        let mut new_items = self.items;
        new_items.extend(other.items);
        Self::new(new_items)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkspaceTransactionItem {
    Snapshot(Workspace),
    AddType((Type, Type)), // New Type, Parent
    AddGeneratedType(GeneratedType),
    AddInterpretation(Interpretation),
    RemoveInterpretation(usize),
    AddHypothesis(SymbolNode),
    AddTransformation(Transformation),
    Derive(SymbolNode, Provenance),
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
    ContainsArbitraryNode,
    ParserError(ParserError),
    UnableToSerialize(String),
    TransformationError(TransformationError),
    StatementContainsTypesNotInHierarchy(HashSet<Type>),
    IncompatibleTypeRelationships(HashSet<Type>),
    ConflictingTypes(String, Type, Type),
    StatementsAlreadyInclude(SymbolNode),
    TypeHierarchyAlreadyIncludes(Type),
    InvalidType(Type),
    NoSuchType(String),
    NoSuchGeneratedTypeParent(String),
    AmbiguousTypeName(String),
    AmbiguousGeneratedTypeParent(String),
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
    fn test_truncation_store_snapshots_all_if_truncation_is_too_much() {
        let mut store = WorkspaceTransactionStore::empty();
        assert_eq!(store.compile(), Workspace::default());

        store.add(WorkspaceTransactionItem::AddType(("Real".into(), Type::Object)).into());
        store.add(WorkspaceTransactionItem::AddType(("Integer".into(), "Real".into())).into());
        store.add(
            WorkspaceTransactionItem::AddInterpretation(Interpretation::infix_operator(
                "+".into(),
                1,
                "Real".into(),
            ))
            .into(),
        );
        store.add(
            WorkspaceTransactionItem::AddInterpretation(Interpretation::infix_operator(
                "-".into(),
                1,
                "Real".into(),
            ))
            .into(),
        );
        assert_eq!(store.len(), 4);
        store.add(WorkspaceTransactionItem::RemoveInterpretation(1).into());
        assert_eq!(store.len(), 5);

        store.undo();
        store.undo();
        store.undo();
        store.undo();
        assert_eq!(store.next_index, 1);

        let cached_workspace = store.compile();

        store.truncate(3);
        assert_eq!(store.len(), 1);
        assert_eq!(store.next_index, 1);

        assert_eq!(store.compile(), cached_workspace);
    }

    #[test]
    fn test_transaction_store_truncates() {
        let mut store = WorkspaceTransactionStore::empty();
        assert_eq!(store.compile(), Workspace::default());

        store.add(WorkspaceTransactionItem::AddType(("Real".into(), Type::Object)).into());
        store.add(WorkspaceTransactionItem::AddType(("Integer".into(), "Real".into())).into());
        store.add(
            WorkspaceTransactionItem::AddInterpretation(Interpretation::infix_operator(
                "+".into(),
                1,
                "Real".into(),
            ))
            .into(),
        );
        let cached_workspace = store.compile();
        store.add(
            WorkspaceTransactionItem::AddInterpretation(Interpretation::infix_operator(
                "-".into(),
                1,
                "Real".into(),
            ))
            .into(),
        );
        assert_eq!(store.len(), 4);
        store.add(WorkspaceTransactionItem::RemoveInterpretation(1).into());
        assert_eq!(store.len(), 5);
        assert_eq!(store.compile(), cached_workspace);

        let cached_workspace = store.compile();
        store.truncate(10);
        assert_eq!(store.compile(), cached_workspace);
        assert_eq!(store.len(), 5);

        store.truncate(4);
        assert_eq!(store.len(), 5); // Snapshot + 4
        assert_eq!(store.next_index, 5);
        assert_eq!(store.compile(), cached_workspace);

        store.truncate(3);
        assert_eq!(store.len(), 4); // Snapshot + 3
        assert_eq!(store.next_index, 4);
        assert_eq!(store.compile(), cached_workspace);

        store.truncate(0);
        assert_eq!(store.len(), 1); // Just the snapshot
        assert_eq!(store.next_index, 1);
        assert_eq!(store.compile(), cached_workspace);

        let cached_store = store.clone();
        store.truncate(10);
        assert_eq!(store, cached_store);
    }

    #[test]
    fn test_transaction_store_compiles() {
        let mut store = WorkspaceTransactionStore::empty();
        assert_eq!(store.compile(), Workspace::default());

        store.add(WorkspaceTransactionItem::AddType(("Real".into(), Type::Object)).into());

        let expected_0 = Workspace::new(
            TypeHierarchy::chain(vec!["Real".into()]).unwrap(),
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(store.compile(), expected_0);

        store.add(WorkspaceTransactionItem::AddType(("Integer".into(), "Real".into())).into());

        let expected_1 = Workspace::new(
            TypeHierarchy::chain(vec!["Real".into(), "Integer".into()]).unwrap(),
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(store.compile(), expected_1);

        store.undo();
        assert_eq!(store.compile(), expected_0);

        store.undo();
        assert_eq!(store.compile(), Workspace::default());

        store.redo();
        assert_eq!(store.compile(), expected_0);

        store.redo();
        assert_eq!(store.compile(), expected_1);
    }

    #[test]
    fn test_workspace_try_transform_into_with_arbitrary() {
        let mut types = TypeHierarchy::chain(vec!["Boolean".into(), "=".into()]).unwrap();
        types
            .add_child_to_parent("^".into(), "Boolean".into())
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("^".into(), 2, "^".into()),
            Interpretation::singleton("p".into(), "Boolean".into()),
            Interpretation::singleton("q".into(), "Boolean".into()),
            Interpretation::singleton("r".into(), "Boolean".into()),
            Interpretation::singleton("s".into(), "Boolean".into()),
            Interpretation::arbitrary_functional("Any".into(), 99, "Boolean".into()),
        ];

        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);

        workspace_store
            .add_parsed_transformation(false, "p=q", "Any(p)=Any(q)")
            .unwrap();

        workspace_store.add_parsed_hypothesis("p=q").unwrap();

        let expected = workspace_store
            .compile()
            .parse_from_string("p^s=q^s")
            .unwrap();
        assert_eq!(
            workspace_store
                .try_transform_into_parsed("p^s=q^s")
                .unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));
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

        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);

        workspace_store
            .add_parsed_transformation(false, "p=q", "Any(p)=Any(q)")
            .unwrap();
        workspace_store.add_parsed_hypothesis("p=q").unwrap();
        workspace_store.add_parsed_hypothesis("p^s").unwrap();

        let instantiate = |s: &str| {
            (
                ExplicitTransformation::new(
                    workspace_store.compile().parse_from_string("p=q").unwrap(),
                    workspace_store.compile().parse_from_string(s).unwrap(),
                )
                .into(),
                0,
            )
        };

        let expected = vec![
            instantiate("p=q"),
            instantiate("(p=q)=(q=q)"),
            instantiate("(p=p)=(p=q)"),
            instantiate("(p^s)=(q^s)"),
            instantiate("(p^p)=(p^q)"),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            workspace_store
                .compile()
                .get_instantiated_transformations_with_indices(None)
                .unwrap(),
            expected
        );

        let instantiate = |from: &str, to: &str| {
            (
                ExplicitTransformation::new(
                    workspace_store.compile().parse_from_string(from).unwrap(),
                    workspace_store.compile().parse_from_string(to).unwrap(),
                )
                .into(),
                0,
            )
        };
        let mut workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);

        workspace_store
            .add_parsed_transformation(false, "Any(p)", "p=p")
            .unwrap();
        workspace_store.add_parsed_hypothesis("p=q").unwrap();
        workspace_store.add_parsed_hypothesis("p^s").unwrap();

        let expected: HashSet<_> = vec![
            instantiate("p", "p=p"),
            instantiate("p=q", "p=p"),
            instantiate("p=p", "p=p"),
            instantiate("p^s", "p=p"),
            instantiate("p^p", "p=p"),
        ]
        .into_iter()
        .collect();

        let actual = workspace_store
            .compile()
            .get_instantiated_transformations_with_indices(None)
            .unwrap();
        assert_eq!(
            actual,
            expected,
            "Actual:\n{}\n\nExpected:\n{}",
            actual
                .iter()
                .map(|(n, _)| n.to_interpreted_string(&interpretations))
                .collect::<Vec<_>>()
                .join("\n"),
            expected
                .iter()
                .map(|(n, _)| n.to_interpreted_string(&interpretations))
                .collect::<Vec<_>>()
                .join("\n"),
        );
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

        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        assert_eq!(
            workspace_store.add_parsed_hypothesis("Any(p)"),
            Err(WorkspaceError::ContainsArbitraryNode)
        );
        assert_eq!(
            workspace_store.add_parsed_hypothesis("Any(p)=q"),
            Err(WorkspaceError::ContainsArbitraryNode)
        );

        workspace_store.add_parsed_hypothesis("p=q").unwrap();
        assert_eq!(
            workspace_store.try_transform_into_parsed("Any(p)=Any(q)"),
            Err(WorkspaceError::ContainsArbitraryNode)
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
            Interpretation::singleton("j".into(), "Real".into()),
            Interpretation::singleton("k".into(), "Real".into()),
            Interpretation::singleton("a".into(), "Real".into()),
            Interpretation::singleton("b".into(), "Real".into()),
            Interpretation::singleton("c".into(), "Real".into()),
            Interpretation::infix_operator("^".into(), 1, "^".into()),
            Interpretation::singleton("p".into(), "Proposition".into()),
            Interpretation::singleton("q".into(), "Proposition".into()),
            Interpretation::singleton("r".into(), "Proposition".into()),
            Interpretation::singleton("s".into(), "Proposition".into()),
            Interpretation::singleton("s".into(), "Proposition".into()),
        ];
        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);

        workspace_store
            .add_parsed_transformation(false, "a+b", "b+a")
            .unwrap();

        workspace_store.add_parsed_hypothesis("x+y").unwrap();
        let expected = workspace_store.compile().parse_from_string("y+x").unwrap();
        assert_eq!(
            workspace_store.try_transform_into_parsed("y+x").unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));

        workspace_store.add_parsed_hypothesis("j+k").unwrap();
        let expected = workspace_store.compile().parse_from_string("k+j").unwrap();
        assert_eq!(
            workspace_store.try_transform_into_parsed("k+j").unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));

        workspace_store.add_parsed_hypothesis("a+(b+c)").unwrap();
        let expected = workspace_store
            .compile()
            .parse_from_string("(b+c)+a")
            .unwrap();
        assert_eq!(
            workspace_store
                .try_transform_into_parsed("(b+c)+a")
                .unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));

        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        workspace_store
            .add_parsed_transformation(false, "a+b", "b+a")
            .unwrap();

        workspace_store.add_parsed_hypothesis("a+(b+c)").unwrap();
        let expected = workspace_store
            .compile()
            .parse_from_string("a+(c+b)")
            .unwrap();
        assert_eq!(
            workspace_store
                .try_transform_into_parsed("a+(c+b)")
                .unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));

        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        workspace_store
            .add_parsed_joint_transformation("p", "q", "p^q")
            .unwrap();

        workspace_store.add_parsed_hypothesis("r").unwrap();
        workspace_store.add_parsed_hypothesis("s").unwrap();

        let expected = workspace_store.compile().parse_from_string("r^s").unwrap();
        assert_eq!(
            workspace_store.try_transform_into_parsed("r^s").unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));

        let expected = workspace_store.compile().parse_from_string("s^r").unwrap();
        assert_eq!(
            workspace_store.try_transform_into_parsed("s^r").unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));
    }

    #[test]
    fn test_workspace_gets_valid_transformations_from() {
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
        types
            .add_child_to_parent("|".into(), "Proposition".into())
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("+".into(), 6, "+".into()),
            Interpretation::singleton("x".into(), "Real".into()),
            Interpretation::singleton("y".into(), "Real".into()),
            Interpretation::singleton("j".into(), "Real".into()),
            Interpretation::singleton("k".into(), "Real".into()),
            Interpretation::singleton("a".into(), "Real".into()),
            Interpretation::singleton("b".into(), "Real".into()),
            Interpretation::singleton("c".into(), "Real".into()),
            Interpretation::infix_operator("^".into(), 7, "^".into()),
            Interpretation::infix_operator("|".into(), 7, "^".into()),
            Interpretation::singleton("p".into(), "Proposition".into()),
            Interpretation::singleton("q".into(), "Proposition".into()),
            Interpretation::singleton("r".into(), "Proposition".into()),
            Interpretation::singleton("s".into(), "Proposition".into()),
            Interpretation::arbitrary_functional("Any".into(), 98, "Proposition".into()),
        ];
        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        workspace_store
            .add_parsed_transformation(false, "a+b", "b+a")
            .unwrap();

        workspace_store.add_parsed_hypothesis("x+y").unwrap();
        let expected = vec![workspace_store.compile().parse_from_string("y+x").unwrap()];
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations_from(0)
                .unwrap(),
            expected
        );
        workspace_store.try_transform_into_parsed("y+x").unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations_from(0)
                .unwrap(),
            vec![],
        );

        workspace_store.add_parsed_hypothesis("j+k").unwrap();
        assert_eq!(workspace_store.compile().get_statements().len(), 3);
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations_from(0)
                .unwrap(),
            vec![],
        );

        let expected = vec![workspace_store.compile().parse_from_string("k+j").unwrap()];
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations_from(2)
                .unwrap(),
            expected
        );

        workspace_store.add_parsed_hypothesis("a+(b+c)").unwrap();
        let expected = vec![
            workspace_store
                .compile()
                .parse_from_string("(b+c)+a")
                .unwrap(),
            workspace_store
                .compile()
                .parse_from_string("(c+b)+a")
                .unwrap(),
            workspace_store
                .compile()
                .parse_from_string("a+(c+b)")
                .unwrap(),
        ]
        .into_iter()
        .collect::<HashSet<_>>();
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations_from(3)
                .unwrap()
                .into_iter()
                .collect::<HashSet<_>>(),
            expected
        );
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
        types
            .add_child_to_parent("|".into(), "Proposition".into())
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("+".into(), 6, "+".into()),
            Interpretation::singleton("x".into(), "Real".into()),
            Interpretation::singleton("y".into(), "Real".into()),
            Interpretation::singleton("j".into(), "Real".into()),
            Interpretation::singleton("k".into(), "Real".into()),
            Interpretation::singleton("a".into(), "Real".into()),
            Interpretation::singleton("b".into(), "Real".into()),
            Interpretation::singleton("c".into(), "Real".into()),
            Interpretation::infix_operator("^".into(), 7, "^".into()),
            Interpretation::infix_operator("|".into(), 7, "^".into()),
            Interpretation::singleton("p".into(), "Proposition".into()),
            Interpretation::singleton("q".into(), "Proposition".into()),
            Interpretation::singleton("r".into(), "Proposition".into()),
            Interpretation::singleton("s".into(), "Proposition".into()),
            Interpretation::arbitrary_functional("Any".into(), 98, "Proposition".into()),
        ];
        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace.clone());
        workspace_store
            .add_parsed_transformation(false, "a+b", "b+a")
            .unwrap();

        workspace_store.add_parsed_hypothesis("x+y").unwrap();
        let expected = vec![workspace_store.compile().parse_from_string("y+x").unwrap()];
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations("y+x")
                .unwrap(),
            expected,
            "workspace_store:\n{:?}\nworkspace:\n{:?}",
            workspace_store,
            workspace_store.compile(),
        );

        workspace_store.add_parsed_hypothesis("j+k").unwrap();
        assert_eq!(workspace_store.compile().get_statements().len(), 2);
        let expected = vec![workspace_store.compile().parse_from_string("k+j").unwrap()];
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations("k+j")
                .unwrap(),
            expected
        );

        workspace_store.add_parsed_hypothesis("a+(b+c)").unwrap();
        let expected = workspace_store
            .compile()
            .parse_from_string("(b+c)+a")
            .unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations("(b+c)+a")
                .unwrap(),
            vec![expected]
        );

        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        workspace_store
            .add_parsed_transformation(false, "a+b", "b+a")
            .unwrap();

        workspace_store.add_parsed_hypothesis("a+(b+c)").unwrap();
        let expected = workspace_store
            .compile()
            .parse_from_string("a+(c+b)")
            .unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations("a+(c+b)")
                .unwrap(),
            vec![expected]
        );

        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        workspace_store
            .add_parsed_joint_transformation("p", "q", "p^q")
            .unwrap();

        workspace_store.add_parsed_hypothesis("r").unwrap();
        workspace_store.add_parsed_hypothesis("s").unwrap();

        let expected = workspace_store.compile().parse_from_string("r^s").unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations("r^s")
                .unwrap(),
            vec![expected]
        );

        let expected = workspace_store.compile().parse_from_string("s^r").unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations("s^r")
                .unwrap(),
            vec![expected]
        );

        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        workspace_store.add_parsed_hypothesis("p=q").unwrap();
        workspace_store
            .add_parsed_transformation(false, "p=q", "Any(p)=Any(q)")
            .unwrap();

        let expected = workspace_store
            .compile()
            .parse_from_string("p^q=q^q")
            .unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations("p^q=q^q")
                .unwrap(),
            vec![expected]
        );

        let workspace = Workspace::new(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        workspace_store.add_parsed_hypothesis("r=s").unwrap();
        workspace_store
            .add_parsed_transformation(false, "p=q", "Any(p)=Any(q)")
            .unwrap();

        let expected = workspace_store
            .compile()
            .parse_from_string("r^s=s^s")
            .unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations("r^s=s^s")
                .unwrap(),
            vec![expected]
        );
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
        let workspace = Workspace::new(
            types,
            vec![integer_generated_type],
            vec![plus, integer_interpretation],
        );
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);

        assert!(workspace_store.add_parsed_hypothesis("2+2").is_ok());
        let mut expected =
            TypeHierarchy::chain(vec!["Real".into(), "Integer".into(), "2".into()]).unwrap();
        expected.add_chain(vec!["+".into()]).unwrap();
        assert_eq!(workspace_store.compile().types, expected);

        let expected = SymbolNode::new_from_symbol(
            Symbol::new("+".to_string(), "Integer".into()),
            vec![SymbolNode::singleton("2"), SymbolNode::singleton("2")],
        );
        assert_eq!(workspace_store.compile().statements, vec![expected]);
    }

    #[test]
    fn test_workspace_adds_and_executes_algorithms() {
        let mut types = TypeHierarchy::chain(vec!["Real".into(), "+".into()]).unwrap();
        types
            .add_child_to_parent("/".into(), "Real".into())
            .unwrap();
        let plus_interpretation = Interpretation::infix_operator("+".into(), 1, "+".into());
        let divides_interpretation = Interpretation::infix_operator("/".into(), 2, "/".into());
        let real_interpretation = Interpretation::generated_type(GeneratedTypeCondition::IsNumeric);
        let real_generated_type = GeneratedType::new(
            GeneratedTypeCondition::IsNumeric,
            vec!["Real".into()].into_iter().collect(),
        );
        let generated_types = vec![real_generated_type];
        let interpretations = vec![
            plus_interpretation,
            divides_interpretation,
            real_interpretation,
        ];
        let workspace = Workspace::new(types, generated_types, interpretations);

        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);

        assert!(workspace_store.add_parsed_hypothesis("2+2").is_ok());
        assert_eq!(workspace_store.compile().statements.len(), 1);
        let two = SymbolNode::leaf(Symbol::new("2".to_string(), "2".into()));
        assert_eq!(
            workspace_store.compile().statements,
            vec![SymbolNode::new(
                Symbol::new_with_same_type_as_value("+").into(),
                vec![two.clone(), two.clone()]
            )]
        );

        workspace_store
            .add_algorithm(&AlgorithmType::Addition, "+", "Real")
            .unwrap();
        assert_eq!(workspace_store.compile().transformations.len(), 1);

        let _transformed = workspace_store.try_transform_into_parsed("4").unwrap();
        assert_eq!(workspace_store.compile().statements.len(), 2);
        let four = SymbolNode::leaf(Symbol::new("4".to_string(), "4".into()));
        assert_eq!(workspace_store.compile().statements[1], four);

        let hypothesis_result = workspace_store.add_parsed_hypothesis("8/4");
        assert!(hypothesis_result.is_ok(), "{:?}", hypothesis_result);
        assert_eq!(workspace_store.compile().statements.len(), 3);
        workspace_store
            .add_algorithm(&AlgorithmType::Division, "/", "Real")
            .unwrap();
        assert_eq!(workspace_store.compile().transformations.len(), 2);

        let _transformed = workspace_store.try_transform_into_parsed("2").unwrap();
        assert_eq!(workspace_store.compile().statements.len(), 4);
        assert_eq!(workspace_store.compile().statements[3], two);
    }

    #[test]
    fn test_workspace_transforms_statement_and_maintains_provenance() {
        let types = TypeHierarchy::new();
        let workspace = Workspace::new(types, vec![], vec![]);
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        assert!(workspace_store.add_parsed_hypothesis("a").is_ok());
        assert_eq!(workspace_store.compile().statements.len(), 1);

        workspace_store
            .add_parsed_transformation(false, "a", "b")
            .unwrap();
        assert_eq!(workspace_store.compile().transformations.len(), 1);

        let _transformed = workspace_store.try_transform_into_parsed("b").unwrap();
        assert_eq!(workspace_store.compile().statements.len(), 2);
        assert_eq!(
            workspace_store.compile().statements,
            vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")]
        );

        assert_eq!(
            workspace_store.compile().get_provenance(0),
            Ok(Provenance::Hypothesis)
        );
        // TODO Currently we don't derive the transformation addresses but this assertion should
        // fail
        assert_eq!(
            workspace_store.compile().get_provenance(1),
            Ok(Provenance::Derived((0, 0, vec![])))
        );

        assert_eq!(
            workspace_store.compile().get_provenance_lineage(0),
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
