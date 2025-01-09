use log::{debug, trace};
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
            AlgorithmTransformation, AvailableTransformation, ExplicitTransformation,
            Transformation, TransformationError, TransformationLattice, TransformationProvenance,
        },
    },
};

pub type StatementIndex = usize;
pub type TransformationIndex = usize;
pub type InterpretationIndex = usize;

type SymbolNodeString = String;
type TransformationString = String;

#[derive(Clone, Debug, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DisplayWorkspace {
    types: Vec<DisplayTypeHierarchyNode>,
    generated_types: Vec<DisplayGeneratedType>,
    interpretations: Vec<DisplayInterpretation>,
    statements: Vec<DisplaySymbolNode>,
    transformations: Vec<DisplayAvailableTransformation>,
    transformation_lattice: DisplayTransformationLattice,
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
            transformations: workspace.get_display_available_transformations()?,
            transformation_lattice: DisplayTransformationLattice::from_transformation_lattice(
                &workspace.interpretations,
                &workspace.transformation_lattice,
            ),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Workspace {
    types: TypeHierarchy,
    generated_types: Vec<GeneratedType>,
    interpretations: Vec<Interpretation>,
    transformation_lattice: TransformationLattice,
}

impl Default for Workspace {
    fn default() -> Self {
        Self::initialize(TypeHierarchy::new(), vec![], vec![])
    }
}

impl Workspace {
    pub fn new(
        types: TypeHierarchy,
        generated_types: Vec<GeneratedType>,
        interpretations: Vec<Interpretation>,
        transformation_lattice: TransformationLattice,
    ) -> Self {
        Self {
            types,
            generated_types,
            interpretations,
            transformation_lattice,
        }
    }

    pub fn initialize(
        types: TypeHierarchy,
        generated_types: Vec<GeneratedType>,
        interpretations: Vec<Interpretation>,
    ) -> Workspace {
        Self {
            types,
            generated_types,
            interpretations,
            transformation_lattice: TransformationLattice::empty(),
        }
    }

    pub fn scope_down(
        &self,
        maybe_statement_scope: Option<HashSet<SymbolNode>>,
        maybe_transformation_scope: Option<HashSet<AvailableTransformation>>,
    ) -> Result<Self, WorkspaceError> {
        Ok(Self::new(
            self.types.clone(),
            self.generated_types.clone(),
            self.interpretations.clone(),
            self.transformation_lattice
                .scope_down(maybe_statement_scope, maybe_transformation_scope)?,
        ))
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

    pub fn get_interpretation(
        &self,
        idx: InterpretationIndex,
    ) -> Result<&Interpretation, WorkspaceError> {
        if idx >= self.interpretations.len() {
            Err(WorkspaceError::InvalidInterpretationIndex(idx))
        } else {
            Ok(&self.interpretations[idx])
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
        self.transformation_lattice = context.get_transformation_lattice().clone();
        Ok(())
    }

    pub fn evaluate_from_string(&self, s: &str) -> Result<SymbolNode, WorkspaceError> {
        let statement = self.parse_from_string(s)?;
        self.evaluate(statement)
    }

    pub fn evaluate(&self, statement: SymbolNode) -> Result<SymbolNode, WorkspaceError> {
        debug!("evaluate()");
        let alg_only_workspace = self.copy_with_only_algorithms();
        let mut to_return = statement;
        let mut has_changed = true;
        while has_changed {
            let before = to_return.clone();
            let results =
                alg_only_workspace.get_valid_transformations_from(to_return.clone(), None, None)?;
            if results.len() == 0 {
                return Err(WorkspaceError::NoTransformationsPossible);
            }
            debug!("evaluate() results in transformations: {:?}", results);
            has_changed = false;
            for result in results {
                if result != before {
                    has_changed = true;
                    to_return = result;
                }
            }
        }
        Ok(to_return)
    }

    fn copy_with_only_algorithms(&self) -> Self {
        let lattice = self.transformation_lattice.copy_with_only_algorithms();
        Self::new(
            self.types.clone(),
            self.generated_types.clone(),
            self.interpretations.clone(),
            lattice,
        )
    }

    fn parse_from_string(&self, s: &str) -> Result<SymbolNode, WorkspaceError> {
        let parser = Parser::new(self.interpretations.clone());
        parser
            .parse_from_string(parser.get_interpretation_custom_tokens(), s)
            .map_err(|e| e.into())
    }

    pub fn get_statements(&self) -> &HashSet<SymbolNode> {
        self.transformation_lattice.get_statements()
    }

    pub fn get_ordered_statements(&self) -> Vec<SymbolNode> {
        self.transformation_lattice.get_ordered_statements()
    }

    pub fn contains_statement(&self, statement: &SymbolNode) -> bool {
        self.transformation_lattice.contains_statement(statement)
    }

    pub fn remove_statement_and_all_dependents(
        &mut self,
        statement: &SymbolNode,
    ) -> Result<HashSet<SymbolNode>, WorkspaceError> {
        self.transformation_lattice
            .remove_statement_and_all_dependents(statement)
            .map_err(|e| e.into())
    }

    pub fn get_available_transformations(&self) -> &HashSet<AvailableTransformation> {
        self.transformation_lattice.get_available_transformations()
    }

    pub fn get_ordered_available_transformations(&self) -> Vec<AvailableTransformation> {
        self.transformation_lattice
            .get_ordered_available_transformations()
    }

    pub fn contains_transformation(&self, transformation: &Transformation) -> bool {
        self.transformation_lattice
            .contains_transformation(transformation)
    }

    pub fn get_transformation_lattice(&self) -> &TransformationLattice {
        &self.transformation_lattice
    }

    pub fn get_statement(&self, index: StatementIndex) -> Result<SymbolNode, WorkspaceError> {
        if self.statement_index_is_invalid(index) {
            Err(WorkspaceError::InvalidStatementIndex(index))
        } else {
            Ok(self.get_ordered_statements()[index].clone())
        }
    }

    pub fn get_available_transformation(
        &self,
        index: TransformationIndex,
    ) -> Result<AvailableTransformation, WorkspaceError> {
        if self.transformation_index_is_invalid(index) {
            Err(WorkspaceError::InvalidTransformationIndex(index))
        } else {
            trace!(
                "ordered_available_transformations:\n{}",
                self.get_ordered_available_transformations()
                    .iter()
                    .map(|t| t.get_transformation().to_symbol_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            Ok(self.get_ordered_available_transformations()[index].clone())
        }
    }

    pub fn remove_transformation_and_all_dependents(
        &mut self,
        transformation: &Transformation,
    ) -> Result<(HashSet<Transformation>, HashSet<SymbolNode>), WorkspaceError> {
        self.transformation_lattice
            .remove_transformation_and_all_dependents(transformation)
            .map_err(|e| e.into())
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

    pub fn update_interpretation(
        &mut self,
        interpretation_index: InterpretationIndex,
        new_interpretation: Interpretation,
    ) -> Result<Interpretation, WorkspaceError> {
        if interpretation_index >= self.interpretations.len() {
            return Err(WorkspaceError::InvalidInterpretationIndex(
                interpretation_index,
            ));
        }
        self.interpretations[interpretation_index] = new_interpretation.clone();
        Ok(new_interpretation)
    }

    pub fn duplicate_interpretation(
        &mut self,
        interpretation_index: InterpretationIndex,
    ) -> Result<Interpretation, WorkspaceError> {
        if interpretation_index >= self.interpretations.len() {
            return Err(WorkspaceError::InvalidInterpretationIndex(
                interpretation_index,
            ));
        }
        let new_interpretation = self.interpretations[interpretation_index].clone();
        self.add_interpretation(new_interpretation.clone())?;
        Ok(new_interpretation)
    }

    pub fn remove_interpretation(
        &mut self,
        interpretation_index: InterpretationIndex,
    ) -> Result<Interpretation, WorkspaceError> {
        if interpretation_index >= self.interpretations.len() {
            return Err(WorkspaceError::InvalidInterpretationIndex(
                interpretation_index,
            ));
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
        if statement.contains_arbitrary_nodes() {
            return Err(WorkspaceError::StatementContainsArbitraryNode(statement));
        }
        self.transformation_lattice.add_hypothesis(statement)?;
        Ok(())
    }

    fn force_apply_transformation(
        &mut self,
        from_statement: SymbolNode,
        transformation: AvailableTransformation,
        to_statement: SymbolNode,
    ) {
        self.transformation_lattice.force_apply_transformation(
            from_statement,
            transformation,
            to_statement,
        );
    }

    fn force_remove_transformations(&mut self, transformations: HashSet<Transformation>) {
        self.transformation_lattice
            .force_remove_transformations(&transformations)
    }

    fn force_derive_theorem(
        &mut self,
        transformation: Transformation,
        provenance: TransformationProvenance,
    ) {
        self.transformation_lattice
            .force_derive_theorem(transformation, provenance);
    }

    fn force_remove_statements(&mut self, statements: HashSet<SymbolNode>) {
        self.transformation_lattice
            .force_remove_statements(&statements)
    }

    fn get_upstream_statement_and_transformation(
        &self,
        statement: &SymbolNode,
    ) -> Option<(SymbolNode, Transformation)> {
        self.transformation_lattice
            .get_upstream_statement_and_transformation(statement)
    }

    fn add_axiom(&mut self, transformation: Transformation) -> Result<(), WorkspaceError> {
        self.types.binds_transformation_or_error(&transformation)?;
        self.transformation_lattice
            .add_available_transformation(AvailableTransformation::Axiom(transformation))
            .map_err(|e| e.into())
    }

    pub fn get_valid_transformations(
        &self,
        partial_statement: &str,
        maybe_statement_scope: Option<HashSet<SymbolNode>>,
        maybe_transformation_scope: Option<HashSet<AvailableTransformation>>,
    ) -> Result<Vec<SymbolNode>, WorkspaceError> {
        // TODO Try to complete partial statements
        let desired = match self.parse_from_string(partial_statement) {
            Err(_) => {
                return Ok(vec![]);
            }
            Ok(s) => s,
        };
        let scoped_ws = self.scope_down(maybe_statement_scope, maybe_transformation_scope)?;
        debug!("scoped_down_workspace: {}", scoped_ws.to_json().unwrap());
        let instantiated_transformations = scoped_ws
            .transformation_lattice
            .get_instantiated_transformations_with_arbitrary(
                scoped_ws.get_types(),
                Some(desired.clone()),
            )?;
        debug!(
            "instantiated_transformations:\n{}",
            instantiated_transformations
                .clone()
                .into_iter()
                .map(|(instantiated, arbitrary)| format!(
                    "Instantiated: {}; Arbitrary: {}",
                    instantiated.to_symbol_string(),
                    arbitrary.to_symbol_string()
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );
        for (instantiated, _) in instantiated_transformations {
            let instantiated_transformation = instantiated.get_transformation();
            let statements = if instantiated_transformation.is_joint_transform() {
                scoped_ws.transformation_lattice.get_statement_pairs()
            } else {
                scoped_ws.transformation_lattice.get_statements().clone()
            };
            for statement in statements {
                let valid_transformations = instantiated_transformation.get_valid_transformations(
                    scoped_ws.get_types(),
                    &statement,
                    Some(&desired),
                );
                if valid_transformations.contains(&desired)
                    && !scoped_ws.get_statements().contains(&desired)
                {
                    return Ok(vec![desired.clone()]);
                }
            }
        }
        return Ok(vec![]);
    }

    pub fn get_valid_transformations_from(
        &self,
        from_statement: SymbolNode,
        maybe_statement_scope: Option<HashSet<SymbolNode>>,
        maybe_transformation_scope: Option<HashSet<AvailableTransformation>>,
    ) -> Result<Vec<SymbolNode>, WorkspaceError> {
        let scoped_ws = self.scope_down(maybe_statement_scope, maybe_transformation_scope)?;
        debug!("scoped_down_workspace: {}", scoped_ws.to_json().unwrap());
        let instantiated_transformations = scoped_ws
            .transformation_lattice
            .get_instantiated_transformations_with_arbitrary(scoped_ws.get_types(), None)?;
        debug!(
            "instantiated_transformations:\n{}",
            instantiated_transformations
                .clone()
                .into_iter()
                .map(|(instantiated, arbitrary)| format!(
                    "instantiated: {}; arbitrary: {}",
                    instantiated.to_symbol_string(),
                    arbitrary.to_symbol_string()
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );
        let mut to_return = HashSet::new();
        for (transformation, _) in instantiated_transformations {
            let instantiated_transformation = transformation.get_transformation();
            let statements = if instantiated_transformation.is_joint_transform() {
                scoped_ws
                    .transformation_lattice
                    .get_statement_pairs_including(Some(&from_statement))
            } else {
                vec![from_statement.clone()].into_iter().collect()
            };
            for statement in statements {
                let valid_transformations = instantiated_transformation.get_valid_transformations(
                    scoped_ws.get_types(),
                    &statement,
                    None,
                );
                to_return.extend(valid_transformations);
            }
        }
        return Ok(to_return
            .into_iter()
            .filter(|s| !self.get_statements().contains(s))
            .collect());
    }

    pub fn get_display_available_transformations(
        &self,
    ) -> Result<Vec<DisplayAvailableTransformation>, WorkspaceError> {
        let mut to_return = Vec::new();
        for i in 0..self.get_ordered_available_transformations().len() {
            let t = self.get_display_available_transformation(i)?;
            to_return.push(t);
        }
        Ok(to_return)
    }

    pub fn get_display_available_transformation(
        &self,
        index: usize,
    ) -> Result<DisplayAvailableTransformation, WorkspaceError> {
        let available_transformation = self.get_available_transformation(index)?;
        Ok(
            DisplayAvailableTransformation::from_available_transformation(
                self.get_interpretations(),
                available_transformation,
            ),
        )
    }

    pub fn get_display_symbol_nodes(&self) -> Result<Vec<DisplaySymbolNode>, WorkspaceError> {
        let mut to_return = Vec::new();
        for i in 0..self.get_statements().len() {
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
        let (upstream_statement, upstream_transformation) =
            match self.get_upstream_statement_and_transformation(&statement) {
                Some((s, t)) => (
                    Some(s.to_interpreted_string(&self.interpretations)),
                    Some(t.to_interpreted_string(&self.interpretations)),
                ),
                None => (None, None),
            };
        Ok(DisplaySymbolNode::new(
            interpreted_string,
            type_map
                .into_iter()
                .map(|(name, t)| (name, t.to_string()))
                .collect(),
            upstream_statement,
            upstream_transformation,
        ))
    }

    pub fn get_generated_types(&self) -> &Vec<GeneratedType> {
        &self.generated_types
    }

    fn add_generated_type(&mut self, generated_type: GeneratedType) {
        self.generated_types.push(generated_type);
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
        index >= self.get_statements().len()
    }

    fn transformation_index_is_invalid(&self, index: TransformationIndex) -> bool {
        index >= self.get_available_transformations().len()
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
    command_history: CommandHistory,
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
            command_history: CommandHistory::empty(),
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
            WorkspaceTransactionItem::DuplicateInterpretation(idx) => {
                workspace.duplicate_interpretation(idx)?;
            }
            WorkspaceTransactionItem::UpdateInterpretation((idx, interpretation)) => {
                workspace.update_interpretation(idx, interpretation)?;
            }
            WorkspaceTransactionItem::RemoveInterpretation(index) => {
                workspace.remove_interpretation(index)?;
            }
            WorkspaceTransactionItem::AddHypothesis(hypothesis) => {
                workspace.add_hypothesis(hypothesis)?;
            }
            WorkspaceTransactionItem::RemoveStatements(statements) => {
                workspace.force_remove_statements(statements)
            }
            WorkspaceTransactionItem::AddAxiom(transformation) => {
                workspace.add_axiom(transformation)?;
            }
            WorkspaceTransactionItem::AddTheorem((transformation, provenance)) => {
                workspace.force_derive_theorem(transformation, provenance);
            }
            WorkspaceTransactionItem::RemoveTransformations(transformations) => {
                workspace.force_remove_transformations(transformations)
            }
            WorkspaceTransactionItem::Derive((from_statement, transform, to_statement)) => {
                workspace.force_apply_transformation(from_statement, transform, to_statement);
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
        transaction.add(WorkspaceTransactionItem::AddAxiom(transformation.clone()));
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
        transaction.add(WorkspaceTransactionItem::AddAxiom(transformation.clone()));
        if is_equivalence {
            let transformation: Transformation =
                ExplicitTransformation::new(parsed_to, parsed_from).into();
            transaction.add(WorkspaceTransactionItem::AddAxiom(transformation.clone()));
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
        let transaction = WorkspaceTransactionItem::AddAxiom(transformation.clone()).into();
        self.add(transaction)?;
        Ok(transformation)
    }

    pub fn remove_transformation_and_all_dependents(
        &mut self,
        transformation: &Transformation,
    ) -> Result<HashSet<Transformation>, WorkspaceError> {
        let mut workspace = self.compile();
        let (transformations_to_remove, statements_to_remove) =
            workspace.remove_transformation_and_all_dependents(transformation)?;
        let mut transaction: WorkspaceTransaction =
            WorkspaceTransactionItem::RemoveTransformations(transformations_to_remove.clone())
                .into();
        transaction.add(WorkspaceTransactionItem::RemoveStatements(
            statements_to_remove.clone(),
        ));
        self.add(transaction)?;
        Ok(transformations_to_remove)
    }
    pub fn try_transform_into_parsed(
        &mut self,
        desired: &str,
        maybe_statement_scope: Option<HashSet<SymbolNode>>,
        maybe_transformation_scope: Option<HashSet<AvailableTransformation>>,
    ) -> Result<SymbolNode, WorkspaceError> {
        let parsed = self.compile().parse_from_string(desired)?;
        self.try_transform_into(parsed, maybe_statement_scope, maybe_transformation_scope)
    }

    pub fn try_transform_into(
        &mut self,
        desired: SymbolNode,
        maybe_statement_scope: Option<HashSet<SymbolNode>>,
        maybe_transformation_scope: Option<HashSet<AvailableTransformation>>,
    ) -> Result<SymbolNode, WorkspaceError> {
        let mut transaction = self.get_generated_types_transaction(&desired)?;

        // Ensures that the type hierarchy we pass around includes any generated types without
        // mucking up the transaction we eventually apply
        let type_hierarchy = self.get_type_hierarchy_with_transaction_applied(&transaction)?;

        let mut workspace = self
            .compile()
            .scope_down(maybe_statement_scope, maybe_transformation_scope)?;
        debug!("scoped_down_workspace: {}", workspace.to_json().unwrap());

        let (from_statement, transformation, to_statement) = workspace
            .transformation_lattice
            .try_transform_into(&type_hierarchy, desired.clone())?;
        transaction.add(WorkspaceTransactionItem::Derive((
            from_statement,
            transformation,
            to_statement,
        )));
        self.add(transaction)?;
        Ok(desired)
    }

    pub fn try_derive_theorem_parsed(&mut self, s: &str) -> Result<SymbolNode, WorkspaceError> {
        let conclusion = self.compile().parse_from_string(s)?;
        self.try_derive_theorem(conclusion)
    }

    pub fn try_derive_theorem(
        &mut self,
        conclusion: SymbolNode,
    ) -> Result<SymbolNode, WorkspaceError> {
        let mut workspace = self.compile();
        let transformation = workspace
            .transformation_lattice
            .derive_theorem(&conclusion)?;
        let transaction = WorkspaceTransactionItem::AddTheorem(transformation).into();
        self.add(transaction)?;
        Ok(conclusion)
    }

    pub fn remove_statement_and_all_dependents(
        &mut self,
        statement: &SymbolNode,
    ) -> Result<HashSet<SymbolNode>, WorkspaceError> {
        let mut workspace = self.compile();
        let to_remove = workspace.remove_statement_and_all_dependents(statement)?;
        let transaction = WorkspaceTransactionItem::RemoveStatements(to_remove.clone()).into();
        self.add(transaction)?;
        Ok(to_remove)
    }

    fn get_type_hierarchy_with_transaction_applied(
        &mut self,
        transaction: &WorkspaceTransaction,
    ) -> Result<TypeHierarchy, WorkspaceError> {
        let mut self_cloned = self.clone();
        self_cloned.add(transaction.clone())?;
        let type_hierarchy = self_cloned.compile().get_types().clone();
        Ok(type_hierarchy)
    }

    fn get_generated_types_transaction(
        &mut self,
        statement: &SymbolNode,
    ) -> Result<WorkspaceTransaction, WorkspaceError> {
        let workspace = self.compile();
        let mut items = Vec::new();
        let mut already_added = HashSet::new();
        for generated_type in workspace.get_generated_types().clone() {
            for (t, parents) in generated_type.generate(statement) {
                if !already_added.contains(&t) && !workspace.get_types().contains_type(&t) {
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

    pub fn add_command_to_history(&mut self, command: String) {
        self.command_history.add(command);
    }

    pub fn get_command_history_as_string(&self) -> String {
        self.command_history.to_string()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandHistory {
    commands: Vec<String>,
}

impl CommandHistory {
    pub fn new(commands: Vec<String>) -> Self {
        Self { commands }
    }

    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    pub fn add(&mut self, command: String) {
        self.commands.push(command);
    }

    pub fn add_commands(&mut self, mut commands: Vec<String>) {
        self.commands.append(&mut commands);
    }

    pub fn to_string(&self) -> String {
        self.commands.join("\n")
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
    DuplicateInterpretation(InterpretationIndex),
    UpdateInterpretation((InterpretationIndex, Interpretation)),
    RemoveInterpretation(InterpretationIndex),
    AddHypothesis(SymbolNode),
    RemoveStatements(HashSet<SymbolNode>),
    AddAxiom(Transformation),
    AddTheorem((Transformation, TransformationProvenance)),
    RemoveTransformations(HashSet<Transformation>),
    Derive((SymbolNode, AvailableTransformation, SymbolNode)),
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
    from_transformation: TransformationString,
    from_statement: SymbolNodeString,
    applied_addresses: Vec<SymbolNodeAddress>,
}

impl DerivedDisplayProvenance {
    pub fn new(
        from_transformation: TransformationString,
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
    Derived(
        (
            TransformationIndex,
            StatementProvenance,
            Vec<SymbolNodeAddress>,
        ),
    ),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatementProvenance {
    Single(StatementIndex),
    Joint((StatementIndex, StatementIndex)),
}

#[derive(Clone, Debug, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DisplayTransformationLattice {
    nodes: Vec<DisplayTransformationLatticeNode>,
    links: Vec<DisplayTransformationLatticeLink>,
}

impl DisplayTransformationLattice {
    pub fn new(
        nodes: Vec<DisplayTransformationLatticeNode>,
        links: Vec<DisplayTransformationLatticeLink>,
    ) -> Self {
        Self { nodes, links }
    }

    pub fn from_transformation_lattice(
        interpretations: &Vec<Interpretation>,
        lattice: &TransformationLattice,
    ) -> Self {
        let node_to_id = lattice
            .get_ordered_statements()
            .iter()
            .enumerate()
            .map(|(i, statement)| (statement.to_interpreted_string(interpretations), i))
            .collect::<HashMap<String, usize>>();
        let links = lattice
            .get_ordered_applied_transformations()
            .into_iter()
            .map(|(from, transformation, to)| {
                let separate_froms = if from.is_join() {
                    assert_eq!(from.get_n_children(), 2);
                    from.get_children().clone()
                } else {
                    vec![from]
                };
                let mut to_return = Vec::new();
                for separate_from in separate_froms {
                    let new_link = DisplayTransformationLatticeLink::new(
                        node_to_id
                            .get(&separate_from.to_interpreted_string(interpretations))
                            .expect(&format!(
                                "Nodes are coming from the same lattice but {} is missing (from_statement).",
                                separate_from.to_interpreted_string(interpretations)
                            ))
                            .to_string(),
                        transformation.to_interpreted_string(interpretations),
                        node_to_id
                            .get(&to.to_interpreted_string(interpretations))
                            .expect(&format!(
                                "Nodes are coming from the same lattice but {} is missing (to_statement).",
                                to.to_interpreted_string(interpretations)
                            ))
                            .to_string(),
                    );
                    to_return.push(new_link);
                }
                to_return
            }).flatten()
            .collect::<Vec<_>>();
        let nodes = node_to_id
            .clone()
            .into_iter()
            .map(|(node, id)| {
                // We're better off computing this above but this is fine for now
                DisplayTransformationLatticeNode::new(
                    id.to_string(),
                    node,
                    !links.iter().any(|link| link.target == id.to_string()),
                )
            })
            .collect();
        Self::new(nodes, links)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DisplayTransformationLatticeNode {
    id: String,
    description: String,
    is_initial: bool,
}

impl DisplayTransformationLatticeNode {
    pub fn new(id: String, description: String, is_initial: bool) -> Self {
        Self {
            id,
            description,
            is_initial,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DisplayTransformationLatticeLink {
    source: String,
    relation: String,
    target: String,
}

impl DisplayTransformationLatticeLink {
    pub fn new(source: String, relation: String, target: String) -> Self {
        Self {
            source,
            relation,
            target,
        }
    }
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

#[derive(Clone, Debug, Serialize, Deserialize, TS)]
#[serde(tag = "kind", content = "value", rename_all = "camelCase")]
#[ts(export)]
pub enum DisplayAvailableTransformation {
    Axiom(DisplayAxiom),
    Theorem(DisplayTheorem),
}

impl DisplayAvailableTransformation {
    pub fn from_available_transformation(
        interpretations: &Vec<Interpretation>,
        t: AvailableTransformation,
    ) -> Self {
        match t {
            AvailableTransformation::Axiom(t) => {
                Self::Axiom(DisplayAxiom::new(t.to_interpreted_string(interpretations)))
            }
            AvailableTransformation::Theorem((t, p)) => Self::Theorem(DisplayTheorem::new(
                t.to_interpreted_string(interpretations),
                DisplayTransformationProvenance::from_transformation_provenance(interpretations, p),
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DisplayAxiom {
    statement: String,
}

impl DisplayAxiom {
    pub fn new(statement: String) -> Self {
        Self { statement }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DisplayTheorem {
    statement: String,
    provenance: DisplayTransformationProvenance,
}

impl DisplayTheorem {
    pub fn new(statement: String, provenance: DisplayTransformationProvenance) -> Self {
        Self {
            statement,
            provenance,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export)]
pub struct DisplayTransformationProvenance {
    derivation: String,
}

impl DisplayTransformationProvenance {
    pub fn new(derivation: String) -> Self {
        Self { derivation }
    }

    pub fn from_transformation_provenance(
        interpretations: &Vec<Interpretation>,
        provenance: TransformationProvenance,
    ) -> Self {
        Self::new(format!(
            "Derivation of {}",
            provenance.conclusion.to_interpreted_string(interpretations)
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceError {
    InvalidStatementIndex(StatementIndex),
    InvalidTransformationIndex(TransformationIndex),
    InvalidInterpretationIndex(InterpretationIndex),
    InvalidTransformationAddress,
    StatementContainsArbitraryNode(SymbolNode),
    ParserError(ParserError),
    UnableToSerialize(String),
    TransformationError(TransformationError),
    StatementContainsTypesNotInHierarchy(HashSet<Type>),
    MissingStatementsInTransformationLattice(Vec<SymbolNode>),
    IncompatibleTypeRelationships(HashSet<Type>),
    ConflictingTypes(String, Type, Type),
    StatementsAlreadyInclude(SymbolNode),
    TransformationsAlreadyInclude(Transformation),
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
            TransformationError::StatementContainsArbitraryNode(s) => {
                Self::StatementContainsArbitraryNode(s)
            }
            TransformationError::ArbitraryNodeHasNonOneChildren => {
                Self::ArbitraryNodeHasNonOneChildren
            }
            TransformationError::MissingStatementsInTransformationLattice(statements) => {
                Self::MissingStatementsInTransformationLattice(statements)
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
        parsing::interpretation::Interpretation,
        symbol::{
            symbol_node::Symbol,
            symbol_type::{GeneratedType, GeneratedTypeCondition, TypeHierarchy},
        },
    };

    use super::*;

    #[test]
    fn test_workspace_store_serializes() {
        let mut store = WorkspaceTransactionStore::empty();
        store
            .add(WorkspaceTransactionItem::AddType(("Real".into(), Type::Object)).into())
            .unwrap();
        store
            .add(
                WorkspaceTransactionItem::AddInterpretation(Interpretation::singleton(
                    "x".into(),
                    "Real".into(),
                ))
                .into(),
            )
            .unwrap();
        store
            .add(
                WorkspaceTransactionItem::AddInterpretation(Interpretation::infix_operator(
                    "+".into(),
                    1,
                    "Real".into(),
                ))
                .into(),
            )
            .unwrap();
        let workspace = store.compile();
        let parse = |s: &str| workspace.parse_from_string(s).unwrap();
        let mut sum = "x".to_string();
        for _ in 1..50 {
            sum = format!("{}{}", sum, "+x");
            store
                .add(WorkspaceTransactionItem::AddHypothesis(parse(&sum)).into())
                .unwrap();
        }
        let serialized = store.serialize().unwrap();
        let deserialized: WorkspaceTransactionStore = toml::from_str(&serialized).unwrap();
        assert_eq!(deserialized, store);
    }

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

        store
            .add(WorkspaceTransactionItem::AddType(("Real".into(), Type::Object)).into())
            .unwrap();

        let expected_0 = Workspace::initialize(
            TypeHierarchy::chain(vec!["Real".into()]).unwrap(),
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(store.compile(), expected_0);

        store
            .add(WorkspaceTransactionItem::AddType(("Integer".into(), "Real".into())).into())
            .unwrap();

        let expected_1 = Workspace::initialize(
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
    pub fn test_workspace_try_transform_into_with_substitution() {
        let mut types = TypeHierarchy::chain(vec!["Boolean".into(), "=".into()]).unwrap();
        types.add_chain(vec!["Real".into(), "+".into()]).unwrap();
        types
            .add_child_to_parent("-".into(), "Real".into())
            .unwrap();
        types
            .add_child_to_parent("*".into(), "Real".into())
            .unwrap();
        types
            .add_child_to_parent("/".into(), "Real".into())
            .unwrap();
        types
            .add_child_to_parent("^".into(), "Real".into())
            .unwrap();
        types
            .add_child_to_parent("Negative".into(), "Real".into())
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("+".into(), 2, "+".into()),
            Interpretation::infix_operator("-".into(), 2, "-".into()),
            Interpretation::infix_operator("*".into(), 3, "*".into()),
            Interpretation::infix_operator("/".into(), 3, "/".into()),
            Interpretation::infix_operator("^".into(), 4, "^".into()),
            Interpretation::singleton("p".into(), "Boolean".into()),
            Interpretation::singleton("q".into(), "Boolean".into()),
            Interpretation::singleton("r".into(), "Boolean".into()),
            Interpretation::singleton("s".into(), "Boolean".into()),
            Interpretation::singleton("x".into(), "Real".into()),
            Interpretation::singleton("y".into(), "Real".into()),
            Interpretation::singleton("x".into(), "Real".into()),
            Interpretation::singleton("a".into(), "Real".into()),
            Interpretation::singleton("b".into(), "Real".into()),
            Interpretation::singleton("c".into(), "Real".into()),
            Interpretation::function("Negative".into(), 99),
            Interpretation::arbitrary_functional("Any".into(), 99, "Real".into()),
        ];

        let real_interpretation = GeneratedType::new_numeric("Real".into());

        let setup_and_transform = |hypotheses, transformations, desired| {
            let workspace = Workspace::initialize(
                types.clone(),
                vec![real_interpretation],
                interpretations.clone(),
            );

            let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace.clone());

            for hypothesis in hypotheses {
                workspace_store.add_parsed_hypothesis(hypothesis).unwrap();
            }

            for (from, to) in transformations {
                workspace_store
                    .add_parsed_transformation(false, from, to)
                    .unwrap();
            }

            let actual = workspace_store
                .try_transform_into_parsed(desired, None, None)
                .unwrap();
            let expected = workspace_store
                .compile()
                .parse_from_string(desired)
                .unwrap();
            return (actual, expected);
        };

        let hypotheses = vec!["0=a*x+b", "x=b+c"];

        let transformations = vec![("x=y", "Any(x)=Any(y)")];

        let desired = "a*x+b=a*(b+c)+b";

        let (actual, expected) = setup_and_transform.clone()(hypotheses, transformations, desired);
        assert_eq!(actual, expected);

        let hypotheses = vec![
            "0=a*x^2+b*x+c",
            "x=((Negative(b)+(b^2 - 4*a*c)^(1/2))/(2*a))",
        ];

        let transformations = vec![("x=y", "Any(x)=Any(y)")];

        let desired = "a*x^2+b*x+c=a*x^2+b*(((Negative(b)+(b^2 - 4*a*c)^(1/2))/(2*a)))+c";

        let (actual, expected) = setup_and_transform.clone()(hypotheses, transformations, desired);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_workspace_try_transform_into_with_arbitrary() {
        let mut types = TypeHierarchy::chain(vec!["Boolean".into(), "=".into()]).unwrap();
        types
            .add_child_to_parent("^".into(), "Boolean".into())
            .unwrap();
        types
            .add_child_to_parent("+".into(), "Boolean".into())
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("+".into(), 2, "+".into()),
            Interpretation::infix_operator("^".into(), 3, "^".into()),
            Interpretation::singleton("p".into(), "Boolean".into()),
            Interpretation::singleton("q".into(), "Boolean".into()),
            Interpretation::singleton("r".into(), "Boolean".into()),
            Interpretation::singleton("s".into(), "Boolean".into()),
            Interpretation::arbitrary_functional("Any".into(), 99, "Boolean".into()),
        ];

        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
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
                .try_transform_into_parsed("p^s=q^s", None, None)
                .unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));

        let actual = workspace_store
            .try_transform_into_parsed("p+q^s=q+q^s", None, None)
            .unwrap();

        let expected = workspace_store
            .compile()
            .parse_from_string("p+q^s=q+q^s")
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_workspace_instantiates_arbitrary_to_add_on_both_sides() {
        let mut types = TypeHierarchy::chain(vec!["Boolean".into(), "=".into()]).unwrap();
        types
            .add_child_to_parent("^".into(), "Boolean".into())
            .unwrap();
        types
            .add_chain(vec!["Real".into(), "+".into(), "2".into()])
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("+".into(), 2, "+".into()),
            Interpretation::infix_operator("^".into(), 3, "^".into()),
            Interpretation::singleton("p".into(), "Boolean".into()),
            Interpretation::singleton("q".into(), "Boolean".into()),
            Interpretation::singleton("r".into(), "Boolean".into()),
            Interpretation::singleton("s".into(), "Boolean".into()),
            Interpretation::singleton("x".into(), "Real".into()),
            Interpretation::singleton("y_0".into(), "Real".into()), //  For disambiguation
            Interpretation::singleton("y".into(), "Real".into()),
            Interpretation::singleton("x".into(), "Real".into()),
            Interpretation::arbitrary_functional("Any".into(), 99, "Real".into()),
        ];

        let real_interpretation = GeneratedType::new_numeric("Real".into());

        let workspace = Workspace::initialize(
            types.clone(),
            vec![real_interpretation],
            interpretations.clone(),
        );

        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace.clone());

        workspace_store
            .add_parsed_transformation(false, "x=y", "Any(x)=Any(y)")
            .unwrap();
        workspace_store.add_parsed_hypothesis("x=5").unwrap();
        workspace_store.add_parsed_hypothesis("y=10").unwrap();

        let instantiate = |from: &str, to: &str| {
            AvailableTransformation::Axiom(
                ExplicitTransformation::new(
                    workspace_store.compile().parse_from_string(from).unwrap(),
                    workspace_store.compile().parse_from_string(to).unwrap(),
                )
                .into(),
            )
        };

        // Looking for x+y=5+y but that'll appear as the more general x=y -> ...
        // except that y will be replaced with y_0 to disambiguate the transform y
        // from the predicate y
        let expected = instantiate("x=y_0", "x+y=y_0+y");
        let workspace = workspace_store.compile();
        let results = workspace
            .transformation_lattice
            .get_instantiated_transformations_with_arbitrary(
                &types,
                Some(workspace.parse_from_string("x+y=5+y").unwrap()),
            )
            .unwrap()
            .into_iter()
            .map(|(r, _)| r)
            .collect::<HashSet<_>>();
        assert!(
            results.contains(&expected),
            "\n{}",
            results
                .clone()
                .into_iter()
                .map(|t| t.to_symbol_string())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn test_workspace_instantiates_arbitrary_transforms() {
        let mut types = TypeHierarchy::chain(vec!["Boolean".into(), "=".into()]).unwrap();
        types
            .add_child_to_parent("^".into(), "Boolean".into())
            .unwrap();
        types
            .add_chain(vec!["Real".into(), "+".into(), "2".into()])
            .unwrap();

        let interpretations = vec![
            Interpretation::infix_operator("=_0".into(), 1, "=".into()),
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("+".into(), 2, "+".into()),
            Interpretation::infix_operator("^".into(), 3, "^".into()),
            Interpretation::singleton("p_0".into(), "Boolean".into()),
            Interpretation::singleton("p".into(), "Boolean".into()),
            Interpretation::singleton("q_0".into(), "Boolean".into()),
            Interpretation::singleton("q".into(), "Boolean".into()),
            Interpretation::singleton("r".into(), "Boolean".into()),
            Interpretation::singleton("s".into(), "Boolean".into()),
            Interpretation::singleton("x".into(), "Real".into()),
            Interpretation::singleton("y".into(), "Real".into()),
            Interpretation::singleton("x".into(), "Real".into()),
            Interpretation::arbitrary_functional("Any".into(), 99, "Boolean".into()),
        ];

        let real_interpretation = GeneratedType::new_numeric("Real".into());

        let workspace = Workspace::initialize(
            types.clone(),
            vec![real_interpretation],
            interpretations.clone(),
        );
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace.clone());

        workspace_store
            .add_parsed_transformation(false, "p=q", "Any(p)=Any(q)")
            .unwrap();
        workspace_store.add_parsed_hypothesis("p=q").unwrap();
        workspace_store.add_parsed_hypothesis("p^s").unwrap();

        let instantiate = |from: &str, to: &str| {
            ExplicitTransformation::new(
                workspace_store.compile().parse_from_string(from).unwrap(),
                workspace_store.compile().parse_from_string(to).unwrap(),
            )
            .into()
        };

        let expected = vec![
            instantiate("p=q", "p=q"),
            instantiate("p=_0q_0", "(p=q)=_0(q_0=q)"),
            instantiate("p_0=_0q", "(p=p_0)=_0(p=q)"),
            instantiate("p=q", "(p^s)=(q^s)"),
            instantiate("p_0=q", "(p^p_0)=(p^q)"),
        ]
        .into_iter()
        .collect();
        let workspace = workspace_store.compile();
        let actual: HashSet<Transformation> = workspace
            .transformation_lattice
            .get_instantiated_transformations_with_arbitrary(&types, None)
            .unwrap()
            .into_iter()
            .map(|(t, _)| t.get_transformation().clone())
            .collect();
        assert_eq!(
            actual,
            expected,
            "actual:\n{}\n\nexpected:\n{}",
            actual
                .iter()
                .map(|t| t.to_interpreted_string(&interpretations))
                .collect::<Vec<_>>()
                .join("\n"),
            expected
                .iter()
                .map(|t| t.to_interpreted_string(&interpretations))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);

        workspace_store
            .add_parsed_transformation(false, "Any(p)", "p=p")
            .unwrap();
        workspace_store.add_parsed_hypothesis("p=q").unwrap();
        workspace_store.add_parsed_hypothesis("p^s").unwrap();

        let expected: HashSet<_> = vec![
            instantiate("p", "p=p"),
            instantiate("p=q", "p=_0p"),
            instantiate("p=p_0", "p_0=_0p_0"),
            instantiate("p^s", "p=p"),
            instantiate("p^p_0", "p_0=p_0"),
        ]
        .into_iter()
        .collect();

        let actual: HashSet<_> = workspace_store
            .compile()
            .transformation_lattice
            .get_instantiated_transformations_with_arbitrary(&types, None)
            .unwrap()
            .into_iter()
            .map(|(t, _)| t.get_transformation().clone())
            .collect();
        assert_eq!(
            actual,
            expected,
            "Actual:\n{}\n\nExpected:\n{}",
            actual
                .iter()
                .map(|n| n.to_interpreted_string(&interpretations))
                .collect::<Vec<_>>()
                .join("\n"),
            expected
                .iter()
                .map(|n| n.to_interpreted_string(&interpretations))
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

        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        assert!(workspace_store.add_parsed_hypothesis("Any(p)").is_err(),);
        assert!(workspace_store.add_parsed_hypothesis("Any(p)=q").is_err());

        workspace_store.add_parsed_hypothesis("p=q").unwrap();
        assert!(workspace_store
            .try_transform_into_parsed("Any(p)=Any(q)", None, None)
            .is_err());
    }

    #[test]
    fn test_workspace_derives_theorem_and_deletes_statements() {
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
        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);

        workspace_store
            .add_parsed_transformation(false, "a+b", "b+a")
            .unwrap();

        workspace_store.add_parsed_hypothesis("x+y").unwrap();
        let expected = workspace_store.compile().parse_from_string("y+x").unwrap();
        assert_eq!(
            workspace_store
                .try_transform_into_parsed("y+x", None, None)
                .unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));

        // Can't derive this -- we don't have any statement with a and b in it
        let invalid_derivation = workspace_store.try_derive_theorem_parsed("a+b");
        assert!(invalid_derivation.is_err(), "{:#?}", invalid_derivation);

        let b_plus_a = workspace_store.add_parsed_hypothesis("b+a").unwrap();
        workspace_store
            .try_transform_into_parsed("a+b", None, None)
            .unwrap();
        // Derive b+a -> a+b
        let a_plus_b = workspace_store.try_derive_theorem_parsed("a+b").unwrap();

        assert_eq!(
            workspace_store
                .compile()
                .get_available_transformations()
                .len(),
            2
        );
        assert_eq!(workspace_store.compile().get_statements().len(), 4);

        let c = workspace_store.add_parsed_hypothesis("c").unwrap();
        assert_eq!(workspace_store.compile().get_statements().len(), 5);
        assert_eq!(
            workspace_store
                .remove_statement_and_all_dependents(&c.clone())
                .unwrap(),
            vec![c].into_iter().collect()
        );
        assert_eq!(
            workspace_store.compile().get_statements().len(),
            4,
            "\n{}",
            workspace_store
                .compile()
                .get_statements()
                .iter()
                .map(|s| s.to_symbol_string())
                .collect::<Vec<_>>()
                .join("\n")
        );

        workspace_store
            .remove_statement_and_all_dependents(&b_plus_a.clone())
            .unwrap();

        assert_eq!(
            workspace_store.compile().get_statements().len(),
            2,
            "\n{}",
            workspace_store
                .compile()
                .get_statements()
                .iter()
                .map(|s| s.to_symbol_string())
                .collect::<Vec<_>>()
                .join("\n")
        );

        assert_eq!(
            workspace_store
                .compile()
                .get_available_transformations()
                .len(),
            2
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
        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);

        workspace_store
            .add_parsed_transformation(false, "a+b", "b+a")
            .unwrap();

        workspace_store.add_parsed_hypothesis("x+y").unwrap();
        let expected = workspace_store.compile().parse_from_string("y+x").unwrap();
        assert_eq!(
            workspace_store
                .try_transform_into_parsed("y+x", None, None)
                .unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));

        workspace_store.add_parsed_hypothesis("j+k").unwrap();
        let expected = workspace_store.compile().parse_from_string("k+j").unwrap();
        assert_eq!(
            workspace_store
                .try_transform_into_parsed("k+j", None, None)
                .unwrap(),
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
                .try_transform_into_parsed("(b+c)+a", None, None)
                .unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));

        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
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
                .try_transform_into_parsed("a+(c+b)", None, None)
                .unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));

        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        workspace_store
            .add_parsed_joint_transformation("p", "q", "p^q")
            .unwrap();

        workspace_store.add_parsed_hypothesis("r").unwrap();
        workspace_store.add_parsed_hypothesis("s").unwrap();

        let expected = workspace_store.compile().parse_from_string("r^s").unwrap();
        assert_eq!(
            workspace_store
                .try_transform_into_parsed("r^s", None, None)
                .unwrap(),
            expected
        );
        assert!(workspace_store
            .compile()
            .get_statements()
            .contains(&expected));

        let expected = workspace_store.compile().parse_from_string("s^r").unwrap();
        assert_eq!(
            workspace_store
                .try_transform_into_parsed("s^r", None, None)
                .unwrap(),
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
        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        workspace_store
            .add_parsed_transformation(false, "a+b", "b+a")
            .unwrap();

        let x_plus_y = workspace_store.add_parsed_hypothesis("x+y").unwrap();
        let expected = vec![workspace_store.compile().parse_from_string("y+x").unwrap()];
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations_from(x_plus_y.clone(), None, None)
                .unwrap(),
            expected
        );
        let y_plus_x = workspace_store
            .try_transform_into_parsed("y+x", None, None)
            .unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations_from(x_plus_y.clone(), None, None)
                .unwrap(),
            vec![],
        );
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations_from(y_plus_x.clone(), None, None)
                .unwrap(),
            vec![],
        );

        let j_plus_k = workspace_store.add_parsed_hypothesis("j+k").unwrap();
        assert_eq!(workspace_store.compile().get_statements().len(), 3);
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations_from(x_plus_y.clone(), None, None)
                .unwrap(),
            vec![],
        );

        let k_plus_j = workspace_store.compile().parse_from_string("k+j").unwrap();
        let expected = vec![k_plus_j.clone()];
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations_from(j_plus_k, None, None)
                .unwrap(),
            expected
        );

        let a_plus_b_plus_c = workspace_store.add_parsed_hypothesis("a+(b+c)").unwrap();
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
                .get_valid_transformations_from(a_plus_b_plus_c, None, None)
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
        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace.clone());
        workspace_store
            .add_parsed_transformation(false, "a+b", "b+a")
            .unwrap();

        workspace_store.add_parsed_hypothesis("x+y").unwrap();
        let expected = vec![workspace_store.compile().parse_from_string("y+x").unwrap()];
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations("y+x", None, None)
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
                .get_valid_transformations("k+j", None, None)
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
                .get_valid_transformations("(b+c)+a", None, None)
                .unwrap(),
            vec![expected]
        );

        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
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
                .get_valid_transformations("a+(c+b)", None, None)
                .unwrap(),
            vec![expected]
        );

        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
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
                .get_valid_transformations("r^s", None, None)
                .unwrap(),
            vec![expected]
        );

        let expected = workspace_store.compile().parse_from_string("s^r").unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .get_valid_transformations("s^r", None, None)
                .unwrap(),
            vec![expected]
        );

        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
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
                .get_valid_transformations("p^q=q^q", None, None)
                .unwrap(),
            vec![expected]
        );

        let workspace = Workspace::initialize(types.clone(), vec![], interpretations.clone());
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
                .get_valid_transformations("r^s=s^s", None, None)
                .unwrap(),
            vec![expected]
        );
    }

    #[test]
    fn test_workspace_adds_hypotheses() {
        let types = TypeHierarchy::new();
        let mut workspace = Workspace::initialize(types, vec![], vec![]);
        let statement = SymbolNode::leaf_object("a");
        workspace.add_hypothesis(statement).unwrap();
        assert_eq!(workspace.get_ordered_statements().len(), 1);

        let statement = SymbolNode::leaf_object("b");
        workspace.add_hypothesis(statement).unwrap();

        assert_eq!(workspace.get_ordered_statements().len(), 2);
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
        let workspace = Workspace::initialize(
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
        assert_eq!(
            workspace_store.compile().get_ordered_statements(),
            vec![expected]
        );
    }

    #[test]
    fn test_workspace_adds_and_executes_algorithms() {
        let mut types = TypeHierarchy::chain(vec!["Real".into(), "+".into()]).unwrap();
        types
            .add_child_to_parent("*".into(), "Real".into())
            .unwrap();
        types
            .add_child_to_parent("/".into(), "Real".into())
            .unwrap();
        let plus_interpretation = Interpretation::infix_operator("+".into(), 1, "+".into());
        let multiply_interpretation = Interpretation::infix_operator("*".into(), 2, "*".into());
        let divides_interpretation = Interpretation::infix_operator("/".into(), 2, "/".into());
        let real_interpretation = Interpretation::generated_type(GeneratedTypeCondition::IsNumeric);
        let real_generated_type = GeneratedType::new(
            GeneratedTypeCondition::IsNumeric,
            vec!["Real".into()].into_iter().collect(),
        );
        let generated_types = vec![real_generated_type];
        let interpretations = vec![
            plus_interpretation,
            multiply_interpretation,
            divides_interpretation,
            real_interpretation,
        ];
        let workspace = Workspace::initialize(types, generated_types, interpretations);

        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);

        assert!(workspace_store.add_parsed_hypothesis("2+2").is_ok());
        assert_eq!(workspace_store.compile().get_ordered_statements().len(), 1);
        let two = SymbolNode::leaf(Symbol::new("2".to_string(), "2".into()));
        assert_eq!(
            workspace_store.compile().get_ordered_statements(),
            vec![SymbolNode::new(
                Symbol::new_with_same_type_as_value("+").into(),
                vec![two.clone(), two.clone()]
            )]
        );

        workspace_store
            .add_algorithm(&AlgorithmType::Addition, "+", "Real")
            .unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .transformation_lattice
                .get_available_transformations()
                .len(),
            1
        );

        let _transformed = workspace_store
            .try_transform_into_parsed("4", None, None)
            .unwrap();
        assert_eq!(workspace_store.compile().get_ordered_statements().len(), 2);
        let four = SymbolNode::leaf(Symbol::new("4".to_string(), "4".into()));
        assert_eq!(workspace_store.compile().get_ordered_statements()[1], four);

        let hypothesis_result = workspace_store.add_parsed_hypothesis("8/4");
        assert!(hypothesis_result.is_ok(), "{:?}", hypothesis_result);
        assert_eq!(workspace_store.compile().get_ordered_statements().len(), 3);
        workspace_store
            .add_algorithm(&AlgorithmType::Division, "/", "Real")
            .unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .transformation_lattice
                .get_available_transformations()
                .len(),
            2
        );

        let _transformed = workspace_store
            .try_transform_into_parsed("2", None, None)
            .unwrap();
        assert_eq!(workspace_store.compile().get_ordered_statements().len(), 4);
        assert!(workspace_store.compile().contains_statement(&two));

        assert!(workspace_store.add_parsed_hypothesis("3*0").is_ok());
        assert_eq!(workspace_store.compile().get_ordered_statements().len(), 5);
        workspace_store
            .add_algorithm(&AlgorithmType::Multiplication, "*", "Real")
            .unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .transformation_lattice
                .get_available_transformations()
                .len(),
            3
        );

        let _transformed = workspace_store
            .try_transform_into_parsed("0", None, None)
            .unwrap();
        let zero = SymbolNode::leaf(Symbol::new("0".to_string(), "0".into()));
        assert_eq!(workspace_store.compile().get_ordered_statements().len(), 6);
        assert!(workspace_store.compile().contains_statement(&zero));
    }

    #[test]
    fn test_workspace_transforms_statement() {
        let types = TypeHierarchy::new();
        let workspace = Workspace::initialize(types, vec![], vec![]);
        let mut workspace_store = WorkspaceTransactionStore::snapshot(workspace);
        assert!(workspace_store.add_parsed_hypothesis("a").is_ok());
        assert_eq!(workspace_store.compile().get_ordered_statements().len(), 1);

        workspace_store
            .add_parsed_transformation(false, "a", "b")
            .unwrap();
        assert_eq!(
            workspace_store
                .compile()
                .transformation_lattice
                .get_available_transformations()
                .len(),
            1
        );

        let _transformed = workspace_store
            .try_transform_into_parsed("b", None, None)
            .unwrap();
        assert_eq!(workspace_store.compile().get_ordered_statements().len(), 2);
        assert_eq!(
            workspace_store.compile().get_ordered_statements(),
            vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")]
        );
    }

    #[test]
    fn test_workspace_imports_context() {
        let types = TypeHierarchy::new();
        let mut workspace = Workspace::initialize(types, vec![], vec![]);

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
            TransformationLattice::from_transformations(
                vec![
                    equality_reflexivity.clone().into(),
                    equality_symmetry.clone().into(),
                ]
                .into_iter()
                .map(|t| AvailableTransformation::Axiom(t))
                .collect(),
            )
            .unwrap(),
        )
        .unwrap();

        assert_eq!(workspace.try_import_context(context.clone()), Ok(()));
        assert_eq!(workspace.types, types);
        assert_eq!(
            workspace
                .transformation_lattice
                .get_available_transformations()
                .len(),
            2
        );

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
            TransformationLattice::from_transformations(
                vec![equality_reflexivity.into(), equality_symmetry.into()]
                    .into_iter()
                    .map(|t| AvailableTransformation::Axiom(t))
                    .collect(),
            )
            .unwrap(),
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

        assert_eq!(
            workspace
                .transformation_lattice
                .get_available_transformations()
                .len(),
            2
        );

        let inverted_types = TypeHierarchy::chain(vec!["Rational".into(), "Real".into()]).unwrap();
        let conflicting_context = Context::new(
            inverted_types,
            vec![],
            vec![],
            TransformationLattice::empty(),
        )
        .unwrap();

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
