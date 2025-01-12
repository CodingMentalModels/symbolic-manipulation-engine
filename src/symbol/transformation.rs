use log::{debug, trace, warn};
use std::collections::{HashMap, HashSet};
use std::isize;

use crate::constants::MAX_ADDITIONAL_VALID_TRANSFORMATION_DEPTH;
use crate::custom_serde::{deserialize_vector_as_hashmap, serialize_hashmap_as_vector};
use crate::parsing::interpretation::Interpretation;
use crate::symbol::symbol_node::{Symbol, SymbolNode, SymbolNodeError};
use crate::symbol::symbol_type::{Type, TypeError};
use serde::{Deserialize, Serialize};

use super::algorithm::AlgorithmType;
use super::symbol_node::{SymbolName, SymbolNodeAddress, SymbolNodeRoot};
use super::symbol_type::{GeneratedType, TypeHierarchy};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformationLattice {
    statements: HashSet<SymbolNode>,
    available_transformations: HashSet<AvailableTransformation>,
    #[serde(
        serialize_with = "serialize_hashmap_as_vector",
        deserialize_with = "deserialize_vector_as_hashmap"
    )]
    transformations_from: HashMap<SymbolNode, Vec<Transformation>>,
    #[serde(
        serialize_with = "serialize_hashmap_as_vector",
        deserialize_with = "deserialize_vector_as_hashmap"
    )]
    transformations_to: HashMap<(SymbolNode, Transformation), SymbolNode>,
}

impl TransformationLattice {
    pub fn new(
        statements: HashSet<SymbolNode>,
        available_transformations: HashSet<AvailableTransformation>,
        transformations_from: HashMap<SymbolNode, Vec<Transformation>>,
        transformations_to: HashMap<(SymbolNode, Transformation), SymbolNode>,
    ) -> Self {
        Self {
            statements,
            available_transformations,
            transformations_from,
            transformations_to,
        }
    }

    pub fn empty() -> Self {
        Self::new(
            HashSet::new(),
            HashSet::new(),
            HashMap::new(),
            HashMap::new(),
        )
    }

    pub fn from_transformations(
        transformations: HashSet<AvailableTransformation>,
    ) -> Result<Self, TransformationError> {
        let mut to_return = Self::empty();
        to_return.add_available_transformations(transformations)?;
        Ok(to_return)
    }

    pub fn copy_with_only_algorithms(&self) -> Self {
        Self::new(
            HashSet::new(),
            self.available_transformations
                .iter()
                .filter(|t| t.is_algorithm())
                .cloned()
                .collect(),
            HashMap::new(),
            HashMap::new(),
        )
    }

    pub fn scope_down(
        &self,
        maybe_scoped_statements: Option<HashSet<SymbolNode>>,
        maybe_scoped_transformations: Option<HashSet<AvailableTransformation>>,
    ) -> Result<Self, TransformationError> {
        let scoped_statements = match maybe_scoped_statements {
            None => self.statements.clone(),
            Some(statements) => self.verify_statements(statements)?,
        };
        let scoped_available_transformations = match maybe_scoped_transformations {
            None => self.available_transformations.clone(),
            Some(available_transformations) => {
                self.verify_available_transformations(available_transformations)?
            }
        };
        Ok(Self::new(
            scoped_statements,
            scoped_available_transformations,
            HashMap::new(),
            HashMap::new(),
        ))
    }

    pub fn verify_statements(
        &self,
        to_verify: HashSet<SymbolNode>,
    ) -> Result<HashSet<SymbolNode>, TransformationError> {
        let missing_statements: Vec<&SymbolNode> = to_verify
            .iter()
            .filter(|s| !self.statements.contains(s))
            .collect();
        if missing_statements.len() == 0 {
            Ok(to_verify.clone())
        } else {
            Err(
                TransformationError::MissingStatementsInTransformationLattice(
                    missing_statements.into_iter().cloned().collect(),
                ),
            )
        }
    }

    pub fn get_statements(&self) -> &HashSet<SymbolNode> {
        &self.statements
    }

    pub fn get_ordered_statements(&self) -> Vec<SymbolNode> {
        let mut to_return = self
            .get_statements()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        to_return.sort();
        to_return
    }

    pub fn get_available_transformations(&self) -> &HashSet<AvailableTransformation> {
        &self.available_transformations
    }

    pub fn verify_available_transformations(
        &self,
        to_verify: HashSet<AvailableTransformation>,
    ) -> Result<HashSet<AvailableTransformation>, TransformationError> {
        let missing_transformation: Vec<&AvailableTransformation> = to_verify
            .iter()
            .filter(|t| !self.available_transformations.contains(t))
            .collect();
        if missing_transformation.len() == 0 {
            Ok(to_verify.clone())
        } else {
            Err(
                TransformationError::MissingTransformationsInTransformationLattice(
                    missing_transformation
                        .into_iter()
                        .map(|t| t.get_transformation())
                        .cloned()
                        .collect(),
                ),
            )
        }
    }

    pub fn get_ordered_available_transformations(&self) -> Vec<AvailableTransformation> {
        let mut to_return = self
            .get_available_transformations()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        to_return.sort();
        to_return
    }

    pub fn get_transformations_from(&self, statement: &SymbolNode) -> Vec<&Transformation> {
        self.transformations_from
            .get(statement)
            .map(|v| v.iter().collect())
            .unwrap_or_else(Vec::new)
    }

    pub fn get_transformations_to(&self, statement: &SymbolNode) -> Vec<&Transformation> {
        self.transformations_to
            .iter()
            .filter(|((_, _), to)| *to == statement)
            .map(|((_, transformation), _)| transformation)
            .collect()
    }

    pub fn get_upstream_statement_and_transformation(
        &self,
        statement: &SymbolNode,
    ) -> Option<(SymbolNode, Transformation)> {
        let statements_and_transformations: Vec<_> = self
            .transformations_to
            .iter()
            .filter(|((_, _), to)| *to == statement)
            .map(|((from, transformation), _)| (from.clone(), transformation.clone()))
            .collect();
        if statements_and_transformations.len() == 0 {
            return None;
        }

        let first_statement_and_transformation = statements_and_transformations[0].clone();

        return Some(first_statement_and_transformation);
    }

    pub fn get_ordered_applied_transformations(
        &self,
    ) -> Vec<(SymbolNode, Transformation, SymbolNode)> {
        let mut mapped = self
            .transformations_to
            .clone()
            .into_iter()
            .map(|((from, transformation), to)| (from, transformation, to))
            .collect::<Vec<_>>();
        mapped.sort();
        mapped
    }

    pub fn contains_statement(&self, statement: &SymbolNode) -> bool {
        self.statements.contains(statement)
    }

    pub fn contains_available_transformation(
        &self,
        transformation: &AvailableTransformation,
    ) -> bool {
        self.available_transformations.contains(transformation)
    }

    pub fn contains_transformation(&self, transformation: &Transformation) -> bool {
        self.available_transformations
            .iter()
            .any(|t| t.get_transformation() == transformation)
    }

    pub fn add_hypothesis(&mut self, hypothesis: SymbolNode) -> Result<(), TransformationError> {
        if self.contains_statement(&hypothesis) {
            Err(TransformationError::AlreadyContainsStatement(
                hypothesis.clone(),
            ))
        } else {
            self.statements.insert(hypothesis);
            Ok(())
        }
    }

    pub fn add_available_transformations(
        &mut self,
        transformations: HashSet<AvailableTransformation>,
    ) -> Result<(), TransformationError> {
        transformations
            .into_iter()
            .try_for_each(|t| self.add_available_transformation(t))
    }

    pub fn add_available_transformation(
        &mut self,
        transformation: AvailableTransformation,
    ) -> Result<(), TransformationError> {
        if self.contains_available_transformation(&transformation) {
            Err(TransformationError::AlreadyContainsTransformation(
                transformation.get_transformation().clone(),
            ))
        } else {
            self.available_transformations.insert(transformation);
            Ok(())
        }
    }

    pub fn remove_transformation_and_all_dependents(
        &mut self,
        transformation: &Transformation,
    ) -> Result<(HashSet<Transformation>, HashSet<SymbolNode>), TransformationError> {
        let (transformations_to_remove, statements_to_remove) =
            self.get_all_dependent_transformations_and_statements(transformation)?;
        self.force_remove_transformations(&transformations_to_remove);
        self.force_remove_statements(&statements_to_remove);
        Ok((transformations_to_remove, statements_to_remove))
    }

    fn get_all_dependent_transformations_and_statements(
        &self,
        transformation: &Transformation,
    ) -> Result<(HashSet<Transformation>, HashSet<SymbolNode>), TransformationError> {
        trace!(
            "get_all_dependent_transformations_and_statements: {}",
            transformation.to_symbol_string()
        );
        let dependent_theorems = self.get_all_dependent_theorems(transformation)?;
        trace!(
            "downstream_transformations: {}",
            dependent_theorems
                .iter()
                .map(|s| s.to_symbol_string())
                .collect::<Vec<_>>()
                .join("\n")
        );
        let mut theorems_to_return = dependent_theorems;
        theorems_to_return.insert(transformation.clone());
        let downstream_statement_results = theorems_to_return
            .iter()
            .map(|t| self.get_statements_derived_from(t))
            .collect::<Vec<_>>();
        if downstream_statement_results.iter().any(|r| r.is_err()) {
            return Err(downstream_statement_results
                .into_iter()
                .filter(|r| r.is_err())
                .map(|r| r.unwrap_err())
                .next()
                .unwrap());
        }
        let downstream_statements = downstream_statement_results
            .into_iter()
            .map(|r| r.unwrap())
            .flatten()
            .collect();

        let to_return = (theorems_to_return, downstream_statements);

        Ok(to_return)
    }

    fn get_all_dependent_theorems(
        &self,
        transformation: &Transformation,
    ) -> Result<HashSet<Transformation>, TransformationError> {
        let downstream_theorems = self.get_downstream_theorems(transformation)?;

        let dependent_theorems: Vec<_> = downstream_theorems
            .iter()
            .map(|t| self.get_all_dependent_theorems(t))
            .collect();
        if dependent_theorems.iter().any(|r| r.is_err()) {
            return Err(dependent_theorems
                .into_iter()
                .filter(|r| r.is_err())
                .next()
                .expect("We just checked that there is an error.")
                .unwrap_err());
        }
        let mut to_return: HashSet<_> = dependent_theorems
            .into_iter()
            .map(|r| r.expect("We just checked that there are no errors."))
            .flatten()
            .collect();
        to_return.insert(transformation.clone());
        Ok(to_return)
    }

    fn get_downstream_theorems(
        &self,
        transformation: &Transformation,
    ) -> Result<HashSet<Transformation>, TransformationError> {
        if !self.contains_transformation(transformation) {
            return Err(
                TransformationError::MissingTransformationsInTransformationLattice(vec![
                    transformation.clone(),
                ]),
            );
        }
        debug!(
            "get_downstream_theorems:\ntransformation: {}\navailable_transformations:\n{:#?}",
            transformation.to_symbol_string(),
            self.available_transformations
                .iter()
                .map(|t| t.to_symbol_string())
                .collect::<Vec<_>>()
                .join("\n")
        );
        Ok(self
            .available_transformations
            .iter()
            .filter(|t| match t {
                AvailableTransformation::Theorem((theorem, provenance)) => {
                    provenance.contains_transformation(transformation)
                }
                _ => false,
            })
            .map(|t| t.get_transformation())
            .cloned()
            .collect())
    }

    pub fn get_statements_derived_from(
        &self,
        transformation: &Transformation,
    ) -> Result<HashSet<SymbolNode>, TransformationError> {
        let immediate_descendents: Vec<_> = self
            .transformations_to
            .iter()
            .filter(|((_from, t), _to)| t == transformation)
            .map(|((_from, _t), to)| to)
            .cloned()
            .collect();
        let mut to_return = HashSet::new();
        for descendent in immediate_descendents {
            let further_downstream = self.get_all_dependent_statements(&descendent)?;
            to_return = to_return.union(&further_downstream).cloned().collect();
        }
        Ok(to_return)
    }

    pub fn force_remove_transformations(&mut self, transformations: &HashSet<Transformation>) {
        transformations
            .iter()
            .for_each(|s| self.force_remove_transformation(s))
    }

    fn force_remove_transformation(&mut self, transformation: &Transformation) {
        self.remove_available_transformation_by_transformation(transformation);
        self.transformations_from = self
            .transformations_from
            .iter()
            .map(|(s, t)| {
                (
                    s.clone(),
                    t.clone().into_iter().filter(|t| t != t).collect::<Vec<_>>(),
                )
            })
            .filter(|(_, t)| t.len() > 0)
            .collect();
        self.transformations_to = self
            .transformations_to
            .iter()
            .filter(|((_s, t), _to)| t != transformation)
            .map(|((s, t), to)| ((s.clone(), t.clone()), to.clone()))
            .filter(|((_, _), to)| to.len() > 0)
            .collect();
    }

    fn remove_available_transformation_by_transformation(
        &mut self,
        transformation: &Transformation,
    ) {
        self.available_transformations = self
            .available_transformations
            .clone()
            .into_iter()
            .filter(|t| t.get_transformation() != transformation)
            .collect();
    }

    pub fn remove_statement_and_all_dependents(
        &mut self,
        statement: &SymbolNode,
    ) -> Result<HashSet<SymbolNode>, TransformationError> {
        let statements_to_remove = self.get_all_dependent_statements(statement)?;
        self.force_remove_statements(&statements_to_remove);
        Ok(statements_to_remove)
    }

    fn get_all_dependent_statements(
        &self,
        statement: &SymbolNode,
    ) -> Result<HashSet<SymbolNode>, TransformationError> {
        trace!(
            "get_all_dependent_statements: {}",
            statement.to_symbol_string()
        );
        let downstream = self.get_downstream_statements(statement)?;
        trace!(
            "downstream: {}",
            downstream
                .iter()
                .map(|s| s.to_symbol_string())
                .collect::<Vec<_>>()
                .join("\n")
        );
        let mut to_return = downstream.clone();

        // N.B. get_downstream_statements would have errored already if statement was not in the
        // lattice
        to_return.insert(statement.clone());
        let further_downstream = downstream
            .into_iter()
            .map(|s| self.get_all_dependent_statements(&s))
            .collect::<Vec<_>>();
        if further_downstream.iter().any(|s| s.is_err()) {
            return Err(further_downstream
                .into_iter()
                .filter(|s| s.is_err())
                .map(|e| e.unwrap_err())
                .next()
                .unwrap());
        }
        let further_downstream: HashSet<SymbolNode> = further_downstream
            .into_iter()
            .map(|s| s.unwrap())
            .flatten()
            .collect();
        let to_return = to_return
            .union(&further_downstream)
            .cloned()
            .collect::<HashSet<_>>();
        trace!(
            "get_all_dependent_statements returning:\n{}",
            to_return
                .iter()
                .map(|s| s.to_symbol_string())
                .collect::<Vec<_>>()
                .join("\n")
        );
        Ok(to_return)
    }

    fn get_downstream_statements(
        &self,
        statement: &SymbolNode,
    ) -> Result<HashSet<SymbolNode>, TransformationError> {
        if !self.contains_statement(statement) {
            return Err(
                TransformationError::MissingStatementsInTransformationLattice(vec![
                    statement.clone()
                ]),
            );
        }
        debug!(
            "get_downstream_statements:\nstatement: {}\ntransformations_to:\n{:#?}",
            statement.to_symbol_string(),
            self.transformations_to
                .iter()
                .map(|((from, t), to)| format!(
                    "\nfrom: {:?}\nt: {:?}\nto: {:?}",
                    from.to_symbol_string(),
                    t.to_symbol_string(),
                    to.to_symbol_string()
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );
        Ok(self
            .transformations_to
            .iter()
            .filter(|((from, _), _)| from == statement || from.is_join_containing(statement))
            .map(|((_, _), to)| to)
            .cloned()
            .collect())
    }

    pub fn force_remove_statements(&mut self, statements: &HashSet<SymbolNode>) {
        statements
            .iter()
            .for_each(|s| self.force_remove_statement(s))
    }

    fn force_remove_statement(&mut self, statement: &SymbolNode) {
        self.statements.remove(statement);
        self.transformations_from = self
            .transformations_from
            .iter()
            .filter(|(s, _t)| s != &statement)
            .map(|(s, t)| (s.clone(), t.clone()))
            .collect();
        self.transformations_to = self
            .transformations_to
            .iter()
            .filter(|((s, _t), to)| s != statement && to != &statement)
            .map(|((s, t), to)| ((s.clone(), t.clone()), to.clone()))
            .collect();
    }

    pub fn try_transform_into(
        &mut self,
        types: &TypeHierarchy,
        desired: SymbolNode,
    ) -> Result<(SymbolNode, AvailableTransformation, SymbolNode), TransformationError> {
        if self.contains_statement(&desired) {
            return Err(TransformationError::AlreadyContainsStatement(
                desired.clone(),
            ));
        }
        if desired.contains_arbitrary_nodes() {
            return Err(TransformationError::StatementContainsArbitraryNode(
                desired.clone(),
            ));
        }
        let instantiated_transformations =
            self.get_instantiated_transformations_with_arbitrary(types, Some(desired.clone()))?;

        debug!(
            "instantiated_transformations:\n{}",
            instantiated_transformations
                .clone()
                .into_iter()
                .map(|(instantiated, arbitrary)| format!(
                    "{} ({})",
                    instantiated.get_transformation().to_symbol_string(),
                    arbitrary.get_transformation().to_symbol_string()
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );
        for (instantiated_transform, arbitrary_transform) in instantiated_transformations {
            let statements =
                self.get_candidate_statements(instantiated_transform.get_transformation());
            for statement in statements.into_iter() {
                match instantiated_transform
                    .get_transformation()
                    .try_transform_into(types, &statement, &desired)
                {
                    Ok(output) => {
                        // TODO Log the instantiated transform too and show it in the front end
                        self.force_apply_transformation(
                            statement.clone(),
                            arbitrary_transform.clone(),
                            output.clone(),
                        );
                        return Ok((statement, arbitrary_transform.clone(), output));
                    }
                    Err(_) => {
                        // Do nothing, keep trying transformations
                    }
                }
            }
        }
        return Err(TransformationError::NoValidTransformationsPossible);
    }

    fn get_candidate_statements(&mut self, transform: &Transformation) -> HashSet<SymbolNode> {
        let statements = if transform.is_joint_transform() {
            self.get_statement_pairs()
        } else {
            self.get_statements().clone()
        };
        statements
    }

    pub fn force_apply_transformation(
        &mut self,
        from: SymbolNode,
        transformation: AvailableTransformation,
        to: SymbolNode,
    ) {
        self.statements.insert(to.clone());
        if !self.contains_available_transformation(&transformation) {
            warn!(
                "force_apply_transformation was used without the transformation being available."
            );
            self.available_transformations
                .insert(transformation.clone());
            // TODO Remove the assertion and consider whether we need to catch this upstream
            assert!(false);
        }
        self.transformations_from
            .entry(from.clone())
            .or_default()
            .push(transformation.get_transformation().clone());
        self.transformations_to
            .insert((from, transformation.get_transformation().clone()), to);
    }

    pub fn force_derive_theorem(
        &mut self,
        transformation: Transformation,
        provenance: TransformationProvenance,
    ) {
        self.available_transformations
            .insert(AvailableTransformation::Theorem((
                transformation,
                provenance,
            )));
    }

    pub fn get_instantiated_transformations_with_arbitrary(
        &self,
        types: &TypeHierarchy,
        maybe_desired: Option<SymbolNode>,
    ) -> Result<Vec<(AvailableTransformation, AvailableTransformation)>, TransformationError> {
        let statements = if let Some(desired) = maybe_desired {
            let mut statements = vec![desired];
            statements.append(&mut self.get_ordered_statements().clone());
            statements
        } else {
            self.get_ordered_statements().clone()
        };
        let substatements = statements
            .iter()
            .map(|statement| statement.get_substatements())
            .flatten()
            .collect::<HashSet<_>>();

        debug!(
            "substatements:\n{}",
            substatements
                .clone()
                .into_iter()
                .map(|s| s.to_symbol_string())
                .collect::<Vec<_>>()
                .join("\n")
        );
        let mut to_return = Vec::new();
        for arbitrary_transform in self.get_arbitrary_transformations() {
            to_return.append(
                &mut arbitrary_transform
                    .instantiate_arbitrary_nodes(types, &substatements)?
                    .into_iter()
                    .map(|instantiated| (instantiated, arbitrary_transform.clone()))
                    .collect(),
            );
        }

        to_return.append(
            &mut self
                .get_non_arbitrary_transformations()
                .into_iter()
                .map(|t| (t.clone(), t)) // Instantiated == Arbitrary for non-arbitrary
                .collect(),
        );

        return Ok(to_return);
    }

    fn get_arbitrary_transformations(&self) -> HashSet<AvailableTransformation> {
        self.get_transformations_filtered_on_arbitrariness(true)
    }

    fn get_non_arbitrary_transformations(&self) -> HashSet<AvailableTransformation> {
        self.get_transformations_filtered_on_arbitrariness(false)
    }

    fn get_transformations_filtered_on_arbitrariness(
        &self,
        is_arbitrary: bool,
    ) -> HashSet<AvailableTransformation> {
        self.get_available_transformations()
            .iter()
            .filter(|t| t.get_transformation().contains_arbitrary_nodes() == is_arbitrary)
            .cloned()
            .collect()
    }

    pub fn get_statement_pairs(&self) -> HashSet<SymbolNode> {
        self.get_statement_pairs_including(None)
    }

    pub fn get_statement_pairs_including(
        &self,
        statement: Option<&SymbolNode>,
    ) -> HashSet<SymbolNode> {
        // Returns pairs of statements for use in joint transforms
        // Returns both orders as well as duplicates since those are valid joint transform args
        let left = self.get_ordered_statements();
        let right = self.get_ordered_statements();
        let mut to_return = HashSet::new();
        for i in 0..left.len() {
            for j in 0..right.len() {
                if statement.is_none()
                    || (Some(&left[i]) == statement)
                    || (Some(&right[i]) == statement)
                {
                    to_return.insert(left[i].clone().join(right[j].clone()));
                }
            }
        }
        to_return
    }

    pub fn derive_theorem(
        &mut self,
        conclusion: &SymbolNode,
    ) -> Result<(Transformation, TransformationProvenance), TransformationError> {
        let (hypotheses, steps) = self.get_ancestor_hypotheses_and_steps(conclusion)?;

        let theorem = if hypotheses.len() == 1 {
            let hypothesis = hypotheses
                .iter()
                .next()
                .expect("There is guaranteed to be one.");
            Transformation::ExplicitTransformation((hypothesis.clone(), conclusion.clone()).into())
        } else if hypotheses.len() == 2 {
            let mut two_hypotheses = hypotheses.iter().take(2).collect::<Vec<_>>();
            two_hypotheses.sort();
            Transformation::ExplicitTransformation(
                (
                    two_hypotheses[0].clone().join(two_hypotheses[1].clone()),
                    conclusion.clone(),
                )
                    .into(),
            )
        } else {
            return Err(TransformationError::MoreThanTwoHypothesesForTheorem(
                hypotheses.into_iter().collect(),
                conclusion.clone(),
            ));
        };

        if self.contains_transformation(&theorem) {
            return Err(TransformationError::AlreadyContainsTransformation(
                theorem.clone(),
            ));
        }
        let mut hypotheses: Vec<_> = hypotheses.into_iter().collect();
        hypotheses.sort();
        let provenance = TransformationProvenance::new(hypotheses, conclusion.clone(), steps);
        self.add_available_transformation(AvailableTransformation::Theorem((
            theorem.clone(),
            provenance.clone(),
        )))?;
        Ok((theorem, provenance))
    }

    fn get_ancestor_hypotheses_and_steps(
        &self,
        conclusion: &SymbolNode,
    ) -> Result<
        (
            HashSet<SymbolNode>,
            Vec<(SymbolNode, Transformation, SymbolNode)>,
        ),
        TransformationError,
    > {
        if !self.contains_statement(conclusion) {
            return Err(
                TransformationError::MissingStatementsInTransformationLattice(vec![
                    conclusion.clone()
                ]),
            );
        }
        match self.get_upstream_statement_and_transformation(conclusion) {
            None => Ok((vec![conclusion.clone()].into_iter().collect(), Vec::new())),
            Some((parent, transformation)) => {
                if parent.is_join() {
                    let children = parent.get_children();
                    assert_eq!(children.len(), 2, "Joins must have two children.");
                    let (left, right) = (children[0].clone(), children[1].clone());
                    let (left_hypotheses, mut left_steps): (HashSet<SymbolNode>, Vec<_>) =
                        self.get_ancestor_hypotheses_and_steps(&left)?;
                    let (right_hypotheses, right_steps): (HashSet<SymbolNode>, Vec<_>) =
                        self.get_ancestor_hypotheses_and_steps(&right)?;
                    let hypotheses = left_hypotheses.union(&right_hypotheses).cloned().collect();
                    let right_steps_deduplicated = right_steps
                        .into_iter()
                        .filter(|step| !left_steps.contains(&step))
                        .collect::<Vec<_>>();
                    left_steps.extend(right_steps_deduplicated);
                    left_steps.push((parent, transformation, conclusion.clone()));
                    Ok((hypotheses, left_steps))
                } else {
                    let (ancestor_hypotheses, mut steps) =
                        self.get_ancestor_hypotheses_and_steps(&parent)?;
                    steps.push((parent, transformation, conclusion.clone()));
                    Ok((ancestor_hypotheses, steps))
                }
            }
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AvailableTransformation {
    Axiom(Transformation),
    Theorem((Transformation, TransformationProvenance)),
}

impl AvailableTransformation {
    pub fn get_transformation(&self) -> &Transformation {
        match self {
            Self::Axiom(t) => t,
            Self::Theorem((t, _)) => t,
        }
    }

    pub fn to_symbol_string(&self) -> String {
        match self {
            Self::Axiom(t) => format!("{} (Axiom)", t.to_symbol_string()),
            Self::Theorem((t, _)) => format!("{} (Theorem)", t.to_symbol_string()),
        }
    }

    pub fn is_algorithm(&self) -> bool {
        self.get_transformation().is_algorithm()
    }

    pub fn instantiate_arbitrary_nodes(
        &self,
        hierarchy: &TypeHierarchy,
        substatements: &HashSet<SymbolNode>,
    ) -> Result<HashSet<Self>, TransformationError> {
        match self {
            Self::Axiom(t) => Ok(t
                .instantiate_arbitrary_nodes(hierarchy, substatements)?
                .into_iter()
                .map(|r| Self::Axiom(r))
                .collect()),
            Self::Theorem((t, provenance)) => Ok(t
                .instantiate_arbitrary_nodes(hierarchy, substatements)?
                .into_iter()
                .map(|r| Self::Theorem((r, provenance.clone())))
                .collect()),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TransformationProvenance {
    pub hypotheses: Vec<SymbolNode>,
    pub conclusion: SymbolNode,
    pub steps: Vec<(SymbolNode, Transformation, SymbolNode)>,
}

impl TransformationProvenance {
    pub fn new(
        hypotheses: Vec<SymbolNode>,
        conclusion: SymbolNode,
        steps: Vec<(SymbolNode, Transformation, SymbolNode)>,
    ) -> Self {
        Self {
            hypotheses,
            conclusion,
            steps,
        }
    }

    pub fn contains_transformation(&self, transformation: &Transformation) -> bool {
        self.steps.iter().any(|(_, t, _)| t == transformation)
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Transformation {
    ExplicitTransformation(ExplicitTransformation),
    AlgorithmTransformation(AlgorithmTransformation),
    ApplyToBothSidesTransformation(ApplyToBothSidesTransformation),
}

impl From<ExplicitTransformation> for Transformation {
    fn from(t: ExplicitTransformation) -> Self {
        Self::ExplicitTransformation(t)
    }
}

impl From<AlgorithmTransformation> for Transformation {
    fn from(algorithm: AlgorithmTransformation) -> Self {
        Self::AlgorithmTransformation(algorithm)
    }
}

impl From<ApplyToBothSidesTransformation> for Transformation {
    fn from(t: ApplyToBothSidesTransformation) -> Self {
        Self::ApplyToBothSidesTransformation(t)
    }
}

impl Transformation {
    pub fn to_symbol_string(&self) -> String {
        match self {
            Self::ExplicitTransformation(t) => t.to_symbol_string(),
            Self::AlgorithmTransformation(t) => t.to_symbol_string(),
            Self::ApplyToBothSidesTransformation(t) => t.to_symbol_string(),
        }
    }

    pub fn might_produce_type(&self, hierarchy: &TypeHierarchy, t: &Type) -> bool {
        let result = match self {
            Self::ExplicitTransformation(transformation) => {
                transformation.might_produce_type(hierarchy, t)
            }
            Self::AlgorithmTransformation(transformation) => {
                transformation.might_produce_type(hierarchy, t)
            }
            Self::ApplyToBothSidesTransformation(transformation) => {
                transformation.might_produce_type(hierarchy, t)
            }
        };
        trace!(
            "might_produce_type:\ntransformation: {:#?}\ntype: {:#?}\nresult: {}",
            self,
            t,
            result
        );
        result
    }

    pub fn transform(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        match self {
            Self::ExplicitTransformation(t) => t.typed_relabel_and_transform(hierarchy, statement),
            Self::AlgorithmTransformation(t) => t.transform(hierarchy, statement),
            Self::ApplyToBothSidesTransformation(t) => t.transform(hierarchy, statement),
        }
    }

    pub fn is_joint_transform(&self) -> bool {
        if let Self::ExplicitTransformation(t) = self {
            t.from.is_join()
        } else {
            false
        }
    }

    pub fn joint_transform(
        &self,
        hierarchy: &TypeHierarchy,
        left: &SymbolNode,
        right: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        let statement = left.clone().join(right.clone());
        let in_order = self.transform(hierarchy, &statement);
        if in_order.is_ok() {
            return in_order;
        }

        let reversed = right.clone().join(left.clone());
        self.transform(hierarchy, &reversed)
    }

    pub fn is_algorithm(&self) -> bool {
        match self {
            Transformation::AlgorithmTransformation(_) => true,
            _ => false,
        }
    }

    pub fn try_transform_into(
        &self,
        hierarchy: &TypeHierarchy,
        from: &SymbolNode,
        to: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        if from.contains_arbitrary_nodes() {
            return Err(TransformationError::StatementContainsArbitraryNode(
                from.clone(),
            ));
        }
        if to.contains_arbitrary_nodes() {
            return Err(TransformationError::StatementContainsArbitraryNode(
                to.clone(),
            ));
        }
        let valid_transformations = self.get_valid_transformations(hierarchy, from, Some(to));
        if valid_transformations.contains(to) {
            Ok(to.clone())
        } else {
            Err(TransformationError::NoValidTransformationsPossible)
        }
    }

    pub fn get_valid_transformations(
        &self,
        hierarchy: &TypeHierarchy,
        from_statement: &SymbolNode,
        maybe_to_statement: Option<&SymbolNode>,
    ) -> HashSet<SymbolNode> {
        // Getting valid transformations can recurse indefinitely, so we use
        // MAX_ADDITIONAL_VALID_TRANSFORMATION_DEPTH to limit.
        // We also implement our own call stack to avoid stack overflows and make it easier to
        // inspect the contents
        debug!(
            "get_valid_transformations({}, {}) called on {}",
            from_statement.to_symbol_string(),
            maybe_to_statement.map_or("None".to_string(), |s| s.to_symbol_string()),
            self.to_symbol_string(),
        );

        // Optimize away this function if the transformation can't possibly work
        if self.cant_possibly_transform_into(hierarchy, from_statement, maybe_to_statement) {
            return vec![from_statement.clone()].into_iter().collect();
        }

        let mut call_stack = vec![(from_statement.clone(), true, false, 0)];
        let mut already_processed: HashSet<SymbolNode> = HashSet::new();
        let mut child_to_valid_transformations: HashMap<SymbolNode, HashSet<SymbolNode>> =
            HashMap::new();
        let mut to_return = HashSet::new();
        let max_depth = from_statement.get_depth() + MAX_ADDITIONAL_VALID_TRANSFORMATION_DEPTH;

        while let Some((current_statement, should_return, are_children_processed, depth)) =
            call_stack.pop()
        {
            trace!(
                "({}, {}, {}, {})",
                current_statement.to_symbol_string(),
                should_return,
                are_children_processed,
                depth,
            );
            if !are_children_processed {
                // Push the statement back onto the stack with children marked as processed
                // since we're about to process them
                call_stack.push((current_statement.clone(), should_return, true, depth));
                already_processed.insert(current_statement.clone());

                // Push the children on to be processed first and don't return them
                for child in current_statement.get_children() {
                    if already_processed.contains(&child) {
                        debug!("Child already processed!");
                    } else {
                        call_stack.push((child.clone(), false, false, depth + 1));
                    }
                }
            } else {
                let mut valid_roots = vec![current_statement.clone()]
                    .into_iter()
                    .collect::<HashSet<_>>();
                trace!(
                    "transform_at(hierarchy, {}, vec![])",
                    current_statement.to_symbol_string(),
                );
                match self.transform_at(hierarchy, &current_statement, vec![]) {
                    Ok(result) => {
                        trace!("transform_at passed: {}", result.to_symbol_string());
                        let final_max_depth = max_depth.saturating_sub(depth);

                        // Also push the transformed statement on so that it gets processed
                        valid_roots.insert(result.clone());

                        let to_statement_might_contain = |substatement: &SymbolNode| {
                            maybe_to_statement.map_or(true, |to_statement| {
                                to_statement.contains_substatement(substatement)
                            })
                        };

                        // Bail out if we've gotten too deep or the to_statement doesn't contain
                        // the result
                        if result.get_depth() > final_max_depth {
                            debug!("Max depth reached!");
                        } else if !to_statement_might_contain(&result) {
                            debug!("to_statement_might_contain = false");
                        } else {
                            // Log the transformation as a valid one
                            child_to_valid_transformations
                                .entry(current_statement.clone())
                                .and_modify(|value| {
                                    value.insert(result.clone());
                                })
                                .or_insert_with(|| {
                                    vec![result.clone()].into_iter().collect::<HashSet<_>>()
                                });

                            // Push the result onto the call stack for further processing
                            if already_processed.contains(&result) {
                                debug!("Already processed!");
                            } else {
                                call_stack.push((result.clone(), should_return, false, depth));
                            }
                        }
                    }
                    e => {
                        trace!("transform_at errored");
                        // trace!("{:?}", e);
                    }
                };

                for statement_to_apply_to in valid_roots {
                    let result = Self::apply_valid_transformations_to_children(
                        &child_to_valid_transformations,
                        &statement_to_apply_to,
                        None,
                    );
                    if should_return {
                        match maybe_to_statement {
                            None => {
                                to_return.extend(result.clone());
                            }
                            Some(to_statement) => {
                                if result.contains(to_statement) {
                                    debug!("Found result matching to_statement:\nfrom_statement: {}\ntransformation: {}\nresult: {}", from_statement.to_symbol_string(), self.to_symbol_string(), to_statement.to_symbol_string());
                                    return vec![to_statement.clone()].into_iter().collect();
                                }
                            }
                        }
                    }

                    // Log the child transformations as valid
                    child_to_valid_transformations
                        .entry(statement_to_apply_to.clone())
                        .and_modify(|value| {
                            value.extend(result.clone());
                        })
                        .or_insert_with(|| result.clone());
                }
            }
        }

        to_return
    }

    fn apply_valid_transformations_to_children(
        child_to_valid_transformations: &HashMap<SymbolNode, HashSet<SymbolNode>>,
        statement: &SymbolNode,
        max_additional_depth: Option<usize>,
    ) -> HashSet<SymbolNode> {
        let children = statement.get_children();
        let filtered_map: HashMap<SymbolNode, HashSet<SymbolNode>> = child_to_valid_transformations
            .clone()
            .into_iter()
            .filter(|(k, _)| children.contains(k))
            .collect();

        let mut new_statements = vec![statement.clone()].into_iter().collect::<HashSet<_>>();

        let n_subsets = 1 << filtered_map.len();
        for bitmask in 0..n_subsets {
            // Bitmask indicates whether to take the child or its transformed versions

            let mut transformed_statements =
                vec![statement.clone()].into_iter().collect::<HashSet<_>>();
            for (i, child) in statement.get_children().iter().enumerate() {
                let mut updated_statements = transformed_statements.clone();
                let should_transform_ith_child = bitmask & (1 << i) != 0;
                if should_transform_ith_child {
                    if let Some(transformed_children_set) = filtered_map.get(child) {
                        for c in transformed_children_set {
                            for transformed_statement in transformed_statements.iter() {
                                let updated_statement = transformed_statement
                                    .clone()
                                    .with_child_replaced(i, c.clone())
                                    .expect("Child index is guaranteed to be in range.");
                                updated_statements.insert(updated_statement);
                            }
                        }
                    }
                } else {
                    // Do nothing; child is fine as is
                }
                transformed_statements = updated_statements;
            }
            new_statements = new_statements
                .union(&transformed_statements)
                .cloned()
                .collect();
        }
        if let Some(d) = max_additional_depth {
            let max_depth = statement.get_depth() + d;
            new_statements = new_statements
                .into_iter()
                .filter(|s| s.get_depth() <= max_depth)
                .collect();
        }
        new_statements
    }

    pub fn transform_at(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
        address: SymbolNodeAddress,
    ) -> Result<SymbolNode, TransformationError> {
        let substatement_to_transform = statement.get_node(address.clone()).ok_or_else(|| {
            TransformationError::InvalidSymbolNodeError(SymbolNodeError::InvalidAddress)
        })?;
        let transformed_substatement = self.transform(hierarchy, &substatement_to_transform)?;
        match statement.replace_node(&address, &transformed_substatement) {
            Ok(transformed_statement) => Ok(transformed_statement),
            Err(e) => Err(TransformationError::InvalidSymbolNodeError(e)),
        }
    }

    pub fn to_interpreted_string(&self, interpretations: &Vec<Interpretation>) -> String {
        match self {
            Self::ExplicitTransformation(t) => t.to_interpreted_string(interpretations),
            Self::AlgorithmTransformation(a) => a.to_string(),
            Self::ApplyToBothSidesTransformation(t) => t.to_interpreted_string(interpretations),
        }
    }

    pub fn contains_arbitrary_nodes(&self) -> bool {
        match self {
            Self::AlgorithmTransformation(_) => false,
            Self::ExplicitTransformation(t) => t.contains_arbitrary_nodes(),
            Self::ApplyToBothSidesTransformation(t) => {
                t.get_transformation().contains_arbitrary_nodes()
            }
        }
    }

    pub fn get_arbitrary_nodes(&self) -> HashSet<SymbolNode> {
        match self {
            Self::AlgorithmTransformation(_) => HashSet::new(),
            Self::ExplicitTransformation(t) => t.get_arbitrary_nodes(),
            Self::ApplyToBothSidesTransformation(t) => t.get_transformation().get_arbitrary_nodes(),
        }
    }

    pub fn instantiate_arbitrary_nodes(
        &self,
        hierarchy: &TypeHierarchy,
        substatements: &HashSet<SymbolNode>,
    ) -> Result<HashSet<Self>, TransformationError> {
        match self {
            Self::AlgorithmTransformation(_) => Ok(vec![self.clone()].into_iter().collect()),
            Self::ExplicitTransformation(t) => Ok(t
                .instantiate_arbitrary_nodes(hierarchy, substatements)?
                .into_iter()
                .map(|t| t.into())
                .collect()),
            Self::ApplyToBothSidesTransformation(t) => Ok(t
                .instantiate_arbitrary_nodes(hierarchy, substatements)?
                .into_iter()
                .map(|t| Self::ApplyToBothSidesTransformation(t))
                .collect()),
        }
    }

    fn cant_possibly_transform_into(
        &self,
        hierarchy: &TypeHierarchy,
        from_statement: &SymbolNode,
        maybe_to_statement: Option<&SymbolNode>,
    ) -> bool {
        match self {
            Self::ExplicitTransformation(t) => {
                let type_to_transform = t.get_from().get_evaluates_to_type();
                // If none of the from statements' types are subtypes of the root of what we're
                // trying to transform, then the transform won't apply
                if !from_statement
                    .get_types()
                    .iter()
                    .any(|type_to_be_transformed| {
                        hierarchy
                            .is_subtype_of(&type_to_be_transformed, &type_to_transform)
                            .unwrap_or(false)
                    })
                {
                    trace!("None of the from_statement's types are subtypes of the root of what we're trying to transform.\nfrom_statement: {:#?}\ntransformation: {:#?}", from_statement, self);
                    return true;
                }
            }
            _ => {}
        };

        if let Some(to_statement) = maybe_to_statement {
            // Optimization if we know the desired type:
            // either the statement must already have it or the transformation must be able to produce
            // it
            let might_produce_correct_type = {
                let desired_type = to_statement.get_evaluates_to_type();
                trace!("desired_type: {:?}", desired_type);
                let transformation_might_produce_type =
                    self.might_produce_type(hierarchy, &desired_type);
                let from_statement_might_produce_type_result =
                    hierarchy.is_subtype_of(&desired_type, &from_statement.get_evaluates_to_type());
                if (from_statement_might_produce_type_result.is_err()) {
                    trace!(
                        "hierarchy is returning an error when trying to find subtypes: {:#?}",
                        from_statement_might_produce_type_result
                    );
                }
                let from_statement_might_produce_type =
                    from_statement_might_produce_type_result.unwrap_or(false);
                if !transformation_might_produce_type {
                    trace!("transformation can't produce the correct type!",);
                }
                if !from_statement_might_produce_type {
                    trace!("from_statement can't produce the correct type!",);
                }
                transformation_might_produce_type || from_statement_might_produce_type
            };
            if !might_produce_correct_type {
                debug!(
                "from_statement can't produce the correct type: {}\nTransformation: {:#?}\nDesired: {}\nHierarchy: {:#?}",
                from_statement.to_symbol_string(),
                self,
                maybe_to_statement.unwrap().to_symbol_string(),
                hierarchy,
            );
                return true;
            }

            // Look at the node deltas of the transformation and check if N applications of t can
            // produce the desired result
            match self {
                Self::ExplicitTransformation(t) => {
                    trace!("node_delta optimization:\ntransformation: {}\nnode_delta: {}\nfrom_statement: {}\nto_statement: {}", t.to_symbol_string(), t.get_node_delta(), from_statement.to_symbol_string(), to_statement.to_symbol_string());
                    let node_delta = t.get_node_delta();
                    if node_delta == 0 {
                        if from_statement.len() != to_statement.len() {
                            debug!(
                                "node_delta = 0 but from and to statements are different lengths."
                            );
                            return true;
                        }
                    } else if node_delta > 0 {
                        if from_statement.len() >= to_statement.len() {
                            debug!("node_delta > 0 meaning the statement will grow but it's being asked to shrink.");
                            return true;
                        }
                    } else if node_delta < 0 {
                        if from_statement.len() <= to_statement.len() {
                            debug!("node_delta < 0 meaning the statement will shrink but it's being asked to grow.");
                            return true;
                        }
                    } else {
                        unreachable!();
                    }
                }
                _ => {
                    trace!("Not an explicit transform, so can't perform certain optimizations.");
                }
            }
        } else {
            trace!("maybe_to_statement.is_none() so we can't perform certain optimizations.");
        }

        trace!("Passed cant_possibly_transform_into.\nTransformation: {:#?}\nDesired: {}\nHierarchy: {:#?}", self, maybe_to_statement.map_or("None".to_string(), |s| s.to_symbol_string()), hierarchy);
        return false;
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AlgorithmTransformation {
    algorithm_type: AlgorithmType,
    operator: Symbol,
    input_type: GeneratedType,
}

impl AlgorithmTransformation {
    pub fn new(algorithm_type: AlgorithmType, operator: Symbol, input_type: GeneratedType) -> Self {
        Self {
            algorithm_type,
            operator,
            input_type,
        }
    }

    pub fn to_symbol_string(&self) -> String {
        format!(
            "{}({})",
            self.algorithm_type.to_string(),
            self.operator.to_string()
        )
    }

    pub fn might_produce_type(&self, hierarchy: &TypeHierarchy, t: &Type) -> bool {
        let generated_transform_parents = self.input_type.get_parents();
        let t_parents_result = hierarchy.get_parents(t);
        match t_parents_result {
            Err(e) => {
                warn!("Type error returning from get_parents: {:#?}", e);
                true
            }
            Ok(t_parents) => generated_transform_parents == &t_parents,
        }
    }

    pub fn get_operator(&self) -> Symbol {
        self.operator.clone()
    }

    pub fn get_input_type(&self) -> GeneratedType {
        self.input_type.clone()
    }

    pub fn get_algorithm_type(&self) -> AlgorithmType {
        self.algorithm_type.clone()
    }

    pub fn transform(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        if !statement.has_children() {
            if self.input_type.satisfies_condition(statement.get_symbol()?) {
                return Ok(statement.clone());
            } else {
                return Err(
                    TransformationError::GeneratedTypeConditionFailedDuringAlgorithm(
                        statement
                            .get_symbol()
                            .expect("We already checked it above.")
                            .clone(),
                    ),
                );
            }
        }

        if statement.get_n_children() != 2 || statement.get_symbol()? != &self.get_operator() {
            return Err(TransformationError::NoValidTransformationsPossible);
        }

        let children = statement.get_children().clone();
        let left = self.transform(hierarchy, &children[0])?;
        let right = self.transform(hierarchy, &children[1])?;

        let final_value = self
            .algorithm_type
            .transform(&left.get_root_as_string(), &right.get_root_as_string())?;
        let to_return = SymbolNode::leaf(Symbol::new_with_same_type_as_value(&final_value));
        Ok(to_return)
    }

    pub fn to_string(&self) -> String {
        format!(
            "{}({})",
            self.algorithm_type.to_string(),
            self.operator.get_name(),
        )
        .to_string()
    }

    fn try_parse_number(symbol_name: &SymbolName) -> Result<f64, TransformationError> {
        // TODO: This will fail on big enough numbers
        return symbol_name
            .parse::<f64>()
            .map_err(|_| TransformationError::UnableToParse(symbol_name.clone()));
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ApplyToBothSidesTransformation {
    symbol: Symbol,
    transformation: ExplicitTransformation,
}

impl ApplyToBothSidesTransformation {
    pub fn new(symbol: Symbol, transformation: ExplicitTransformation) -> Self {
        Self {
            symbol,
            transformation,
        }
    }

    pub fn to_symbol_string(&self) -> String {
        format!(
            "Apply {} to both sides of {}",
            self.transformation.to_symbol_string(),
            self.symbol.to_string()
        )
    }

    pub fn get_symbol(&self) -> &Symbol {
        &self.symbol
    }

    pub fn get_symbol_type(&self) -> Type {
        self.symbol.get_evaluates_to_type()
    }

    pub fn might_produce_type(&self, hierarchy: &TypeHierarchy, t: &Type) -> bool {
        hierarchy
            .is_subtype_of(&self.get_symbol_type(), t)
            .unwrap_or(false)
    }

    pub fn get_transformation(&self) -> &ExplicitTransformation {
        &self.transformation
    }

    pub fn transform(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        if statement.get_symbol()? != &self.symbol {
            return Err(TransformationError::SymbolDoesntMatch(
                statement.get_symbol()?.clone(),
            ));
        }

        if statement.get_n_children() != 2 {
            return Err(TransformationError::ApplyToBothSidesCalledOnNChildren(
                statement.get_n_children(),
            ));
        }

        let left = statement.get_children()[0].clone();
        let right = statement.get_children()[1].clone();

        let left_transformed = self
            .transformation
            .typed_relabel_and_transform(hierarchy, &left)?;
        let right_transformed = self
            .transformation
            .typed_relabel_and_transform(hierarchy, &right)?;

        Ok(SymbolNode::new(
            self.symbol.clone().into(),
            vec![left_transformed, right_transformed],
        ))
    }

    pub fn to_interpreted_string(&self, interpretations: &Vec<Interpretation>) -> String {
        format!(
            "Apply {} to Both Sides of {}",
            self.transformation.to_interpreted_string(interpretations),
            self.symbol.to_string()
        )
    }

    pub fn instantiate_arbitrary_nodes(
        &self,
        hierarchy: &TypeHierarchy,
        substatements: &HashSet<SymbolNode>,
    ) -> Result<HashSet<Self>, TransformationError> {
        Ok(self
            .get_transformation()
            .instantiate_arbitrary_nodes(hierarchy, substatements)?
            .into_iter()
            .map(|t| Self::new(self.get_symbol().clone(), t))
            .collect())
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExplicitTransformation {
    pub from: SymbolNode,
    pub to: SymbolNode,
}

impl From<(SymbolNode, SymbolNode)> for ExplicitTransformation {
    fn from(value: (SymbolNode, SymbolNode)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<(Symbol, SymbolNode)> for ExplicitTransformation {
    fn from(value: (Symbol, SymbolNode)) -> Self {
        Self::new(value.0.into(), value.1)
    }
}

impl ExplicitTransformation {
    pub fn new(from: SymbolNode, to: SymbolNode) -> ExplicitTransformation {
        ExplicitTransformation { from, to }
    }

    pub fn get_from(&self) -> &SymbolNode {
        &self.from
    }

    pub fn get_to(&self) -> &SymbolNode {
        &self.to
    }

    pub fn might_produce_type(&self, hierarchy: &TypeHierarchy, t: &Type) -> bool {
        trace!(
            "ExplicitTransformation::might_produce_type(hierarchy, {:?}) called with self.get_to().get_evaluates_to_type() = {:?}",
            t,
            self.get_to().get_evaluates_to_type()
        );
        let result = hierarchy.is_subtype_of(t, &self.get_to().get_evaluates_to_type());
        if result.is_err() {
            warn!(
                "hierarchy is throwing an error in might_produce_type: {:?}",
                result
            );
        }
        result.unwrap_or(false)
    }

    pub fn reflexivity(
        operator_name: String,
        operator_type: Type,
        object_name: String,
        object_type: Type,
    ) -> Self {
        let object = SymbolNode::leaf(Symbol::new(object_name, object_type));
        let node = SymbolNode::new(
            Symbol::new(operator_name, operator_type).into(),
            vec![object.clone(), object.clone()],
        );
        ExplicitTransformation::new(node.clone(), node)
    }

    pub fn symmetry(
        operator_name: String,
        operator_type: Type,
        object_names: (String, String),
        object_type: Type,
    ) -> Self {
        let left = SymbolNode::leaf(Symbol::new(object_names.0, object_type.clone()));
        let right = SymbolNode::leaf(Symbol::new(object_names.1, object_type));
        let operator = Symbol::new(operator_name, operator_type);
        let from = SymbolNode::new(operator.clone().into(), vec![left.clone(), right.clone()]);
        let to = SymbolNode::new(operator.into(), vec![right.clone(), left.clone()]);
        ExplicitTransformation::new(from, to)
    }

    pub fn commutivity(
        operator_name: String,
        operator_type: Type,
        object_names: (String, String),
        object_type: Type,
    ) -> Self {
        Self::symmetry(operator_name, operator_type, object_names, object_type)
    }

    pub fn associativity(
        operator_name: String,
        operator_type: Type,
        object_names: (String, String, String),
        object_type: Type,
    ) -> Self {
        let a = SymbolNode::leaf(Symbol::new(object_names.0, object_type.clone()));
        let b = SymbolNode::leaf(Symbol::new(object_names.1, object_type.clone()));
        let c = SymbolNode::leaf(Symbol::new(object_names.2, object_type));
        let operator = Symbol::new(operator_name, operator_type);

        let a_b = SymbolNode::new(operator.clone().into(), vec![a.clone(), b.clone()]);
        let a_b_then_c = SymbolNode::new(operator.clone().into(), vec![a_b.clone(), c.clone()]);

        let b_c = SymbolNode::new(operator.clone().into(), vec![b.clone(), c.clone()]);
        let a_then_b_c = SymbolNode::new(operator.clone().into(), vec![a.clone(), b_c.clone()]);

        ExplicitTransformation::new(a_b_then_c, a_then_b_c)
    }

    pub fn to_string(&self) -> String {
        format!("{} -> {}", self.from.to_string(), self.to.to_string())
    }

    pub fn to_symbol_string(&self) -> String {
        format!(
            "{} -> {}",
            self.from.to_symbol_string(),
            self.to.to_symbol_string()
        )
    }

    pub fn to_interpreted_string(&self, interpretations: &Vec<Interpretation>) -> String {
        format!(
            "{} -> {}",
            self.from.to_interpreted_string(interpretations),
            self.to.to_interpreted_string(interpretations)
        )
    }

    pub fn get_variables(&self) -> HashSet<String> {
        let mut variables = self.from.get_symbols();
        variables.extend(self.to.get_symbols());
        variables.into_iter().map(|s| s.get_name()).collect()
    }

    pub fn get_node_delta(&self) -> isize {
        (self.to.len() as isize) - (self.from.len() as isize)
    }

    pub fn contains_arbitrary_nodes(&self) -> bool {
        self.from.contains_arbitrary_nodes() || self.to.contains_arbitrary_nodes()
    }

    pub fn get_arbitrary_nodes(&self) -> HashSet<SymbolNode> {
        self.from
            .get_arbitrary_nodes()
            .union(&mut self.to.get_arbitrary_nodes())
            .cloned()
            .collect()
    }

    pub fn instantiate_arbitrary_nodes(
        &self,
        hierarchy: &TypeHierarchy,
        substatements: &HashSet<SymbolNode>,
    ) -> Result<HashSet<Self>, TransformationError> {
        debug!(
            "instantiate_arbitrary_nodes for {}",
            self.to_symbol_string()
        );
        self.validate_arbitrary_nodes()?;

        let mut to_return = HashSet::new();
        for arbitrary_node in self.get_arbitrary_nodes() {
            trace!("substatements: {:#?}", substatements);
            trace!("arbitrary_node: {:#?}", arbitrary_node);
            trace!("hierarchy: {:#?}", hierarchy);
            let substatement_predicates = substatements
                .iter()
                .filter_map(|s| {
                    hierarchy
                        .is_subtype_of(
                            &s.get_evaluates_to_type(),
                            &arbitrary_node.get_evaluates_to_type(),
                        )
                        .ok()?
                        .then(|| s.get_predicates())
                })
                .flatten()
                .collect::<HashSet<_>>();
            trace!("substatement_predicates: {:#?}", substatement_predicates);
            for predicate in substatement_predicates {
                trace!("predicate: {}", predicate.to_symbol_string());
                let predicate_symbol_names = predicate.get_symbol_names();
                let disambiguated_from = self.from.relabel_to_avoid(&predicate_symbol_names);
                let disambiguated_to = self.to.relabel_to_avoid(&predicate_symbol_names);
                let new_from = disambiguated_from
                    .replace_arbitrary_using_predicate(arbitrary_node.get_symbol()?, &predicate)?;
                let new_to = disambiguated_to
                    .replace_arbitrary_using_predicate(arbitrary_node.get_symbol()?, &predicate)?;

                let transform = ExplicitTransformation::new(new_from, new_to);
                trace!("transform: {}", transform.to_symbol_string());
                to_return.insert(transform);
            }
        }

        Ok(to_return)
    }

    fn validate_arbitrary_nodes(&self) -> Result<(), TransformationError> {
        if self
            .get_arbitrary_nodes()
            .iter()
            .map(|n| n.get_symbol())
            .collect::<HashSet<_>>()
            .len()
            > 1
        {
            return Err(TransformationError::MultipleArbitraryNodeSymbols);
        }

        if self
            .get_arbitrary_nodes()
            .iter()
            .map(|n| n.get_evaluates_to_type())
            .collect::<HashSet<_>>()
            .len()
            > 1
        {
            return Err(TransformationError::MultipleArbitraryNodeTypes);
        }

        Ok(())
    }

    fn relabel_and_transform_at(
        &self,
        statement: &SymbolNode,
        address: SymbolNodeAddress,
    ) -> Result<SymbolNode, TransformationError> {
        let substatement_to_transform = statement.get_node(address.clone()).ok_or_else(|| {
            TransformationError::InvalidSymbolNodeError(SymbolNodeError::InvalidAddress)
        })?;
        let transformed_substatement = self.relabel_and_transform(&substatement_to_transform)?;
        match statement.replace_node(&address, &transformed_substatement) {
            Ok(transformed_statement) => Ok(transformed_statement),
            Err(e) => Err(TransformationError::InvalidSymbolNodeError(e)),
        }
    }

    pub fn typed_relabel_and_transform(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        if self.contains_arbitrary_nodes() {
            return Err(TransformationError::TransformCalledOnArbitrary);
        }
        let generalized_transform = self.generalize_to_fit(hierarchy, statement)?;
        generalized_transform.relabel_and_transform(statement)
    }

    fn relabel_and_transform(
        &self,
        statement: &SymbolNode,
    ) -> Result<SymbolNode, TransformationError> {
        match self.from.get_relabelling(&statement) {
            Ok(relabellings) => self.transform(&statement, &relabellings),
            Err(e) => Err(TransformationError::InvalidSymbolNodeError(e)),
        }
    }

    fn generalize_to_fit(
        &self,
        hierarchy: &TypeHierarchy,
        statement: &SymbolNode,
    ) -> Result<Self, TransformationError> {
        let new_from = hierarchy
            .instantiate(&self.from, statement)
            .map_err(|e| Into::<TransformationError>::into(e))?;
        let substitutions = self
            .from
            .get_typed_relabelling(hierarchy, &new_from)
            .map_err(|e| Into::<TransformationError>::into(e))?;
        let new_to = substitutions.substitute(&self.to);
        Ok(Self::new(new_from, new_to))
    }

    fn transform_all(
        &self,
        statement: &SymbolNode,
        relabellings: &HashMap<String, String>,
    ) -> Result<(SymbolNode, Vec<SymbolNodeAddress>), TransformationError> {
        self.transform_all_from_address(statement, relabellings, Vec::new())
    }

    fn transform_all_from_address(
        &self,
        statement: &SymbolNode,
        relabellings: &HashMap<String, String>,
        address: SymbolNodeAddress,
    ) -> Result<(SymbolNode, Vec<SymbolNodeAddress>), TransformationError> {
        let children_transformation_result = statement
            .get_children()
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let mut child_address = address.clone();
                child_address.push(i);
                self.transform_all_from_address(c, relabellings, child_address)
            })
            .collect::<Result<Vec<(SymbolNode, Vec<SymbolNodeAddress>)>, TransformationError>>()?;
        let (transformed_children, transformed_children_addresses) = children_transformation_result
            .into_iter()
            .unzip::<_, _, Vec<_>, Vec<_>>();
        let transformed_children_addresses: Vec<SymbolNodeAddress> = transformed_children_addresses
            .into_iter()
            .flatten()
            .collect();

        let new_statement = SymbolNode::new(statement.get_root().clone(), transformed_children);

        match self.try_transform(&new_statement, relabellings) {
            Ok((transformed, true)) => {
                let children_addresses: Vec<SymbolNodeAddress> = transformed_children_addresses
                    .into_iter()
                    .map(|mut a| {
                        let mut new_head = address.clone();
                        new_head.append(&mut a);
                        new_head
                    })
                    .collect();
                let mut addresses = vec![address];
                addresses.extend(children_addresses.clone());
                Ok((transformed, addresses))
            }
            Ok((transformed, false)) => Ok((transformed, transformed_children_addresses)),
            Err(e) => Err(e),
        }
    }

    fn try_transform(
        &self,
        statement: &SymbolNode,
        relabelling: &HashMap<String, String>,
    ) -> Result<(SymbolNode, bool), TransformationError> {
        match self.transform(statement, relabelling) {
            Ok(transformed) => Ok((transformed, true)),
            Err(TransformationError::StatementDoesNotMatch(_, _)) => Ok((statement.clone(), false)),
            Err(e) => Err(e),
        }
    }

    fn transform_strict(
        &self,
        statement: &SymbolNode,
        relabellings: &HashMap<String, String>,
    ) -> Result<SymbolNode, TransformationError> {
        if self.get_variables() != relabellings.keys().cloned().collect() {
            return Err(TransformationError::RelabellingsKeysMismatch);
        }

        self.transform(statement, relabellings)
    }

    fn transform(
        &self,
        statement: &SymbolNode,
        relabellings: &HashMap<String, String>,
    ) -> Result<SymbolNode, TransformationError> {
        statement
            .validate()
            .map_err(|e| TransformationError::InvalidSymbolNodeError(e))?;

        let relabelled_from = self
            .from
            .relabel_all(&relabellings.clone().into_iter().collect());
        let relabelled_to = self
            .to
            .relabel_all(&relabellings.clone().into_iter().collect());

        if statement == &relabelled_from {
            Ok(relabelled_to)
        } else {
            Err(TransformationError::StatementDoesNotMatch(
                statement.clone(),
                relabelled_from,
            ))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransformationError {
    MoreThanTwoHypothesesForTheorem(Vec<SymbolNode>, SymbolNode),
    ConflictingTypes(String, Type, Type),
    InvalidSymbolNodeError(SymbolNodeError),
    InvalidFunctionCalledOn(SymbolNodeRoot),
    InvalidTypes(TypeError),
    AlreadyContainsStatement(SymbolNode),
    AlreadyContainsTransformation(Transformation),
    RelabellingsKeysMismatch,
    StatementDoesNotMatch(SymbolNode, SymbolNode),
    StatementContainsArbitraryNode(SymbolNode),
    SymbolDoesntMatch(Symbol),
    ApplyToBothSidesCalledOnNChildren(usize),
    StatementTypesDoNotMatch,
    MissingStatementsInTransformationLattice(Vec<SymbolNode>),
    MissingTransformationsInTransformationLattice(Vec<Transformation>),
    NoValidTransformationsPossible,
    GeneratedTypeConditionFailedDuringAlgorithm(Symbol),
    TransformCalledOnArbitrary,
    UnableToParse(SymbolName),
    ArbitraryNodeHasNonOneChildren,
    MultipleArbitraryNodeSymbols,
    MultipleArbitraryNodeTypes,
}

impl From<SymbolNodeError> for TransformationError {
    fn from(value: SymbolNodeError) -> Self {
        match value {
            SymbolNodeError::ConflictingTypes(name, t_0, t_1) => {
                Self::ConflictingTypes(name, t_0, t_1)
            }
            SymbolNodeError::ArbitraryNodeHasNonOneChildren => Self::ArbitraryNodeHasNonOneChildren,
            SymbolNodeError::InvalidFunctionCalledOn(root) => Self::InvalidFunctionCalledOn(root),
            _ => Self::InvalidSymbolNodeError(value),
        }
    }
}

impl From<TypeError> for TransformationError {
    fn from(value: TypeError) -> Self {
        Self::InvalidTypes(value)
    }
}

#[cfg(test)]
mod test_transformation {
    use crate::{
        parsing::{interpretation::Interpretation, parser::Parser, tokenizer::Token},
        symbol::symbol_node::SymbolNodeRoot,
    };

    use super::*;

    #[test]
    fn test_transformation_can_be_a_tautology() {
        let mut types = TypeHierarchy::chain(vec!["Boolean".into(), "=".into()]).unwrap();
        types
            .add_child_to_parent("&".into(), "Boolean".into())
            .unwrap();
        types
            .add_child_to_parent("Integer".into(), Type::Object)
            .unwrap();
        let interpretations = vec![
            Interpretation::infix_operator("=_0".into(), 1, "=".into()), //Disambiguation
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::infix_operator("&".into(), 2, "&".into()),
            Interpretation::outfix_operator(("|".into(), "|".into()), 2, "Integer".into()),
            Interpretation::singleton("p", "Boolean".into()),
            Interpretation::singleton("p_0", "Boolean".into()), // Disambiguation
            Interpretation::singleton("q", "Boolean".into()),
            Interpretation::singleton("q_0", "Boolean".into()), // Disambiguation
            Interpretation::arbitrary_functional("F".into(), 99, "Boolean".into()),
        ];

        let parser = Parser::new(interpretations.clone());

        let custom_tokens = vec![
            "=_0".to_string(),
            "=".to_string(),
            "&".to_string(),
            "|".to_string(),
        ];

        let parse = |s: &str| parser.parse_from_string(custom_tokens.clone(), s).unwrap();

        let p = parse("p");
        let q = parse("q");
        let f_of_p = parse("F(p)");
        let p_equals_p = parse("p=p");
        let p_0_equals_p = parse("p_0=p");
        let p_equals_p_0 = parse("p=p_0");
        let p_equals_0_p = parse("p=_0p");
        let p_0_equals_0_p_0 = parse("p_0=_0p_0");
        let reflexivity = ExplicitTransformation::new(f_of_p.clone(), p_equals_p.clone());

        let substatements = vec![p.clone(), q.clone(), p_equals_p.clone()]
            .into_iter()
            .collect::<HashSet<_>>();
        let expected: HashSet<_> = vec![
            ExplicitTransformation::new(p.clone(), p_equals_p.clone()),
            ExplicitTransformation::new(p_0_equals_p.clone(), p_0_equals_0_p_0.clone()),
            ExplicitTransformation::new(p_equals_p_0.clone(), p_0_equals_0_p_0.clone()),
            ExplicitTransformation::new(p_equals_p.clone(), p_equals_0_p.clone()),
        ]
        .into_iter()
        .collect();
        let actual = reflexivity
            .instantiate_arbitrary_nodes(&types, &substatements)
            .unwrap();
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
    }

    #[test]
    fn test_transformation_instantiates_arbitrary_nodes() {
        let mut types = TypeHierarchy::chain(vec!["Boolean".into(), "=".into()]).unwrap();
        types
            .add_child_to_parent("&".into(), "Boolean".into())
            .unwrap();
        types
            .add_child_to_parent("Integer".into(), Type::Object)
            .unwrap();
        let interpretations = vec![
            Interpretation::infix_operator("=_0".into(), 1, "Boolean".into()), // Disambiguation
            Interpretation::infix_operator("=".into(), 1, "Boolean".into()),
            Interpretation::infix_operator("&".into(), 2, "Boolean".into()),
            Interpretation::outfix_operator(("|".into(), "|".into()), 2, "Integer".into()),
            Interpretation::postfix_operator("!".into(), 3, "Integer".into()),
            Interpretation::prefix_operator("-".into(), 4, "Integer".into()),
            Interpretation::singleton("p_0", "Boolean".into()), // Disambiguation
            Interpretation::singleton("p", "Boolean".into()),
            Interpretation::singleton("q_0", "Boolean".into()), // Disambiguation
            Interpretation::singleton("q", "Boolean".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
            Interpretation::arbitrary_functional("F".into(), 99, "Boolean".into()),
            Interpretation::arbitrary_functional("G".into(), 99, "Boolean".into()),
        ];

        let parser = Parser::new(interpretations.clone());

        let custom_tokens = vec![
            "=_0".to_string(),
            "=".to_string(),
            "&".to_string(),
            "|".to_string(),
            "!".to_string(),
            "-".to_string(),
        ];

        let parse = |s: &str| parser.parse_from_string(custom_tokens.clone(), s).unwrap();

        let p = parse("p");
        let q = parse("q");
        let p_equals_q = parse("p=q");
        let f_of_p_equals_f_of_q = parse("F(p)=F(q)");
        let transformation =
            ExplicitTransformation::new(p_equals_q.clone(), f_of_p_equals_f_of_q.clone());
        assert_eq!(
            transformation
                .instantiate_arbitrary_nodes(&types, &vec![].into_iter().collect())
                .unwrap(),
            vec![].into_iter().collect()
        );

        let substatements = vec![p.clone(), q.clone(), p_equals_q.clone()]
            .into_iter()
            .collect::<HashSet<_>>();
        let expected = vec![
            ExplicitTransformation::new(p_equals_q.clone(), p_equals_q.clone()),
            ExplicitTransformation::new(parse("p_0=_0q"), parse("(p=p_0)=_0(p=q)")),
            ExplicitTransformation::new(parse("p=_0q_0"), parse("(p=q)=_0(q_0=q)")),
        ]
        .into_iter()
        .collect();
        let actual = transformation
            .instantiate_arbitrary_nodes(&types, &substatements)
            .unwrap();

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

        let f_of_p_equals_g_of_q = parse("F(p)=G(q)");
        let transformation =
            ExplicitTransformation::new(p_equals_q.clone(), f_of_p_equals_g_of_q.clone());

        assert_eq!(
            transformation.instantiate_arbitrary_nodes(&types, &substatements),
            Err(TransformationError::MultipleArbitraryNodeSymbols)
        );
    }

    #[test]
    fn test_transformation_lattice_removes_transformations() {
        let hierarchy = TypeHierarchy::chain(vec!["Proposition".into()]).unwrap();
        let parser = Parser::new(vec![
            Interpretation::singleton("p", "Proposition".into()),
            Interpretation::singleton("q", "Proposition".into()),
            Interpretation::infix_operator("|".into(), 1, "Proposition".into()),
            Interpretation::infix_operator("^".into(), 2, "Proposition".into()),
            Interpretation::infix_operator("=>".into(), 3, "Proposition".into()),
            Interpretation::singleton("a", "Proposition".into()),
            Interpretation::singleton("b", "Proposition".into()),
            Interpretation::singleton("c", "Proposition".into()),
        ]);

        let as_proposition =
            |name: &str| Symbol::new(name.to_string(), "Proposition".into()).into();

        let mut lattice = TransformationLattice::empty();

        let from = SymbolNode::new(
            SymbolNodeRoot::Join,
            vec![as_proposition("p"), as_proposition("q")],
        );
        let p_and_q = parser
            .parse_from_string(vec!["^".to_string()], "p^q")
            .unwrap();
        let transform: Transformation = ExplicitTransformation::new(from, p_and_q.clone()).into();
        lattice
            .add_available_transformation(AvailableTransformation::Axiom(transform.clone()))
            .unwrap();
        lattice
            .remove_transformation_and_all_dependents(&transform)
            .unwrap();
        assert_eq!(lattice.available_transformations.len(), 0);
        assert_eq!(lattice.statements.len(), 0);

        lattice
            .add_available_transformation(AvailableTransformation::Axiom(transform.clone()))
            .unwrap();
        let p = as_proposition("p");
        let q = as_proposition("q");
        lattice.add_hypothesis(p.clone()).unwrap();
        lattice.add_hypothesis(q.clone()).unwrap();
        lattice
            .try_transform_into(&hierarchy, p_and_q.clone())
            .unwrap();
        lattice
            .remove_transformation_and_all_dependents(&transform)
            .unwrap();
        assert_eq!(lattice.available_transformations.len(), 0);
        assert_eq!(
            lattice.statements,
            vec![p.clone(), q.clone()].into_iter().collect()
        );

        lattice
            .add_available_transformation(AvailableTransformation::Axiom(transform.clone()))
            .unwrap();
        lattice
            .try_transform_into(&hierarchy, p_and_q.clone())
            .unwrap();
        let p_and_p_and_q = parser
            .parse_from_string(vec!["^".to_string()], "p^(p^q)")
            .unwrap();
        lattice
            .try_transform_into(&hierarchy, p_and_p_and_q.clone())
            .unwrap();
        lattice.derive_theorem(&p_and_p_and_q).unwrap();
        let theorem = Transformation::ExplicitTransformation(ExplicitTransformation::new(
            p.clone().join(q.clone()),
            p_and_p_and_q.clone(),
        ));
        assert_eq!(
            lattice
                .available_transformations
                .iter()
                .map(|t| t.get_transformation())
                .cloned()
                .collect::<HashSet<_>>(),
            vec![transform.clone(), theorem.clone()]
                .into_iter()
                .collect::<HashSet<_>>()
        );
        let (removed_transformations, removed_statements) = lattice
            .remove_transformation_and_all_dependents(&theorem)
            .unwrap();
        assert_eq!(
            removed_transformations,
            vec![theorem.clone()].into_iter().collect()
        );
        assert_eq!(removed_statements, HashSet::new());
        assert_eq!(
            lattice.statements,
            vec![p.clone(), q.clone(), p_and_q.clone(), p_and_p_and_q.clone()]
                .into_iter()
                .collect()
        );
        assert_eq!(
            lattice.available_transformations,
            vec![AvailableTransformation::Axiom(transform.clone())]
                .into_iter()
                .collect()
        );

        lattice.derive_theorem(&p_and_p_and_q).unwrap();
        lattice.remove_statement_and_all_dependents(&p).unwrap();
        lattice.remove_statement_and_all_dependents(&q).unwrap();
        let (removed_transformations, removed_statements) = lattice
            .remove_transformation_and_all_dependents(&theorem)
            .unwrap();
        assert_eq!(
            removed_transformations,
            vec![theorem.clone()].into_iter().collect()
        );
        assert_eq!(removed_statements, HashSet::new());
        assert_eq!(lattice.statements, HashSet::new(),);
        assert_eq!(
            lattice.available_transformations,
            vec![AvailableTransformation::Axiom(transform.clone())]
                .into_iter()
                .collect()
        );
        assert_eq!(lattice.transformations_from, HashMap::new());
        assert_eq!(lattice.transformations_to, HashMap::new());
    }

    #[test]
    fn test_transformation_lattice_removes_statements() {
        let hierarchy = TypeHierarchy::chain(vec!["Proposition".into()]).unwrap();
        let parser = Parser::new(vec![
            Interpretation::singleton("p", "Proposition".into()),
            Interpretation::singleton("q", "Proposition".into()),
            Interpretation::infix_operator("|".into(), 1, "Proposition".into()),
            Interpretation::infix_operator("^".into(), 2, "Proposition".into()),
            Interpretation::infix_operator("=>".into(), 3, "Proposition".into()),
            Interpretation::singleton("a", "Proposition".into()),
            Interpretation::singleton("b", "Proposition".into()),
            Interpretation::singleton("c", "Proposition".into()),
        ]);

        let as_proposition =
            |name: &str| Symbol::new(name.to_string(), "Proposition".into()).into();

        let mut lattice = TransformationLattice::empty();

        let from = SymbolNode::new(
            SymbolNodeRoot::Join,
            vec![as_proposition("p"), as_proposition("q")],
        );
        let p_and_q = parser
            .parse_from_string(vec!["^".to_string()], "p^q")
            .unwrap();
        let transform: Transformation = ExplicitTransformation::new(from, p_and_q.clone()).into();
        lattice
            .add_available_transformation(AvailableTransformation::Axiom(transform))
            .unwrap();
        lattice.add_hypothesis(as_proposition("p")).unwrap();
        lattice.add_hypothesis(as_proposition("q")).unwrap();
        lattice
            .try_transform_into(&hierarchy, p_and_q.clone())
            .unwrap();

        let downstream = lattice
            .get_downstream_statements(&as_proposition("p"))
            .unwrap();
        assert_eq!(downstream, vec![p_and_q.clone()].into_iter().collect());

        let removed = lattice
            .remove_statement_and_all_dependents(&as_proposition("p"))
            .unwrap();
        assert_eq!(
            removed,
            vec![as_proposition("p"), p_and_q.clone()]
                .into_iter()
                .collect()
        );
        assert_eq!(
            lattice.get_statements(),
            &vec![as_proposition("q")].into_iter().collect()
        );

        let q_or_p = parser
            .parse_from_string(vec!["|".to_string()], "q|p")
            .unwrap();
        let transform: Transformation =
            ExplicitTransformation::new(as_proposition("q"), q_or_p.clone()).into();
        lattice
            .add_available_transformation(AvailableTransformation::Axiom(transform))
            .unwrap();
        lattice
            .try_transform_into(&hierarchy, q_or_p.clone())
            .unwrap();
        let removed = lattice
            .remove_statement_and_all_dependents(&as_proposition("q"))
            .unwrap();
        assert_eq!(
            removed,
            vec![as_proposition("q"), q_or_p.clone()]
                .into_iter()
                .collect()
        );
        assert_eq!(lattice.get_statements(), &HashSet::new());
    }

    #[test]
    fn test_transformation_joint_transforms() {
        let hierarchy = TypeHierarchy::chain(vec!["Proposition".into()]).unwrap();
        let parser = Parser::new(vec![
            Interpretation::singleton("p", "Proposition".into()),
            Interpretation::singleton("q", "Proposition".into()),
            Interpretation::infix_operator("^".into(), 1, "Proposition".into()),
            Interpretation::infix_operator("=>".into(), 2, "Proposition".into()),
            Interpretation::singleton("a", "Proposition".into()),
            Interpretation::singleton("b", "Proposition".into()),
            Interpretation::singleton("c", "Proposition".into()),
        ]);

        let as_proposition =
            |name: &str| Symbol::new(name.to_string(), "Proposition".into()).into();
        let from = SymbolNode::new(
            SymbolNodeRoot::Join,
            vec![as_proposition("p"), as_proposition("q")],
        );
        let p_and_q = parser
            .parse_from_string(vec!["^".to_string()], "p^q")
            .unwrap();
        let transform: Transformation = ExplicitTransformation::new(from, p_and_q.clone()).into();
        let actual = transform
            .joint_transform(&hierarchy, &as_proposition("p"), &as_proposition("q"))
            .unwrap();
        assert_eq!(actual, p_and_q);

        let actual = transform
            .joint_transform(&hierarchy, &as_proposition("a"), &as_proposition("b"))
            .unwrap();
        let expected = parser
            .parse_from_string(vec!["^".to_string()], "a^b")
            .unwrap();
        assert_eq!(actual, expected);

        let actual = transform
            .joint_transform(&hierarchy, &p_and_q, &as_proposition("b"))
            .unwrap();
        let expected = parser
            .parse_from_string(vec!["^".to_string()], "(p^q)^b")
            .unwrap();
        assert_eq!(actual, expected);

        let p_implies_q = parser
            .parse_from_string(vec!["=>".to_string()], "p=>q")
            .unwrap();
        let from = SymbolNode::new(
            SymbolNodeRoot::Join,
            vec![as_proposition("p"), p_implies_q.clone()],
        );
        let transform: Transformation =
            ExplicitTransformation::new(from, as_proposition("q")).into();

        let expected = as_proposition("q");
        assert_eq!(
            transform
                .joint_transform(&hierarchy, &as_proposition("p"), &p_implies_q)
                .unwrap(),
            expected
        );
        // Order shouldn't matter
        assert_eq!(
            transform
                .joint_transform(&hierarchy, &p_implies_q, &as_proposition("p"))
                .unwrap(),
            expected
        );
    }

    #[test]
    fn test_transformation_applies_algorithm() {
        let algorithm = AlgorithmTransformation::new(
            AlgorithmType::Addition,
            Symbol::new("+".to_string(), "Real".into()),
            GeneratedType::new_numeric("Real".into()),
        );
        let parser = Parser::new(vec![
            Interpretation::singleton("a", "Real".into()),
            Interpretation::singleton("b", "Real".into()),
            Interpretation::singleton("c", "Real".into()),
            Interpretation::singleton("1", "Real".into()),
            Interpretation::singleton("2", "Real".into()),
            Interpretation::infix_operator("+".into(), 1, "Real".into()),
        ]);
        let from = parser
            .parse_from_string(vec!["+".to_string()], "a+b+c")
            .unwrap();
        let hierarchy = TypeHierarchy::chain(vec!["Real".into()]).unwrap();
        assert_eq!(
            algorithm.transform(&hierarchy, &from),
            Err(
                TransformationError::GeneratedTypeConditionFailedDuringAlgorithm(Symbol::new(
                    "a".to_string(),
                    "Real".into()
                ))
            )
        );

        let from = parser
            .parse_from_string(vec!["+".to_string()], "1+2")
            .unwrap();
        assert_eq!(
            algorithm.transform(&hierarchy, &from),
            Ok(SymbolNode::leaf(Symbol::new("3".to_string(), "3".into())))
        );
    }

    #[test]
    fn test_transformation_gets_valid_transformations() {
        let mut hierarchy =
            TypeHierarchy::chain(vec!["Real".into(), "Integer".into(), "=".into()]).unwrap();
        hierarchy.add_chain(vec!["Boolean".into()]).unwrap();
        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
            Interpretation::arbitrary_functional("Any".into(), 99, "Boolean".into()),
        ];
        let parser = Parser::new(interpretations);

        let custom_tokens = vec!["=".to_string()];

        let irrelevant_transform: Transformation = ExplicitTransformation::associativity(
            "*".to_string(),
            "*".into(),
            ("j".to_string(), "l".to_string(), "k".to_string()),
            "Irrelevant".into(),
        )
        .into();

        let x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "x=y")
            .unwrap();

        assert_eq!(
            irrelevant_transform.get_valid_transformations(&hierarchy, &x_equals_y, None),
            vec![x_equals_y.clone()].into_iter().collect()
        );
        let transformation: Transformation = ExplicitTransformation::commutivity(
            "=".to_string(),
            "=".into(),
            ("a".to_string(), "b".to_string()),
            "Integer".into(),
        )
        .into();
        let y_equals_x = parser
            .parse_from_string(custom_tokens.clone(), "y = x")
            .unwrap();
        assert_eq!(
            transformation.transform(&hierarchy, &x_equals_y),
            Ok(y_equals_x.clone())
        );
        assert_eq!(
            transformation.get_valid_transformations(&hierarchy, &x_equals_y, None),
            vec![x_equals_y.clone(), y_equals_x.clone()]
                .into_iter()
                .collect()
        );

        // TODO Figure out if we think this needs to work
        //        let conversion = Transformation::new(
        //            Symbol::new("x".to_string(), "Integer".into()).into(),
        //            Symbol::new("x".to_string(), "Real".into()).into(),
        //        );
        //        assert_eq!(
        //            conversion.get_valid_transformations(
        //                &hierarchy,
        //                &Symbol::new("1".to_string(), "Integer".into()).into()
        //            ),
        //            vec![
        //                Symbol::new("1".to_string(), "Integer".into()).into(),
        //                Symbol::new("1".to_string(), "Real".into()).into(),
        //            ]
        //            .into_iter()
        //            .collect()
        //        );
        //        assert_eq!(
        //            conversion
        //                .get_valid_transformations(&hierarchy, &x_equals_y.clone())
        //                .len(),
        //            4
        //        );

        let x_equals_y_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "x=y=z") // ((x=y)=z)
            .unwrap();

        let expected = vec![
            x_equals_y_equals_z.clone(),
            parser
                .parse_from_string(custom_tokens.clone(), "y=x=z") // ((x=y)=z) => ((y=x)=z)
                .unwrap(),
            parser
                .parse_from_string(custom_tokens.clone(), "z=(x=y)") // ((x=y)=z) => (z=(x=y))
                .unwrap(),
            parser
                .parse_from_string(custom_tokens.clone(), "z=(y=x)") // ((x=y)=z) => (z=(y=x))
                .unwrap(),
        ];
        assert_eq!(
            transformation.get_valid_transformations(&hierarchy, &x_equals_y_equals_z, None),
            expected.into_iter().collect()
        );

        // Using overloaded names shouldn't matter
        let transformation: Transformation = ExplicitTransformation::commutivity(
            "=".to_string(),
            "=".into(),
            ("x".to_string(), "y".to_string()),
            "Integer".into(),
        )
        .into();

        let x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "x=y")
            .unwrap();

        let expected = vec![
            x_equals_y.clone(),
            parser
                .parse_from_string(custom_tokens.clone(), "y=x")
                .unwrap(),
        ];

        let actual = transformation.get_valid_transformations(&hierarchy, &x_equals_y, None);

        assert_eq!(actual, expected.into_iter().collect());

        let x_equals_y_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "x=y=z") // ((x=y)=z)
            .unwrap();

        let expected = vec![
            x_equals_y_equals_z.clone(),
            parser
                .parse_from_string(custom_tokens.clone(), "y=x=z") // ((x=y)=z) => ((y=x)=z)
                .unwrap(),
            parser
                .parse_from_string(custom_tokens.clone(), "z=(x=y)") // ((x=y)=z) => (z=(x=y))
                .unwrap(),
            parser
                .parse_from_string(custom_tokens.clone(), "z=(y=x)") // ((x=y)=z) => (z=(y=x))
                .unwrap(),
        ];

        let actual =
            transformation.get_valid_transformations(&hierarchy, &x_equals_y_equals_z, None);

        assert_eq!(actual, expected.into_iter().collect());

        let arbitrary_x_equals_arbitrary_y = parser
            .parse_from_string(custom_tokens.clone(), "Any(x)=Any(y)")
            .unwrap();
        let transformation: Transformation =
            ExplicitTransformation::new(x_equals_y.clone(), arbitrary_x_equals_arbitrary_y.clone())
                .into();
        assert_eq!(
            transformation.transform(&mut hierarchy, &x_equals_y),
            Err(TransformationError::TransformCalledOnArbitrary)
        );
    }

    #[test]
    fn test_transformation_transforms() {
        let transformation =
            ExplicitTransformation::new(SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b"));

        let statement = SymbolNode::leaf_object("c");

        let transformed = transformation.transform_strict(
            &statement,
            &vec![
                ("a".to_string(), "c".to_string()),
                ("b".to_string(), "d".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(transformed, Ok(SymbolNode::leaf_object("d")));

        let transformation = ExplicitTransformation::new(
            SymbolNode::new(
                "=".into(),
                vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
            ),
            SymbolNode::new(
                "=".into(),
                vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("a")],
            ),
        );

        let self_equals_statement = SymbolNode::new(
            "=".into(),
            vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("a")],
        );

        let transformed = transformation.transform_strict(
            &self_equals_statement,
            &vec![
                ("a".to_string(), "a".to_string()),
                ("b".to_string(), "a".to_string()),
                ("=".to_string(), "=".to_string()),
            ]
            .into_iter()
            .collect(),
        );
        assert_eq!(transformed, Ok(self_equals_statement));
    }

    #[test]
    fn test_transformation_transforms_all() {
        let commutativity = ExplicitTransformation::new(
            SymbolNode::new(
                "=".into(),
                vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
            ),
            SymbolNode::new(
                "=".into(),
                vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("a")],
            ),
        );

        let statement = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );

        let (transformed, addresses) = commutativity
            .transform_all(
                &statement,
                &vec![
                    ("a".to_string(), "a".to_string()),
                    ("b".to_string(), "b".to_string()),
                    ("c".to_string(), "c".to_string()),
                    ("=".to_string(), "=".to_string()),
                ]
                .into_iter()
                .collect(),
            )
            .unwrap();

        let expected = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("b"), SymbolNode::leaf_object("a")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );
        assert_eq!(transformed, expected);
        assert_eq!(addresses, vec![vec![0]]);
    }

    #[test]
    fn test_transformation_generalizes_to_fit() {
        let hierarchy = TypeHierarchy::chain(vec!["Integer".into(), "=".into()]).unwrap();
        let trivial_t =
            ExplicitTransformation::new(SymbolNode::leaf_object("c"), SymbolNode::leaf_object("d"));

        let trivial = SymbolNode::leaf_object("c");
        assert_eq!(
            trivial_t.generalize_to_fit(&hierarchy, &trivial).unwrap(),
            trivial_t
        );

        let different_t =
            ExplicitTransformation::new(SymbolNode::leaf_object("a"), SymbolNode::leaf_object("d"));

        let different_name = SymbolNode::leaf_object("a");
        assert_eq!(
            different_t
                .generalize_to_fit(&hierarchy, &different_name)
                .unwrap(),
            different_t
        );

        let overloaded_t =
            ExplicitTransformation::new(SymbolNode::leaf_object("d"), SymbolNode::leaf_object("d"));
        let overloaded_name = SymbolNode::leaf_object("d");
        assert_eq!(
            overloaded_t
                .generalize_to_fit(&hierarchy, &overloaded_name)
                .unwrap(),
            overloaded_t
        );

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
        ];
        let parser = Parser::new(interpretations);

        let custom_tokens = vec!["=".to_string()];

        let symmetry = ExplicitTransformation::symmetry(
            "=".to_string(),
            "=".into(),
            ("a".to_string(), "b".to_string()),
            "Integer".into(),
        );

        let x_equals_y_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=z")
            .unwrap();

        let z_equals_x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "z=(x=y)")
            .unwrap();

        assert_eq!(
            symmetry
                .generalize_to_fit(&hierarchy, &x_equals_y_equals_z)
                .unwrap(),
            ExplicitTransformation::new(x_equals_y_equals_z.clone(), z_equals_x_equals_y.clone()),
            "\n\n{} \nvs. \n{}",
            symmetry
                .generalize_to_fit(&hierarchy, &x_equals_y_equals_z)
                .unwrap()
                .to_string(),
            ExplicitTransformation::new(x_equals_y_equals_z, z_equals_x_equals_y.clone())
                .to_string(),
        );

        let symmetry = ExplicitTransformation::symmetry(
            "=".to_string(),
            "=".into(),
            ("x".to_string(), "y".to_string()),
            "Integer".into(),
        );

        assert_eq!(
            symmetry
                .generalize_to_fit(&hierarchy, &x_equals_y_equals_z)
                .unwrap(),
            ExplicitTransformation::new(x_equals_y_equals_z.clone(), z_equals_x_equals_y.clone()),
            "\n\n{} \nvs. \n{}",
            symmetry
                .generalize_to_fit(&hierarchy, &x_equals_y_equals_z)
                .unwrap()
                .to_string(),
            ExplicitTransformation::new(x_equals_y_equals_z, z_equals_x_equals_y.clone())
                .to_string(),
        );

        // TODO: Figure out whether this needs to work
        //        let conversion = Transformation::new(
        //            Symbol::new("x".to_string(), "Integer".into()).into(),
        //            Symbol::new("x".to_string(), "Real".into()).into(),
        //        );
        //        assert_eq!(
        //            conversion
        //                .typed_generalize_to_fit(
        //                    &hierarchy,
        //                    &Symbol::new("1".to_string(), "Integer".into()).into(),
        //                )
        //                .unwrap(),
        //            Transformation::new(
        //                Symbol::new("1".to_string(), "Integer".into()).into(),
        //                Symbol::new("1".to_string(), "Real".into()).into(),
        //            )
        //        );
    }

    #[test]
    fn test_transformation_typed_transforms_at() {
        let hierarchy =
            TypeHierarchy::chain(vec!["Real".into(), "Integer".into(), "=".into()]).unwrap();
        let transformation: Transformation =
            ExplicitTransformation::new(SymbolNode::leaf_object("c"), SymbolNode::leaf_object("d"))
                .into();

        let a_equals_b = SymbolNode::new(
            "=".into(),
            vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
        );

        let d_equals_b = SymbolNode::new(
            "=".into(),
            vec![SymbolNode::leaf_object("d"), SymbolNode::leaf_object("b")],
        );

        let transformed = transformation.transform_at(&hierarchy, &a_equals_b, vec![0]);

        assert_eq!(transformed, Ok(d_equals_b));

        let a_equals_b_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );

        let a_equals_d_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("d")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );
        let transformed = transformation.transform_at(&hierarchy, &a_equals_b_equals_c, vec![0, 1]);

        assert_eq!(transformed, Ok(a_equals_d_equals_c));

        let interpretations = vec![
            Interpretation::infix_operator("=".into(), 1, "=".into()),
            Interpretation::singleton("x", "Integer".into()),
            Interpretation::singleton("y", "Integer".into()),
            Interpretation::singleton("z", "Integer".into()),
        ];
        let parser = Parser::new(interpretations);

        let custom_tokens = vec!["=".to_string()];

        let transformation: Transformation = ExplicitTransformation::symmetry(
            "=".to_string(),
            "=".into(),
            ("a".to_string(), "b".to_string()),
            "Integer".into(),
        )
        .into();

        let x_equals_y_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "(x=y)=z")
            .unwrap();

        let z_equals_x_equals_y = parser
            .parse_from_string(custom_tokens.clone(), "z=(x=y)")
            .unwrap();

        assert_eq!(
            transformation.transform_at(&hierarchy, &x_equals_y_equals_z, vec![]),
            Ok(z_equals_x_equals_y.clone())
        );

        let y_equals_x_equals_z = parser
            .parse_from_string(custom_tokens.clone(), "(y=x)=z")
            .unwrap();

        assert_eq!(
            transformation.transform_at(&hierarchy, &x_equals_y_equals_z, vec![0]),
            Ok(y_equals_x_equals_z)
        );

        let transformation: Transformation = ExplicitTransformation::symmetry(
            "=".to_string(),
            "=".into(),
            ("x".to_string(), "y".to_string()),
            "Integer".into(),
        )
        .into();

        assert_eq!(
            transformation.transform_at(&hierarchy, &x_equals_y_equals_z, vec![]),
            Ok(z_equals_x_equals_y)
        );

        // TODO: Figure out if we think this needs to work
        //        let conversion = Transformation::new(
        //            Symbol::new("x".to_string(), "Integer".into()).into(),
        //            Symbol::new("x".to_string(), "Real".into()).into(),
        //        );
        //        assert_eq!(
        //            conversion
        //                .typed_transform_at(
        //                    &hierarchy,
        //                    &Symbol::new("1".to_string(), "Integer".into()).into(),
        //                    vec![],
        //                )
        //                .unwrap(),
        //            Symbol::new("1".to_string(), "Real".into()).into(),
        //        );
    }

    #[test]
    fn test_transformation_transforms_at() {
        let transformation =
            ExplicitTransformation::new(SymbolNode::leaf_object("c"), SymbolNode::leaf_object("d"));

        let a_equals_b = SymbolNode::new(
            "=".into(),
            vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
        );

        let d_equals_b = SymbolNode::new(
            "=".into(),
            vec![SymbolNode::leaf_object("d"), SymbolNode::leaf_object("b")],
        );

        let transformed = transformation.relabel_and_transform_at(&a_equals_b, vec![0]);

        assert_eq!(transformed, Ok(d_equals_b));

        let a_equals_b_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("b")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );

        let a_equals_d_equals_c = SymbolNode::new(
            "=".into(),
            vec![
                SymbolNode::new(
                    "=".into(),
                    vec![SymbolNode::leaf_object("a"), SymbolNode::leaf_object("d")],
                ),
                SymbolNode::leaf_object("c"),
            ],
        );
        let transformed = transformation.relabel_and_transform_at(&a_equals_b_equals_c, vec![0, 1]);

        assert_eq!(transformed, Ok(a_equals_d_equals_c));
    }

    #[test]
    fn test_transformation_reflexivity() {
        let transformation = ExplicitTransformation::reflexivity(
            "=".to_string(),
            "Rational".into(),
            "x".to_string(),
            Type::Object,
        );

        let interpretations = vec![Interpretation::infix_operator(
            Token::Object("=".to_string()),
            3,
            "Rational".into(),
        )];

        let parser = Parser::new(interpretations);
        let expected = parser
            .parse_from_string(vec!["=".to_string()], "x=x")
            .unwrap();

        assert_eq!(transformation.from, expected);
        assert_eq!(transformation.to, expected);
    }

    #[test]
    fn test_transformation_gets_node_delta() {
        assert_eq!(
            ExplicitTransformation::symmetry(
                "+".to_string(),
                "Complex".into(),
                ("x".to_string(), "y".to_string()),
                "Complex".into()
            )
            .get_node_delta(),
            0
        );

        let interpretations = vec![
            Interpretation::infix_operator(Token::Object("+".to_string()), 3, "Complex".into()),
            Interpretation::singleton("x", "Complex".into()),
            Interpretation::singleton("y", "Complex".into()),
            Interpretation::singleton("2", "Complex".into()),
            Interpretation::singleton("4", "Complex".into()),
        ];
        let parser = Parser::new(interpretations);

        let two_plus_two = parser
            .parse_from_string(vec!["+".to_string()], "2+2")
            .unwrap();

        let four = parser
            .parse_from_string(vec!["+".to_string()], "4")
            .unwrap();

        assert_eq!(
            ExplicitTransformation::new(two_plus_two.clone(), four.clone()).get_node_delta(),
            -2
        );
        assert_eq!(
            ExplicitTransformation::new(four.clone(), two_plus_two.clone()).get_node_delta(),
            2
        );
    }

    #[test]
    fn test_transformation_symmetry() {
        let transformation = ExplicitTransformation::symmetry(
            "+".to_string(),
            "Complex".into(),
            ("x".to_string(), "y".to_string()),
            "Complex".into(),
        );

        let interpretations = vec![
            Interpretation::infix_operator(Token::Object("+".to_string()), 3, "Complex".into()),
            Interpretation::singleton("x", "Complex".into()),
            Interpretation::singleton("y", "Complex".into()),
        ];
        let parser = Parser::new(interpretations);

        let expected_from = parser
            .parse_from_string(vec!["+".to_string()], "x+y")
            .unwrap();

        let expected_to = parser
            .parse_from_string(vec!["+".to_string()], "y+x")
            .unwrap();

        assert_eq!(transformation.from, expected_from);
        assert_eq!(transformation.to, expected_to);
    }
}
