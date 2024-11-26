use log::debug;
use std::{collections::HashSet, path::PathBuf};

use clap::ArgMatches;
use serde_json::to_string;

use crate::{
    cli::filesystem::FileSystem,
    config::{
        CONTEXT_DIRECTORY_RELATIVE_PATH, STATE_DIRECTORY_RELATIVE_PATH, WORKSPACE_STATE_FILE_NAME,
        WORKSPACE_STATE_TEST_INPUT_FILE_NAME,
    },
    constants::*,
    context::context::Context,
    parsing::{
        interpretation::{
            ExpressionPrecedence, ExpressionType, Interpretation, InterpretationCondition,
            InterpretedType,
        },
        parser::Parser,
    },
    symbol::{
        algorithm::AlgorithmType,
        symbol_node::SymbolNode,
        symbol_type::{GeneratedType, GeneratedTypeCondition, Type},
        transformation::{ExplicitTransformation, Transformation},
    },
    workspace::workspace::{
        StatementIndex, TransformationIndex, Workspace, WorkspaceTransaction,
        WorkspaceTransactionItem, WorkspaceTransactionStore,
    },
};

pub struct Cli {
    pub filesystem: FileSystem,
    pub mode: CliMode,
}

impl Cli {
    pub fn new(mut filesystem: FileSystem, mode: CliMode) -> Self {
        if mode == CliMode::Testing {
            Self::prepare_for_testing(&mut filesystem).unwrap();
        }
        Self { filesystem, mode }
    }

    fn prepare_for_testing(filesystem: &mut FileSystem) -> Result<(), String> {
        remove_workspace_state_file(filesystem);
        copy_test_input_to_workspace_state(filesystem)?;
        Ok(())
    }

    fn get_relative_path(&self) -> &str {
        STATE_DIRECTORY_RELATIVE_PATH
    }

    pub fn init(&self) -> Result<String, String> {
        if self.filesystem.path_exists(self.get_relative_path()) {
            return Err("A workspace already exists in this directory".to_string());
        }

        match self.filesystem.create_directory(self.get_relative_path()) {
            true => println!("Created directory {}", self.get_relative_path()),
            false => {
                return Err(format!(
                    "Couldn't create directory {}",
                    self.get_relative_path()
                ));
            }
        }

        let workspace_store = WorkspaceTransactionStore::default();
        self.update_workspace_store(workspace_store)?;
        return Ok(format!(
            "Initialized new workspace in {}",
            self.filesystem.get_root_directory_path()
        ));
    }

    pub fn rmws(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        if !self.filesystem.path_exists(self.get_relative_path()) {
            return Err("No workspace exists in this directory".to_string());
        }

        if !sub_matches.get_flag("force") {
            return Err("Use the --force flag to remove the workspace".to_string());
        }

        match self.filesystem.remove_directory(self.get_relative_path()) {
            true => {
                return Ok(format!(
                    "Removed workspace in {}",
                    self.filesystem.get_root_directory_path()
                ));
            }
            false => {
                return Err(format!(
                    "Couldn't remove directory {}",
                    self.get_relative_path()
                ));
            }
        };
    }

    pub fn ls(&self) -> Result<String, String> {
        self.load_workspace_store()?
            .compile()
            .to_json()
            .map_err(|e| format!("Serialization Error during ls: {:?}", e).to_string())
    }

    pub fn add_interpretation(&mut self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        let maybe_condition = sub_matches.get_one::<String>("condition");
        let is_generated_integer = sub_matches.get_flag("any-integer");
        let is_generated_numeric = sub_matches.get_flag("any-numeric");
        let is_arbitrary = sub_matches.get_flag("arbitrary");
        let output_type = sub_matches
            .get_one::<String>("output-type")
            .map(|o| o.into())
            .unwrap_or(Type::Object);
        let condition = match (maybe_condition, is_generated_integer, is_generated_numeric) {
            (_, true, true) => {
                return Err("Interpretation cannot be both any-integer and any-numeric.".to_string())
            }
            (Some(_condition), true, false) => {
                return Err(
                    "Interpretation with any-integer should not have a condition.".to_string(),
                );
            }
            (Some(_condition), false, true) => {
                return Err(
                    "Interpretation with any-numeric should not have a condition.".to_string(),
                );
            }
            (_, true, _) => InterpretationCondition::IsInteger,
            (_, _, true) => InterpretationCondition::IsNumeric,
            (None, false, false) => return Err("No condition provided.".to_string()),
            (Some(condition), _, _) => InterpretationCondition::Matches(condition.into()),
        };
        let expression_type = match sub_matches.get_one::<String>("expression-type") {
            None => return Err("No expression type provided.".to_string()),
            Some(expression_type) => ExpressionType::try_parse(expression_type)
                .map_err(|e| format!("Unable to parse expression type: {:?}", e).to_string())?,
        };
        let precedence = match sub_matches.get_one::<String>("precedence") {
            None => return Err("No precedence provided.".to_string()),
            Some(precedence) => precedence
                .parse::<ExpressionPrecedence>()
                .map_err(|e| format!("Unable to parse precedence: {:?}", e.to_string()))?,
        };
        let interpretation_output_type = match condition {
            InterpretationCondition::IsInteger | InterpretationCondition::IsNumeric => {
                InterpretedType::SameAsValue
            }
            _ => {
                if is_arbitrary {
                    InterpretedType::ArbitraryReturning(output_type.clone())
                } else {
                    output_type.clone().into()
                }
            }
        };
        let new_interpretation = Interpretation::new(
            condition.clone(),
            expression_type,
            precedence,
            interpretation_output_type.clone(),
        );
        workspace_store
            .add(WorkspaceTransactionItem::AddInterpretation(new_interpretation).into())
            .map_err(|e| format!("Workspace Error: {:?}", e).to_string())?;
        if interpretation_output_type == InterpretedType::SameAsValue {
            let generated_type_condition = match condition {
                InterpretationCondition::IsInteger => GeneratedTypeCondition::IsInteger,
                InterpretationCondition::IsNumeric => GeneratedTypeCondition::IsNumeric,
                InterpretationCondition::SatisfiesRegex(p) => {
                    GeneratedTypeCondition::SatisfiesRegex(p)
                }
                InterpretationCondition::IsObject => {
                    unimplemented!();
                }
                InterpretationCondition::Matches(_) => {
                    return Err("Attempted to convert InterpretationCondition::Matches to a GeneratedTypeCondition.".to_string());
                }
            };
            let generated_type = GeneratedType::new(
                generated_type_condition,
                vec![output_type].into_iter().collect(),
            );
            workspace_store
                .add(WorkspaceTransactionItem::AddGeneratedType(generated_type).into())
                .map_err(|e| format!("Unable to add generated type: {:?}", e).to_string())?;
        }
        self.update_workspace_store(workspace_store)?;
        return Ok("Interpretation added.".to_string());
    }

    pub fn duplicate_interpretation(&mut self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        let workspace = workspace_store.compile();
        match sub_matches.get_one::<String>("index") {
            None => Err("Invalid Interpretation Index.".to_string()),
            Some(index_string) => match index_string.parse::<usize>() {
                Ok(index) => {
                    let new_match_string = sub_matches
                        .get_one::<String>("new-match-string")
                        .ok_or("Invalid New Match String.")?;
                    let mut new_interpretation = workspace
                        .get_interpretation(index)
                        .map_err(|e| format!("Workspace Error: {:?}", e))?
                        .clone();
                    new_interpretation
                        .update_match_string(new_match_string)
                        .map_err(|e| format!("Interpretation Error: {:?}", e))?;
                    let new_idx = workspace.get_interpretations().len();
                    let to_return = workspace_store
                        .add(WorkspaceTransaction::new(vec![
                            WorkspaceTransactionItem::DuplicateInterpretation(index),
                            WorkspaceTransactionItem::UpdateInterpretation((
                                new_idx,
                                new_interpretation,
                            )),
                        ]))
                        .map_err(|e| format!("Workspace Error: {:?}", e).to_string())
                        .map(|interpretation| {
                            format!("Updated Interpretation: {:?}", interpretation).to_string()
                        });
                    self.update_workspace_store(workspace_store)?;
                    to_return
                }
                Err(_) => Err(format!("Unable to parse index: {}", index_string).to_string()),
            },
        }
    }

    pub fn update_interpretation(&mut self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        match sub_matches.get_one::<String>("index") {
            None => Err("Invalid Interpretation Index.".to_string()),
            Some(index_string) => match index_string.parse::<usize>() {
                Ok(index) => {
                    let new_match_string = sub_matches
                        .get_one::<String>("new-match-string")
                        .ok_or("Invalid New Match String.")?;
                    let mut new_interpretation = workspace_store
                        .compile()
                        .get_interpretation(index)
                        .map_err(|e| format!("Workspace Error: {:?}", e))?
                        .clone();
                    new_interpretation
                        .update_match_string(new_match_string)
                        .map_err(|e| format!("Interpretation Error: {:?}", e))?;
                    let to_return = workspace_store
                        .add(
                            WorkspaceTransactionItem::UpdateInterpretation((
                                index,
                                new_interpretation,
                            ))
                            .into(),
                        )
                        .map_err(|e| format!("Workspace Error: {:?}", e).to_string())
                        .map(|interpretation| {
                            format!("Updated Interpretation: {:?}", interpretation).to_string()
                        });
                    self.update_workspace_store(workspace_store)?;
                    to_return
                }
                Err(_) => Err(format!("Unable to parse index: {}", index_string).to_string()),
            },
        }
    }

    pub fn remove_interpretation(&mut self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        match sub_matches.get_one::<String>("index") {
            None => Err("Invalid Interpretation Index.".to_string()),
            Some(index_string) => match index_string.parse::<usize>() {
                Ok(index) => {
                    let to_return = workspace_store
                        .add(WorkspaceTransactionItem::RemoveInterpretation(index).into())
                        .map_err(|e| format!("Workspace Error: {:?}", e).to_string())
                        .map(|interpretation| {
                            format!("Removed Interpretation: {:?}", interpretation).to_string()
                        });
                    self.update_workspace_store(workspace_store)?;
                    to_return
                }
                Err(_) => Err(format!("Unable to parse index: {}", index_string).to_string()),
            },
        }
    }

    pub fn add_type(&mut self, sub_matches: &ArgMatches) -> Result<String, String> {
        let maybe_type_name = sub_matches.get_one::<String>("type-name");
        let maybe_parent_name = sub_matches.get_one::<String>("parent-type-name");
        let mut workspace_store = self.load_workspace_store()?;
        let workspace = workspace_store.compile();
        match maybe_type_name {
            Some(type_name) => {
                let parent_type = match maybe_parent_name {
                    None => Type::Object,
                    Some(parent_name) => {
                        match workspace
                            .get_types()
                            .get_types()
                            .iter()
                            .find(|t| t == &&Type::NamedType(parent_name.clone()))
                        {
                            None => {
                                return Err(format!(
                                    "No type in type hierarchy named {}.",
                                    parent_name
                                )
                                .to_string())
                            }
                            Some(parent_type) => parent_type.clone(),
                        }
                    }
                };
                workspace_store
                    .add_type_to_parent(type_name.into(), parent_type.clone())
                    .map_err(|e| format!("Workspace Error: {:?}", e).to_string())?;

                self.update_workspace_store(workspace_store)?;

                Ok(format!("{} added to {}.", type_name, parent_type.pretty_print()).to_string())
            }
            None => return Err(format!("No type name provided.")),
        }
    }

    pub fn get_transformations(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        let workspace_store = self.load_workspace_store()?;
        let workspace = workspace_store.compile();
        let maybe_statement_scope = match sub_matches.get_one::<String>("statements-in-scope") {
            None => None,
            Some(s) => Some(get_statement_scope(&workspace, s)?),
        };
        let maybe_transformation_scope =
            match sub_matches.get_one::<String>("transformations-in-scope") {
                None => None,
                Some(s) => Some(get_transformation_scope(&workspace, s)?),
            };
        match sub_matches.get_one::<String>("partial-statement") {
            None => Err("No partial statement provided.".to_string()),
            Some(partial_statement) => {
                let workspace = workspace_store.compile();
                let serialized_result = to_string(
                    &workspace
                        .get_valid_transformations(
                            partial_statement,
                            maybe_statement_scope,
                            maybe_transformation_scope,
                        )
                        .map_err(|e| format!("Error getting valid transformations: {:?}", e))?
                        .into_iter()
                        .map(|n| {
                            n.to_interpreted_string(workspace_store.compile().get_interpretations())
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| e.to_string())?;
                Ok(serialized_result)
            }
        }
    }

    pub fn get_transformations_from(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        let workspace_store = self.load_workspace_store()?;
        match sub_matches.get_one::<String>("statement-index") {
            None => Err("No statement index provided.".to_string()),
            Some(index_string) => match index_string.parse::<usize>() {
                Ok(statement_index) => {
                    let workspace = workspace_store.compile();
                    let statement = workspace
                        .get_statement(statement_index)
                        .map_err(|e| format!("Workspace error: {:?}", e))?
                        .clone();
                    let mut result = workspace
                        .get_valid_transformations_from(statement)
                        .map_err(|e| format!("Error getting valid transformations: {:?}", e))?
                        .into_iter()
                        .map(|n| {
                            n.to_interpreted_string(workspace_store.compile().get_interpretations())
                        })
                        .collect::<Vec<_>>();
                    result.sort_by(|a, b| a.len().cmp(&b.len()));
                    let serialized_result = to_string(&result).map_err(|e| e.to_string())?;
                    Ok(serialized_result)
                }
                Err(_) => Err(format!("Unable to parse index: {}", index_string).to_string()),
            },
        }
    }

    pub fn add_transformation(&mut self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        let from_as_string = match sub_matches.get_one::<String>("from") {
            None => return Err("No from provided.".to_string()),
            Some(from) => from,
        };
        let to_as_string = match sub_matches.get_one::<String>("to") {
            None => return Err("No to provided.".to_string()),
            Some(to) => to,
        };
        let is_equivalence = sub_matches.get_flag("is-equivalence");
        workspace_store
            .add_parsed_transformation(is_equivalence, &from_as_string, &to_as_string)
            .map_err(|e| format!("Workspace Error: {:?}", e).to_string())?;
        self.update_workspace_store(workspace_store)?;
        return Ok("Transformation added.".to_string());
    }

    pub fn add_algorithm(&mut self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        let algorithm_type = match sub_matches.get_one::<String>("algorithm-type") {
            None => return Err("No algorithm-type provided.".to_string()),
            Some(algorithm_type_string) => AlgorithmType::from_string(algorithm_type_string)
                .map_err(|_| {
                    format!(
                        "Invalid Algorithm Type {}.  Valid types are:\n{}",
                        algorithm_type_string,
                        AlgorithmType::all()
                            .into_iter()
                            .map(|t| t.to_string())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    )
                })?,
        };
        let operator = match sub_matches.get_one::<String>("operator") {
            None => return Err("No operator provided.".to_string()),
            Some(operator) => operator,
        };
        let input_type = match sub_matches.get_one::<String>("input-type") {
            None => return Err("No input-type provided.".to_string()),
            Some(input_type) => input_type,
        };
        workspace_store
            .add_algorithm(&algorithm_type, &operator, &input_type)
            .map_err(|e| format!("Workspace Error: {:?}", e).to_string())?;
        self.update_workspace_store(workspace_store)?;
        return Ok("Algorithm added.".to_string());
    }

    pub fn add_joint_transformation(&mut self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        let left_from_as_string = match sub_matches.get_one::<String>("left-from") {
            None => return Err("No left-from provided.".to_string()),
            Some(from) => from,
        };
        let right_from_as_string = match sub_matches.get_one::<String>("right-from") {
            None => return Err("No right-from provided.".to_string()),
            Some(from) => from,
        };
        let to_as_string = match sub_matches.get_one::<String>("to") {
            None => return Err("No to provided.".to_string()),
            Some(to) => to,
        };
        workspace_store
            .add_parsed_joint_transformation(
                &left_from_as_string,
                right_from_as_string,
                &to_as_string,
            )
            .map_err(|e| format!("Workspace Error: {:?}", e).to_string())?;
        self.update_workspace_store(workspace_store)?;
        return Ok("Transformation added.".to_string());
    }

    pub fn hypothesize(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        match sub_matches.get_one::<String>("statement") {
            None => Err("No statement provided to derive".to_string()),
            Some(statement) => {
                let to_return = workspace_store
                    .add_parsed_hypothesis(statement)
                    .map_err(|e| format!("Parser Error: {:?}", e).to_string())
                    .map(|_| "Hypthesis added.".to_string());
                self.update_workspace_store(workspace_store)?;
                to_return
            }
        }
    }

    pub fn derive(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        let workspace = workspace_store.compile();
        let maybe_statement_scope = match sub_matches.get_one::<String>("statements-in-scope") {
            None => None,
            Some(s) => Some(get_statement_scope(&workspace, s)?),
        };
        let maybe_transformation_scope =
            match sub_matches.get_one::<String>("transformations-in-scope") {
                None => None,
                Some(s) => Some(get_transformation_scope(&workspace, s)?),
            };
        match sub_matches.get_one::<String>("statement") {
            None => return Err("No statement provided to derive".to_string()),
            Some(statement) => {
                let to_return = workspace_store
                    .try_transform_into_parsed(
                        statement,
                        maybe_statement_scope,
                        maybe_transformation_scope,
                    )
                    .map_err(|e| format!("Workspace error: {:?} (Statement: {})", e, statement))
                    .map(|statement| {
                        statement.to_interpreted_string(workspace.get_interpretations())
                    });
                self.update_workspace_store(workspace_store)?;
                to_return
            }
        }
    }

    pub fn undo(&self) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        let did_undo = workspace_store.undo().is_some();
        self.update_workspace_store(workspace_store)?;
        if did_undo {
            return Ok("Undo complete.".to_string());
        } else {
            return Ok("Nothing to undo.".to_string());
        }
    }

    pub fn redo(&self) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        let did_redo = workspace_store.redo().is_some();
        self.update_workspace_store(workspace_store)?;
        if did_redo {
            return Ok("Redo complete.".to_string());
        } else {
            return Ok("Nothing to redo.".to_string());
        }
    }

    pub fn evaluate(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        let workspace_store = self.load_workspace_store()?;
        let statement = match sub_matches.get_one::<String>("statement") {
            None => return Err("No statement provided.".to_string()),
            Some(s) => s,
        };
        let workspace = workspace_store.compile();
        workspace
            .evaluate_from_string(statement)
            .map(|s| s.to_interpreted_string(workspace.get_interpretations()))
            .map_err(|e| format!("Error evaluating: {:?}", e))
    }

    pub fn export_context(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        let workspace_store = self.load_workspace_store()?;
        let context = Context::from_workspace(&workspace_store.compile())
            .map_err(|e| format!("Error exporting workspace: {:?}", e))?;
        let name = match sub_matches.get_one::<String>("name") {
            None => {
                return Err("No name provided.".to_string());
            }
            Some(name) => name,
        };
        let filename = Self::get_context_filename(name);
        if !self.filesystem.path_exists(CONTEXT_DIRECTORY_RELATIVE_PATH) {
            self.filesystem
                .create_directory(CONTEXT_DIRECTORY_RELATIVE_PATH);
        }
        self.filesystem
            .write_file(
                CONTEXT_DIRECTORY_RELATIVE_PATH,
                &filename,
                context.serialize(),
                sub_matches.get_flag("force"),
            )
            .map(|_| format!("Context exported to '{}'.", filename).to_string())
    }

    pub fn import_context(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace_store = self.load_workspace_store()?;
        let name = match sub_matches.get_one::<String>("name") {
            None => {
                return Err("No name provided.".to_string());
            }
            Some(name) => name,
        };
        let filename = Self::get_context_filename(name);
        let serialized_context = self
            .filesystem
            .read_file(CONTEXT_DIRECTORY_RELATIVE_PATH, &filename)?;
        let context = Context::deserialize(&serialized_context)
            .map_err(|e| format!("Couldn't deserialize context: {:?}.", e))?;
        workspace_store
            .import_context(context)
            .map_err(|e| format!("Error importing context: {:?}", e))?;
        self.update_workspace_store(workspace_store)?;
        Ok("Context imported.".to_string())
    }

    pub fn ls_contexts(&self) -> Result<String, String> {
        let names: Vec<String> = self
            .filesystem
            .ls(CONTEXT_DIRECTORY_RELATIVE_PATH)
            .map_err(|e| format!("Error reading context directory path: {:?}.", e).to_string())?
            .iter()
            .filter_map(|path| path.strip_suffix(".toml"))
            .map(|s| s.to_string())
            .collect();
        serde_json::to_string(&names)
            .map_err(|e| format!("Serialization Error during ls_contexts: {:?}", e).to_string())
    }

    fn update_workspace_store(
        &self,
        mut workspace_store: WorkspaceTransactionStore,
    ) -> Result<(), String> {
        workspace_store.truncate(N_TRANSACTIONS_TO_KEEP_IN_WORKSPACE_STORE);
        self.filesystem.write_file(
            self.get_relative_path(),
            WORKSPACE_STATE_FILE_NAME,
            workspace_store
                .serialize()
                .map_err(|e| format!("{:?}", e))?,
            true,
        )
    }

    pub fn load_workspace_store(&self) -> Result<WorkspaceTransactionStore, String> {
        if !self.filesystem.path_exists(self.get_relative_path()) {
            return Err(format!(
                "No workspace exists in this directory: {}",
                self.get_relative_path()
            )
            .to_string());
        }

        match self
            .filesystem
            .read_file(self.get_relative_path(), WORKSPACE_STATE_FILE_NAME)
        {
            Ok(contents) => match WorkspaceTransactionStore::deserialize(&contents) {
                Ok(workspace_store) => {
                    debug!(
                        "Loaded workspace store:\n{:?}\nWorkspace:\n{:?}",
                        workspace_store,
                        workspace_store.compile()
                    );
                    return Ok(workspace_store);
                }
                Err(e) => {
                    return Err(format!(
                        "Couldn't deserialize {}: {}",
                        WORKSPACE_STATE_FILE_NAME, e
                    ));
                }
            },
            Err(_) => {
                return Err(format!("Couldn't read {}", WORKSPACE_STATE_FILE_NAME).to_string());
            }
        };
    }

    pub fn get_command_history(&self) -> Result<String, String> {
        Ok(self.load_workspace_store()?.get_command_history_as_string())
    }

    pub fn update_command_history(&self, command: String) -> Result<(), String> {
        let mut store = self.load_workspace_store()?;
        store.add_command_to_history(command);
        self.update_workspace_store(store)
    }

    fn get_context_filename(name: &str) -> String {
        format!("{}.toml", name).to_string()
    }
}

fn get_statement_scope(workspace: &Workspace, s: &String) -> Result<HashSet<SymbolNode>, String> {
    let statement_indices: Vec<StatementIndex> =
        serde_json::from_str(s).map_err(|e| format!("Deserialization error: {:?}", e))?;
    let mut statement_scope = HashSet::new();
    for i in statement_indices {
        let statement = workspace
            .get_statement(i)
            .map_err(|_| format!("Invalid statement index: {}", i))?;
        statement_scope.insert(statement);
    }
    Ok(statement_scope)
}

fn get_transformation_scope(
    workspace: &Workspace,
    s: &String,
) -> Result<HashSet<Transformation>, String> {
    let transformation_indices: Vec<TransformationIndex> =
        serde_json::from_str(s).map_err(|e| format!("Deserialization error: {:?}", e))?;
    let mut transformation_scope = HashSet::new();
    for i in transformation_indices {
        let transformation = workspace
            .get_transformation(i)
            .map_err(|_| format!("Invalid transformation index: {}", i))?;
        transformation_scope.insert(transformation);
    }
    Ok(transformation_scope)
}

fn copy_test_input_to_workspace_state(filesystem: &mut FileSystem) -> Result<(), String> {
    filesystem.copy_file(
        STATE_DIRECTORY_RELATIVE_PATH,
        WORKSPACE_STATE_TEST_INPUT_FILE_NAME,
        WORKSPACE_STATE_FILE_NAME,
    )
}

fn remove_workspace_state_file(filesystem: &mut FileSystem) {
    let _removed = filesystem.remove_file(STATE_DIRECTORY_RELATIVE_PATH, WORKSPACE_STATE_FILE_NAME);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CliMode {
    Production,
    Testing,
}
