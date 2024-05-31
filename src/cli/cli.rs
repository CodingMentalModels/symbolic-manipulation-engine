use clap::ArgMatches;
use serde_json::to_string;

use crate::{
    cli::filesystem::FileSystem,
    config::STATE_DIRECTORY_RELATIVE_PATH,
    parsing::{
        interpretation::{
            ExpressionPrecedence, ExpressionType, Interpretation, InterpretationCondition,
            InterpretedType,
        },
        parser::Parser,
    },
    symbol::{symbol_type::Type, transformation::Transformation},
    workspace::workspace::Workspace,
};

pub struct Cli {
    pub filesystem: FileSystem,
}

impl Cli {
    pub fn new(filesystem: FileSystem) -> Self {
        Self { filesystem }
    }

    pub fn init(&self) -> Result<String, String> {
        if self.filesystem.path_exists(STATE_DIRECTORY_RELATIVE_PATH) {
            return Err("A workspace already exists in this directory".to_string());
        }

        match self
            .filesystem
            .create_directory(STATE_DIRECTORY_RELATIVE_PATH)
        {
            true => println!("Created directory {}", STATE_DIRECTORY_RELATIVE_PATH),
            false => {
                return Err(format!(
                    "Couldn't create directory {}",
                    STATE_DIRECTORY_RELATIVE_PATH
                ));
            }
        }

        let workspace = Workspace::default();
        self.update_workspace(workspace)?;
        return Ok(format!(
            "Initialized new workspace in {}",
            self.filesystem.get_root_directory_path()
        ));
    }

    pub fn rmws(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        if !self.filesystem.path_exists(STATE_DIRECTORY_RELATIVE_PATH) {
            return Err("No workspace exists in this directory".to_string());
        }

        if !sub_matches.get_flag("force") {
            return Err("Use the --force flag to remove the workspace".to_string());
        }

        match self
            .filesystem
            .remove_directory(STATE_DIRECTORY_RELATIVE_PATH)
        {
            true => {
                return Ok(format!(
                    "Removed workspace in {}",
                    self.filesystem.get_root_directory_path()
                ));
            }
            false => {
                return Err(format!(
                    "Couldn't remove directory {}",
                    STATE_DIRECTORY_RELATIVE_PATH
                ));
            }
        };
    }

    pub fn ls(&self) -> Result<String, String> {
        self.load_workspace()?
            .to_json()
            .map_err(|_| "Serialization Error.".to_string())
    }

    pub fn add_interpretation(&mut self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace = self.load_workspace()?;
        let condition = match sub_matches.get_one::<String>("condition") {
            None => return Err("No condition provided.".to_string()),
            Some(condition) => InterpretationCondition::Matches(condition.into()),
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
        let output_type = match sub_matches.get_one::<String>("output-type") {
            None => return Err("No output type provided.".to_string()),
            Some(output_type) => InterpretedType::Type(output_type.into()),
        };
        workspace
            .add_interpretation(Interpretation::new(
                condition,
                expression_type,
                precedence,
                output_type,
            ))
            .map_err(|e| format!("Workspace Error: {:?}", e).to_string())?;
        self.update_workspace(workspace)?;
        return Ok("Interpretation added.".to_string());
    }

    pub fn add_type(&mut self, sub_matches: &ArgMatches) -> Result<String, String> {
        let maybe_type_name = sub_matches.get_one::<String>("type-name");
        let maybe_parent_name = sub_matches.get_one::<String>("parent-type-name");
        let mut workspace = self.load_workspace()?;
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
                workspace
                    .add_type_to_parent(type_name.into(), parent_type.clone())
                    .map_err(|e| format!("Workspace Error: {:?}", e).to_string())?;

                self.update_workspace(workspace)?;

                Ok(format!("{} added to {}.", type_name, parent_type.pretty_print()).to_string())
            }
            None => return Err(format!("No type name provided.")),
        }
    }

    pub fn get_transformations(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        let workspace = self.load_workspace()?;
        match sub_matches.get_one::<String>("partial-statement") {
            None => Err("No partial statement provided.".to_string()),
            Some(partial_statement) => {
                let serialized_result = to_string(
                    &workspace
                        .get_valid_transformations(partial_statement)
                        .into_iter()
                        .map(|n| n.to_interpreted_string(workspace.get_interpretations()))
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| e.to_string())?;
                Ok(serialized_result)
            }
        }
    }

    pub fn add_transformation(&mut self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace = self.load_workspace()?;
        let from_as_string = match sub_matches.get_one::<String>("from") {
            None => return Err("No from provided.".to_string()),
            Some(from) => from,
        };
        let to_as_string = match sub_matches.get_one::<String>("to") {
            None => return Err("No to provided.".to_string()),
            Some(to) => to,
        };
        let from = workspace
            .parse_from_string(&from_as_string)
            .map_err(|e| format!("Workspace Error: {:?}", e).to_string())?;
        let to = workspace
            .parse_from_string(&to_as_string)
            .map_err(|e| format!("Workspace Error: {:?}", e).to_string())?;
        workspace
            .add_transformation(Transformation::new(from, to))
            .map_err(|e| format!("Workspace Error: {:?}", e).to_string())?;
        self.update_workspace(workspace)?;
        return Ok("Transformation added.".to_string());
    }

    pub fn hypothesize(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace = self.load_workspace()?;
        match sub_matches.get_one::<String>("statement") {
            None => Err("No statement provided to derive".to_string()),
            Some(statement) => {
                let tree = workspace
                    .parse_from_string(statement)
                    .map_err(|e| format!("Parser Error: {:?}", e).to_string())?;
                let to_return = workspace
                    .add_statement(tree)
                    .map_err(|e| format!("Workspace Error: {:?}", e).to_string())
                    .map(|_| "Hypthesis added.".to_string());
                self.update_workspace(workspace)?;
                to_return
            }
        }
    }

    pub fn derive(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace = self.load_workspace()?;
        match sub_matches.get_one::<String>("statement") {
            None => return Err("No statement provided to derive".to_string()),
            Some(statement) => {
                let tree = workspace
                    .parse_from_string(statement)
                    .map_err(|e| format!("Parser Error: {:?}", e).to_string())?;
                let to_return = workspace
                    .try_transform_into(tree)
                    .map_err(|e| format!("Workspace error: {:?}", e))
                    .map(|statement| {
                        statement.to_interpreted_string(workspace.get_interpretations())
                    });
                self.update_workspace(workspace)?;
                to_return
            }
        }
    }

    fn update_workspace(&self, workspace: Workspace) -> Result<(), String> {
        match self.filesystem.write_file(
            STATE_DIRECTORY_RELATIVE_PATH,
            "workspace.toml",
            workspace.serialize().map_err(|e| format!("{:?}", e))?,
        ) {
            true => println!("Overwrote workspace.toml"),
            false => {
                return Err("Couldn't create workspace.toml".to_string());
            }
        }
        Ok(())
    }

    fn load_workspace(&self) -> Result<Workspace, String> {
        if !self.filesystem.path_exists(STATE_DIRECTORY_RELATIVE_PATH) {
            return Err("No workspace exists in this directory".to_string());
        }

        match self
            .filesystem
            .read_file(STATE_DIRECTORY_RELATIVE_PATH, "workspace.toml")
        {
            Ok(contents) => match Workspace::deserialize(&contents) {
                Ok(workspace) => return Ok(workspace),
                Err(e) => {
                    return Err(format!("Couldn't deserialize workspace.toml: {}", e));
                }
            },
            Err(_) => {
                return Err("Couldn't read workspace.toml".to_string());
            }
        };
    }
}

#[cfg(test)]
mod test_cli {
    use super::*;
}
