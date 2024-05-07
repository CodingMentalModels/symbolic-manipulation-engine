use clap::ArgMatches;

use crate::{
    cli::filesystem::FileSystem, config::STATE_DIRECTORY_RELATIVE_PATH, parsing::parser::Parser,
    workspace::workspace::Workspace,
};

pub struct Cli {
    pub filesystem: FileSystem,
}

impl Cli {
    pub fn new(filesystem: FileSystem) -> Self {
        Self { filesystem }
    }

    pub fn hello_world(&self) -> Result<String, String> {
        Ok("Hello World!".to_string())
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
        match self.filesystem.write_file(
            STATE_DIRECTORY_RELATIVE_PATH,
            "workspace.toml",
            workspace.serialize(),
        ) {
            true => println!("Created workspace.toml"),
            false => {
                return Err("Couldn't create workspace.toml".to_string());
            }
        }
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
        self.load_workspace().map(|w| w.to_string())
    }

    pub fn get_transformations(&self, sub_matches: &ArgMatches) -> Result<String, String> {
        let mut workspace = self.load_workspace()?;
        // TODO: Format these strings so that they look like what the user expects
        // TODO: Are we going to have line separator issues across different platforms?
        match sub_matches.get_one::<String>("partial-statement") {
            None => Err("No partial statement provided.".to_string()),
            Some(partial_statement) => Ok(workspace
                .get_valid_transformations(partial_statement)
                .into_iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("\n")),
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
                return workspace
                    .try_transform_into(tree)
                    .map_err(|e| format!("Workspace error: {:?}", e))
                    .map(|statement| statement.to_string());
            }
        }
    }

    fn load_workspace(&self) -> Result<Workspace, String> {
        if !self.filesystem.path_exists(STATE_DIRECTORY_RELATIVE_PATH) {
            return Err("No workspace exists in this directory".to_string());
        }

        let workspace = match self
            .filesystem
            .read_file(STATE_DIRECTORY_RELATIVE_PATH, "workspace.toml")
        {
            Ok(contents) => match Workspace::deserialize(&contents) {
                Ok(workspace) => return Ok(workspace),
                Err(e) => {
                    return Err(format!("Couldn't deserialize workspace.toml: {}", e));
                }
            },
            Err(e) => {
                return Err("Couldn't read workspace.toml".to_string());
            }
        };
    }
}

#[cfg(test)]
mod test_cli {
    use super::*;
}
