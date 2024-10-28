use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::to_string;

use crate::config::{COMMAND_HISTORY_FILE_NAME, STATE_DIRECTORY_RELATIVE_PATH};

use super::filesystem::FileSystem;

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

    fn get_relative_path() -> String {
        STATE_DIRECTORY_RELATIVE_PATH.to_string()
    }

    pub fn load_or_get_new(filesystem: &FileSystem) -> Result<Self, String> {
        if filesystem.file_exists(&Self::get_relative_path(), COMMAND_HISTORY_FILE_NAME) {
            Self::load(filesystem)
        } else {
            Ok(Self::empty())
        }
    }

    pub fn load(filesystem: &FileSystem) -> Result<Self, String> {
        if !filesystem.path_exists(&Self::get_relative_path()) {
            return Err(format!(
                "No command history exists in this directory: {}",
                Self::get_relative_path()
            )
            .to_string());
        }

        match filesystem.read_file(&Self::get_relative_path(), COMMAND_HISTORY_FILE_NAME) {
            Ok(contents) => match Self::deserialize(&contents) {
                Ok(history) => {
                    debug!("Loaded command history:\n{:?}", history);
                    return Ok(history);
                }
                Err(e) => {
                    return Err(format!(
                        "Couldn't deserialize {}: {}",
                        COMMAND_HISTORY_FILE_NAME, e
                    ));
                }
            },
            Err(_) => {
                return Err(format!("Couldn't read {}", COMMAND_HISTORY_FILE_NAME).to_string());
            }
        };
    }

    pub fn update(&self, filesystem: &FileSystem) -> Result<(), String> {
        filesystem.write_file(
            &Self::get_relative_path(),
            COMMAND_HISTORY_FILE_NAME,
            self.serialize().map_err(|e| format!("{:?}", e))?,
            true,
        )
    }

    fn deserialize(serialized: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(serialized)
    }

    fn serialize(&self) -> Result<String, String> {
        toml::to_string(self).map_err(|e| e.to_string())
    }
}
