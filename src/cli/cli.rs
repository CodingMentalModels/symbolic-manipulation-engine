use crate::{cli::filesystem::FileSystem, config::STATE_DIRECTORY_RELATIVE_PATH, workspace::workspace::Workspace};

pub struct Cli {
    pub filesystem: FileSystem,
}

impl Cli {

    pub fn new(filesystem: FileSystem) -> Self {
        Self {
            filesystem,
        }
    }

    pub fn init(&self) -> Result<String, String> {
        if self.filesystem.path_exists(STATE_DIRECTORY_RELATIVE_PATH) {
            return Err("A workspace already exists in this directory".to_string());
        }
        
        match self.filesystem.create_directory(STATE_DIRECTORY_RELATIVE_PATH) {
            true => println!("Created directory {}", STATE_DIRECTORY_RELATIVE_PATH),
            false => {
                return Err(format!("Couldn't create directory {}", STATE_DIRECTORY_RELATIVE_PATH));
            }
        }

        let workspace = Workspace::new();
        match self.filesystem.write_file(STATE_DIRECTORY_RELATIVE_PATH, "workspace.toml", workspace.serialize()) {
            true => println!("Created workspace.toml"),
            false => {
                return Err("Couldn't create workspace.toml".to_string());
            }
        }
        return Ok(format!("Initialized new workspace in {}", self.filesystem.get_root_directory_path()));
    }

}

#[cfg(test)]
mod test_cli {
    use super::*;

}