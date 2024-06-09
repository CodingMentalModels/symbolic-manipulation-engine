use std::fs;
use std::path::PathBuf;

pub struct FileSystem {
    root_directory: PathBuf,
}

impl FileSystem {
    pub fn new(root_directory: PathBuf) -> Self {
        Self { root_directory }
    }

    pub fn get_root_directory_path(&self) -> String {
        self.root_directory.display().to_string()
    }

    pub fn is_directory(&self, path: &str) -> bool {
        let path = self.root_directory.join(path);
        path.is_dir()
    }

    pub fn is_file(&self, path: &str) -> bool {
        let path = self.root_directory.join(path);
        path.is_file()
    }

    pub fn path_exists(&self, path: &str) -> bool {
        let path = self.root_directory.join(path);
        path.exists()
    }

    pub fn create_directory(&self, path: &str) -> bool {
        let path = self.root_directory.join(path);
        if path.exists() {
            return false;
        }
        fs::create_dir(path).is_ok()
    }

    pub fn remove_directory(&self, path: &str) -> bool {
        let path = self.root_directory.join(path);
        if !path.exists() {
            return false;
        }
        fs::remove_dir_all(path).is_ok()
    }

    pub fn write_file(
        &self,
        path: &str,
        filename: &str,
        contents: String,
        overwrite: bool,
    ) -> Result<(), String> {
        let path = self.root_directory.join(path);
        if !path.exists() {
            return Err(format!("Path '{:?}' does not exist", path));
        }
        let path = path.join(filename);
        if path.exists() && !overwrite {
            return Err(format!("A file already exists at '{:?}'.", path));
        }
        fs::write(path, contents).map_err(|e| format!("Error writing file: {:?}", e).to_string())
    }

    pub fn read_file(&self, path: &str, filename: &str) -> Result<String, String> {
        let path = self.root_directory.join(path);
        if !path.exists() {
            return Err("Path doesn't exist".to_string());
        }
        let path = path.join(filename);
        fs::read_to_string(path).map_err(|_| "Couldn't read file".to_string())
    }
}

#[cfg(test)]
mod test_filesystem {
    use super::*;
}
