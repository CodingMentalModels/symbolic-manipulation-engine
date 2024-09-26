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

    pub fn remove_file(&self, path: &str, filename: &str) -> bool {
        let dir_path = self.root_directory.join(path);
        let path = dir_path.join(filename);
        if !path.exists() {
            return false;
        }
        fs::remove_file(path).is_ok()
    }

    pub fn remove_directory(&self, path: &str) -> bool {
        let path = self.root_directory.join(path);
        if !path.exists() {
            return false;
        }
        fs::remove_dir_all(path).is_ok()
    }

    pub fn ls(&self, path: &str) -> Result<Vec<String>, String> {
        let path = self.root_directory.join(path);

        let entries = fs::read_dir(path)
            .map_err(|e| format!("Unable to read directory.  Error: {:?}", e).to_string())?;

        let mut filenames = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|_| "Unable to read an entry.".to_string())?;
            let path = entry.path();

            if path.is_file() {
                let filename = match path.file_name() {
                    Some(f) => f,
                    None => return Err("Unable to get file name.".to_string()),
                };
                filenames.push(filename.to_string_lossy().into_owned());
            }
        }

        Ok(filenames)
    }

    pub fn copy_file(&mut self, path: &str, from_name: &str, to_name: &str) -> Result<(), String> {
        let contents = self.read_file(path, from_name)?;
        self.write_file(path, to_name, contents, false)
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
        fs::read_to_string(path).map_err(|e| format!("Couldn't read file: {:?}", e).to_string())
    }
}

#[cfg(test)]
mod test_filesystem {
    use super::*;
}
