use std::fs;
use std::path::PathBuf;

pub struct FileSystem {
    root_directory: PathBuf,
}

impl FileSystem {

    pub fn new(root_directory: PathBuf) -> Self {
        Self {
            root_directory,
        }
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

    pub fn write_file(&self, path: &str, filename: &str, contents: String) -> bool {
        let path = self.root_directory.join(path);
        if !path.exists() {
            return false;
        }
        let path = path.join(filename);
        fs::write(path, contents).is_ok()
    }

}

#[cfg(test)]
mod test_filesystem {
    use super::*;

}