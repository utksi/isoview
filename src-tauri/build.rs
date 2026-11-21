//! Build script: tauri-build reads tauri.conf.json and capabilities/, embeds the icons and generates the ACL schemas.
fn main() {
    tauri_build::build()
}
