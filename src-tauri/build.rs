// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
//! Build script: tauri-build reads tauri.conf.json and capabilities/, embeds the icons and generates the ACL schemas.
fn main() {
    tauri_build::build()
}
