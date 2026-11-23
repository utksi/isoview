// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * Stages the site for the desktop app: copies the files the page loads (index.html, css/, js/, examples/, LICENSE, NOTICE)
 * into dist/, which src-tauri/tauri.conf.json embeds as the app's frontend. An explicit list, so tests, tools,
 * src-tauri/target and node_modules never end up inside the app. Run from anywhere: node tools/stage-desktop.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const parts = ["index.html", "css", "js", "examples", "LICENSE", "NOTICE"];

fs.rmSync(dist, { recursive: true, force: true });
for (const p of parts) fs.cpSync(path.join(root, p), path.join(dist, p), { recursive: true });
console.log(`staged ${parts.join(", ")} into ${path.relative(root, dist)}/`);
