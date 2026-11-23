// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
//! isoview desktop app: a native window (Tauri 2, the system webview) that serves the isoview site from files
//! embedded in the binary (../dist, staged from the repo by tools/stage-desktop.mjs), so it runs offline. All
//! parsing, orbitals, meshing and drawing stay in the page's JavaScript, WebAssembly, WebGPU or WebGL2, as on the web.
//! Rust adds one command, save_file, because system webviews do not all save `<a download>` links, a fix for blank
//! windows on Linux with NVIDIA drivers, and a smoke test for CI (ISOVIEW_SMOKE).

// WHY: without this a release build on Windows opens a console window next to the app window.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::ipc::{InvokeBody, Request};
use tauri_plugin_dialog::DialogExt;

/// Shows a native Save dialog with the suggested name from the `x-file-name` header (ASCII, from js/app.js
/// `download`) and writes the raw request body (the file's bytes) to the chosen path.
/// Returns Ok(true) when written, Ok(false) when the user cancelled, Err(message) otherwise.
#[tauri::command]
async fn save_file(app: tauri::AppHandle, request: Request<'_>) -> Result<bool, String> {
    let InvokeBody::Raw(bytes) = request.body() else {
        return Err("save_file expects the file's bytes as the request body".into());
    };
    let name = request
        .headers()
        .get("x-file-name")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("isoview");
    // WHY: blocking_save_file must not run on the main thread; async commands run on the async runtime instead.
    let Some(choice) = app.dialog().file().set_file_name(name).blocking_save_file() else {
        return Ok(false);
    };
    let path = choice.into_path().map_err(|e| e.to_string())?;
    std::fs::write(&path, bytes).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(true)
}

/// Smoke test for CI, run only when the environment variable ISOVIEW_SMOKE is set. Probes, each logged with its
/// time: fetching a sample file and the worker script directly; a module worker answering a message; then (after
/// 8 s) the "hydrogen 3d" sample (worker only, no file) and (after 16 s) the "Si model density" sample (file fetch
/// and worker). Toasts, page errors and every fetch are logged too; after 30 s a box shows the log with the
/// renderer, the space group and the web APIs present, so a screenshot shows what works.
const SMOKE_JS: &str = r#"(() => {
  const log = [], t0 = performance.now(), note = m => log.push(((performance.now() - t0) / 1000).toFixed(1) + " s  " + m);
  addEventListener("error", e => note("error: " + e.message));
  addEventListener("unhandledrejection", e => note("rejection: " + e.reason));
  const toast = document.getElementById("toast");
  if (toast) new MutationObserver(() => { const s = toast.textContent.trim(); if (s) note("toast: " + s); })
    .observe(toast, { childList: true, subtree: true, characterData: true });
  const f = window.fetch;
  window.fetch = (...a) => { note("fetch start " + a[0]); return f(...a).then(r => (note("fetch " + a[0] + ": " + r.status), r), e => { note("fetch " + a[0] + " failed: " + e); throw e; }); };
  fetch("examples/Si-model.CHGCAR").then(r => r.blob()).then(b => note("probe file: " + b.size + " B"), e => note("probe file failed: " + e));
  fetch("js/worker.js?v=24").then(r => r.text()).then(t => note("probe worker.js: " + t.length + " chars"), e => note("probe worker.js failed: " + e));
  try {
    const w = new Worker("js/worker.js?v=24", { type: "module" });
    w.onmessage = e => note("probe worker answered: " + JSON.stringify(e.data).slice(0, 80));
    w.onerror = e => note("probe worker error: " + (e.message || "could not start"));
    w.postMessage({ id: -1, op: "smoke-probe" });
  } catch (e) { note("probe worker threw: " + e); }
  const click = id => { const b = document.querySelector('[data-example="' + id + '"]'); note("click " + id + (b ? "" : " (no button)")); if (b) b.click(); };
  setTimeout(() => click("h3d"), 8000);
  setTimeout(() => { const t = id => ((document.getElementById(id) || {}).textContent || "").trim(); note("after h3d: renderer " + t("gfx") + ", title " + t("doc-name")); click("si"); }, 16000);
  setTimeout(() => {
    const t = id => ((document.getElementById(id) || {}).textContent || "").trim();
    const d = document.createElement("pre");
    d.style.cssText = "position:fixed;left:12px;bottom:12px;z-index:99999;margin:0;padding:8px 12px;background:#ffc;color:#000;font:12px monospace;white-space:pre-wrap;max-width:75%";
    d.textContent = ["smoke test", "renderer: " + t("gfx") + ", space group: " + t("doc-sg") + ", title: " + t("doc-name"),
      "webgl2: " + !!document.createElement("canvas").getContext("webgl2") + ", webgpu: " + !!navigator.gpu +
      ", MediaRecorder: " + (typeof MediaRecorder !== "undefined") + ", indexedDB: " + (typeof indexedDB !== "undefined"),
      ...log].join("\n");
    document.body.append(d);
  }, 30000);
})();"#;

fn main() {
    // WHY: WebKitGTK's DMA-BUF renderer gives a blank window with NVIDIA's proprietary driver; turn it off only there
    // (and only if the user has not chosen a value), so other Linux systems keep the faster path.
    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/proc/driver/nvidia/version").exists()
            && std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none()
        {
            std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
        }
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .on_page_load(|webview, payload| {
            if payload.event() == tauri::webview::PageLoadEvent::Finished
                && std::env::var_os("ISOVIEW_SMOKE").is_some()
            {
                let _ = webview.eval(SMOKE_JS);
            }
        })
        .invoke_handler(tauri::generate_handler![save_file])
        .run(tauri::generate_context!())
        .expect("isoview could not start");
}
