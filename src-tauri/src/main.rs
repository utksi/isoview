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

/// Smoke test for CI, run only when the environment variable ISOVIEW_SMOKE is set: opens the "Si model density"
/// sample (worker, WebAssembly symmetry, marching cubes, drawing) and after 10 s shows a box with the renderer, the
/// space group, the web APIs present and any page errors, so a screenshot shows whether the app works.
const SMOKE_JS: &str = r#"(() => {
  const errs = [];
  addEventListener("error", e => errs.push(e.message));
  addEventListener("unhandledrejection", e => errs.push(String(e.reason)));
  const go = () => { const b = document.querySelector('[data-example="si"]'); if (b) b.click(); else setTimeout(go, 200); };
  go();
  setTimeout(() => {
    const t = id => ((document.getElementById(id) || {}).textContent || "").trim();
    const d = document.createElement("pre");
    d.style.cssText = "position:fixed;left:12px;bottom:60px;z-index:99999;margin:0;padding:8px 12px;background:#ffc;color:#000;font:13px monospace;white-space:pre-wrap;max-width:70%";
    d.textContent = ["smoke test", "renderer: " + t("gfx"), "space group: " + t("doc-sg"), "toast: " + t("toast"),
      "webgl2: " + !!document.createElement("canvas").getContext("webgl2") + ", webgpu: " + !!navigator.gpu +
      ", MediaRecorder: " + (typeof MediaRecorder !== "undefined") + ", indexedDB: " + (typeof indexedDB !== "undefined"),
      "errors: " + (errs.join("; ") || "none")].join("\n");
    document.body.append(d);
  }, 10000);
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
