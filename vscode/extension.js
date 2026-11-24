// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * isoview for VS Code: the command "Open in isoview" (Explorer and editor-tab context menus, Command Palette) opens the
 * chosen files (several when several are selected) in a webview panel. The panel holds an iframe of the isoview site
 * in embedded mode (?embed&panel=1); the files are read with workspace.fs (so local, Remote SSH, WSL and container
 * workspaces all work), sent to the webview as bytes, and handed to the iframe by postMessage once it reports
 * { isoview: "ready" }: { isoview: "open", files: [{ name, data: Uint8Array }], mode: "replace" }. Nothing is
 * uploaded; the site's page runs in VS Code's webview and reads the bytes there.
 */
const vscode = require("vscode");

const SITE = "https://utkarsh.science/isoview/";
const ORIGIN = "https://utkarsh.science";

/** Registers the command. */
function activate(context) {
  context.subscriptions.push(vscode.commands.registerCommand("isoview.open", async (uri, uris) => {
    let list = Array.isArray(uris) && uris.length ? uris : uri instanceof vscode.Uri ? [uri] : null;
    if (!list) {
      const a = vscode.window.activeTextEditor;
      list = a ? [a.document.uri] : await vscode.window.showOpenDialog({ canSelectMany: true, openLabel: "Open in isoview" });
    }
    if (list && list.length) await openPanel(list);
  }));
}

/** The base name of a URI's path. */
const baseName = u => u.path.split("/").pop() || "file";

/** Opens a webview panel showing the given files in one isoview scene. */
async function openPanel(uris) {
  const files = await vscode.window.withProgress({ location: vscode.ProgressLocation.Window, title: "isoview: reading files" },
    () => Promise.all(uris.map(async u => ({ name: baseName(u), data: await vscode.workspace.fs.readFile(u) }))));
  const title = files.map(f => f.name).join(" + ");
  const panel = vscode.window.createWebviewPanel("isoview", title, vscode.ViewColumn.Active, { enableScripts: true, retainContextWhenHidden: true });
  const dark = [vscode.ColorThemeKind.Dark, vscode.ColorThemeKind.HighContrast].includes(vscode.window.activeColorTheme.kind);
  panel.webview.html = page(dark ? "dark" : "light");
  // the webview asks for the files once its iframe is ready (again after a reload of the iframe)
  panel.webview.onDidReceiveMessage(m => { if (m && m.want === "files") panel.webview.postMessage({ files }); });
}

/** HTML of the webview: a full-size iframe of the site and the relay between the extension and the iframe. */
function page(theme) {
  const nonce = Math.random().toString(36).slice(2) + Date.now().toString(36);
  return `<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; frame-src ${ORIGIN}; script-src 'nonce-${nonce}'; style-src 'unsafe-inline'">
<style>html,body{margin:0;padding:0;height:100%;overflow:hidden}iframe{border:0;width:100%;height:100vh;display:block}</style></head>
<body><iframe id="v" src="${SITE}?embed&panel=1&theme=${theme}" allow="fullscreen; clipboard-write" title="isoview"></iframe>
<script nonce="${nonce}">
const vs = acquireVsCodeApi(), f = document.getElementById("v");
window.addEventListener("message", e => {
  if (e.source === f.contentWindow && e.data && e.data.isoview === "ready") vs.postMessage({ want: "files" });
  else if (e.data && e.data.files && e.source !== f.contentWindow) f.contentWindow.postMessage({ isoview: "open", files: e.data.files, mode: "replace" }, "${ORIGIN}");
});
</script></body></html>`;
}

function deactivate() {}

module.exports = { activate, deactivate };
