// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Utkarsh Singh
/**
 * The last scene, kept in this browser so that a reload brings it back. Nothing leaves the device.
 *
 * IndexedDB database "isoview" with two object stores: "files" (key = item key, value = the item's files as
 * [{ name, blob }], stored once when the item is added) and "state" (key "last": the scene description written by app.js after
 * changes: items, layers, edits, view settings, camera). Files are kept only while the scene's stored files stay
 * within LIMIT bytes in total; larger items are described in the state but must be opened again after a reload.
 * Every call resolves (to null or false) instead of failing when storage is unavailable (private windows, blocked
 * site data, test environments).
 */
export const LIMIT = 512 * 2 ** 20; // bytes of files kept per scene

let dbp = null;
function db() {
  if (dbp) return dbp;
  dbp = new Promise(res => {
    try {
      if (typeof indexedDB === "undefined") return res(null);
      const r = indexedDB.open("isoview", 1);
      r.onupgradeneeded = () => { r.result.createObjectStore("files"); r.result.createObjectStore("state"); };
      r.onsuccess = () => res(r.result);
      r.onerror = () => res(null);
      r.onblocked = () => res(null);
    } catch (e) { res(null); }
  });
  return dbp;
}

/** Run one request on a store; resolves to its result, or null on any failure. */
async function req(store, mode, fn) {
  const d = await db();
  if (!d) return null;
  return new Promise(res => {
    try {
      const tx = d.transaction(store, mode), r = fn(tx.objectStore(store));
      tx.oncomplete = () => res(r && "result" in r ? r.result ?? null : true);
      tx.onerror = tx.onabort = () => res(null);
    } catch (e) { res(null); }
  });
}

// WHY: stored as { name, blob } pairs so the file names (used to detect formats) survive every structured clone
export const putFiles = (key, files) => req("files", "readwrite", s => s.put(files.map(f => ({ name: f.name, blob: f })), key));
export async function getFiles(key) {
  const v = await req("files", "readonly", s => s.get(key));
  return v ? v.map(e => new File([e.blob], e.name)) : null;
}
export const putState = state => req("state", "readwrite", s => s.put(state, "last"));
export const getState = () => req("state", "readonly", s => s.get("last"));
/** Delete the stored files whose keys are not in `keep` (a Set). */
export async function pruneFiles(keep) {
  const keys = await req("files", "readonly", s => s.getAllKeys());
  if (!keys) return;
  for (const k of keys) if (!keep.has(k)) await req("files", "readwrite", s => s.delete(k));
}
/** Forget the scene and its files. */
export async function clearAll() {
  await req("files", "readwrite", s => s.clear());
  await req("state", "readwrite", s => s.clear());
}
