/**
 * Cloudera app color mode: flips `data-mode` on <html> (cldr-design-template, verbatim mechanism —
 * same storage key + attrs, so preference carries across org apps). Theme stays
 * `data-theme="keiretsu"`; tokens adapt via theme-keiretsu.css (theme-cloudera.css reserved for a future settings page).
 *
 * Aegir extension: `applyColorMode` also dispatches `cldr:color-mode` and `useColorMode()` exposes
 * the mode to React via useSyncExternalStore — several consumers react independently (antd
 * algorithm bridge, the live-viz embed params, xyflow SVG literals) rather than one owner
 * prop-drilling as in the template's single Dashboard.
 */
import { useSyncExternalStore } from "react";

export type ColorMode = "dark" | "light";

const STORAGE_KEY = "cldr-color-mode";
const EVENT = "cldr:color-mode";

export function getColorMode(): ColorMode {
  if (typeof document === "undefined") return "dark";
  const attr = document.documentElement.getAttribute("data-mode");
  return attr === "light" ? "light" : "dark";
}

export function applyColorMode(mode: ColorMode): void {
  document.documentElement.setAttribute("data-mode", mode);
  try {
    localStorage.setItem(STORAGE_KEY, mode);
  } catch {
    /* private mode / blocked storage */
  }
  window.dispatchEvent(new CustomEvent(EVENT, { detail: mode }));
}

/** Restore saved preference before first paint when possible. */
export function initColorMode(): ColorMode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark") {
      applyColorMode(stored);
      return stored;
    }
  } catch {
    /* ignore */
  }
  const current = getColorMode();
  applyColorMode(current);
  return current;
}

export function toggleColorMode(current: ColorMode): ColorMode {
  const next: ColorMode = current === "dark" ? "light" : "dark";
  applyColorMode(next);
  return next;
}

function subscribe(cb: () => void): () => void {
  window.addEventListener(EVENT, cb);
  return () => window.removeEventListener(EVENT, cb);
}

/** The live color mode as React state — re-renders consumers on toggle. */
export function useColorMode(): ColorMode {
  return useSyncExternalStore(subscribe, getColorMode, () => "dark");
}
