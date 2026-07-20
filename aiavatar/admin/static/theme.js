(() => {
  const storageKey = "aiavatar-admin-theme";
  const media = window.matchMedia("(prefers-color-scheme: dark)");

  function storedTheme() {
    try {
      return localStorage.getItem(storageKey);
    } catch {
      return null;
    }
  }

  function applyTheme(theme, notify = false) {
    document.documentElement.dataset.theme = theme;
    if (notify) window.dispatchEvent(new CustomEvent("admin-theme-change", { detail: { theme } }));
  }

  applyTheme(storedTheme() || (media.matches ? "dark" : "light"));

  document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.querySelector("#theme-toggle");
    if (!toggle) return;

    function syncToggle() {
      const isDark = document.documentElement.dataset.theme === "dark";
      const label = `Switch to ${isDark ? "light" : "dark"} mode`;
      toggle.setAttribute("aria-pressed", String(isDark));
      toggle.setAttribute("aria-label", label);
      toggle.setAttribute("title", label);
    }

    toggle.addEventListener("click", () => {
      const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      try {
        localStorage.setItem(storageKey, theme);
      } catch {
        // Theme switching still works when storage is unavailable.
      }
      applyTheme(theme, true);
      syncToggle();
    });

    media.addEventListener("change", event => {
      if (storedTheme()) return;
      applyTheme(event.matches ? "dark" : "light", true);
      syncToggle();
    });

    syncToggle();
  });
})();
