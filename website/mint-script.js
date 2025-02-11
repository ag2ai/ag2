(function () {
  function updateClass() {
    document.body.classList.remove("reference-page");
    if (window.location.pathname.includes("/docs/api-reference/")) {
      document.body.classList.add("reference-page");
    }
  }

  // Handle initial load
  updateClass();

  // Handle URL changes
  window.addEventListener("popstate", updateClass);

  // Handle SPA navigation
  const pushState = history.pushState;
  history.pushState = function () {
    pushState.apply(history, arguments);
    updateClass();
  };
})();
