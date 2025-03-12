document.addEventListener("DOMContentLoaded", function () {
  // Check if the URL contains '/docs/blog'
  if (window.location.pathname.includes("/docs/blog")) {
    let currentPath = window.location.pathname;
    // Check if the URL ends with '/index'
    if (currentPath.endsWith("/index")) {
      // Remove the trailing '/index'
      currentPath = currentPath.slice(0, -6);
    }

    // Convert hyphens to slashes in the date portion
    // Looking for pattern like: /docs/blog/YYYY-MM-DD-Title
    const regex = /\/docs\/blog\/(\d{4})-(\d{2})-(\d{2})-(.*)/;
    if (regex.test(currentPath)) {
      currentPath = currentPath.replace(regex, "/docs/blog/$1/$2/$3/$4");

      // Create the new URL with the transformed path
      const newUrl = window.location.origin + currentPath;

      // Redirect to the new URL
      window.location.href = newUrl;
    }
  }
});
