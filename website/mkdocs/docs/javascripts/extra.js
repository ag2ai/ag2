// Load dependencies and initialize gallery
(function () {
  // Flag to track initialization status
  let initialized = false;

  // Helper function to load script and return a promise
  function loadScript(url) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = url;
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  // Helper function to load CSS
  function loadCSS(url) {
    return new Promise((resolve) => {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = url;
      link.onload = resolve;
      document.head.appendChild(link);
      // Don't reject on error, CSS isn't crucial for functionality
      link.onerror = resolve;
    });
  }

  // Load all required dependencies
  async function loadDependencies() {
    // Only load jQuery if it's not already available
    if (typeof jQuery === "undefined") {
      await loadScript(
        "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"
      );
    }

    // Only load Chosen if it's not already available
    if (typeof jQuery.fn.chosen === "undefined") {
      await Promise.all([
        loadScript(
          "https://cdnjs.cloudflare.com/ajax/libs/chosen/1.8.7/chosen.jquery.min.js"
        ),
        loadCSS(
          "https://cdnjs.cloudflare.com/ajax/libs/chosen/1.8.7/chosen.min.css"
        ),
      ]);
    }

    // Initialize gallery only after all dependencies are loaded
    initGallery();
  }

  function getTagsFromURL() {
    const searchParams = new URLSearchParams(window.location.search);
    const tags = searchParams.get("tags");
    return tags ? tags.split(",") : [];
  }

  // Main gallery initialization
  function initGallery() {
    // Prevent multiple initializations
    if (initialized) return;
    initialized = true;

    console.log("Initializing gallery...");

    // Add click event listeners to all cards
    document.querySelectorAll(".card").forEach(function (card) {
      card.addEventListener("click", function (e) {
        // Don't trigger if user clicked on a link or tag inside the card
        if (!e.target.closest("a") && !e.target.closest(".tag")) {
          const link = this.getAttribute("data-link");
          const link_target = this.getAttribute("data-link-target");
          if (link && link !== "#") {
            window.open(link, link_target);
          }
        }
      });

      // Add cursor pointer style to show cards are clickable
      card.style.cursor = "pointer";
    });

    // Initialize tag filtering
    initGalleryTagFiltering();
  }

  // Tag filtering functionality
  function initGalleryTagFiltering() {
    console.log("Initializing tag filtering...");

    try {
      // Double-check that jQuery and Chosen are available
      if (typeof jQuery === "undefined") {
        console.error("jQuery is not loaded!");
        return;
      }

      if (typeof jQuery.fn.chosen === "undefined") {
        console.error("Chosen plugin is not loaded!");
        return;
      }

      // Initialize chosen.js for the select element
      jQuery(".tag-filter").chosen({
        width: "100%",
        allow_single_deselect: true,
        placeholder_text_multiple: "Filter by tags",
      });

      console.log("Chosen initialized successfully");

      // Handle select change events
      jQuery(".tag-filter").on("change", function () {
        const selected = jQuery(this).val() || [];
        handleTagChange(selected);
      });

      // Handle clicks on tag spans within cards
      document.querySelectorAll(".tag").forEach((tag) => {
        tag.addEventListener("click", function (e) {
          e.preventDefault();
          e.stopPropagation();

          const tagValue = this.getAttribute("data-tag");

          // Get currently selected tags
          let selectedTags = jQuery(".tag-filter").val() || [];
          if (!selectedTags.includes(tagValue)) {
            selectedTags.push(tagValue);
            jQuery(".tag-filter").val(selectedTags).trigger("chosen:updated");
            handleTagChange(selectedTags);
          }
        });
      });

      // Get initial tags from URL if any
      const initialTags = getTagsFromURL();
      if (initialTags.length > 0) {
        // Set initial values based on URL
        jQuery(".tag-filter").val(initialTags).trigger("chosen:updated");
        handleTagChange(initialTags);
      }

      // Handle browser back/forward navigation
      window.addEventListener("popstate", () => {
        const tags = getTagsFromURL();
        jQuery(".tag-filter").val(tags).trigger("chosen:updated");
        handleTagChange(tags);
      });
    } catch (e) {
      console.error("Error initializing tag filtering:", e);
    }
  }

  // Filter gallery items by selected tags
  function handleTagChange(selectedTags) {
    // Update URL
    const searchParams = new URLSearchParams(window.location.search);
    if (selectedTags.length > 0) {
      searchParams.set("tags", selectedTags.join(","));
    } else {
      searchParams.delete("tags");
    }
    const searchParamsString = searchParams.toString();
    const newURL = searchParamsString
      ? `${window.location.pathname}?${searchParamsString}`
      : window.location.pathname;
    window.history.pushState({}, "", newURL);

    // Filter items
    const cards = document.querySelectorAll(".card");
    cards.forEach((card) => {
      const cardTags = card.getAttribute("data-tags").split(",");
      if (
        selectedTags.length === 0 ||
        selectedTags.some((tag) => cardTags.includes(tag))
      ) {
        card.style.display = "";
      } else {
        card.style.display = "none";
      }
    });
  }

  // Blog URL transformation (not related to gallery but keeping it for compatibility)
  function handleBlogURLs() {
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
  }

  // User Story URL transformation
  function handleUserStoryURLs() {
    // Check if the URL contains '/docs/user-stories'
    if (window.location.pathname.includes("/docs/user-stories")) {
      let currentPath = window.location.pathname;
      // Check if the URL ends with '/index'
      if (currentPath.endsWith("/index")) {
        // Remove the trailing '/index'
        currentPath = currentPath.slice(0, -6);

        // Create the new URL with the transformed path
        const newUrl = window.location.origin + currentPath;

        // Redirect to the new URL
        window.location.href = newUrl;
      }
    }
  }

  // Fix edit URLs
  function fixEditUrls() {
    // Find the edit link on the page
    const editLink = document.querySelector('a[title="Edit this page"]');

    // If edit link exists, check and modify if needed
    if (editLink) {
      // Hide edit link for API reference pages
      if (window.location.pathname.includes("/docs/api-reference/")) {
        console.log("Insede API reference page");
        editLink.classList.add("hide-edit-link");
      }

      const href = editLink.getAttribute("href");

      // Special case for notebooks
      if (
        href &&
        href.includes("/website/docs/use-cases/notebooks/notebooks/")
      ) {
        // Replace the path segment and change extension from .md to .ipynb
        const newHref = href
          .replace("/website/docs/use-cases/notebooks/notebooks/", "/notebook/")
          .replace(/\.md$/, ".ipynb");

        // Update the href attribute
        editLink.setAttribute("href", newHref);
      }

      // Handle blog urls
      else if (href && href.includes("/blog/posts/")) {
        const newHref = href
          .replace("/blog/posts/", "/_blogs/")
          .replace(/\.md$/, ".mdx");

        // Update the href attribute
        editLink.setAttribute("href", newHref);
      }

      // Regular case for other markdown files
      else if (href && href.endsWith(".md")) {
        // Replace .md with .mdx at the end
        const newHref = href.replace(/\.md$/, ".mdx");

        // Update the href attribute
        editLink.setAttribute("href", newHref);
      }
    }
  }

  // Initialize everything when the document is ready
  document.addEventListener("DOMContentLoaded", function () {
    handleBlogURLs();
    handleUserStoryURLs();
    loadDependencies();
    fixEditUrls();
  });

  // Watch for URL changes using MutationObserver
  const observer = new MutationObserver((mutations) => {
    if (window.location.pathname.includes("/use-cases") && !initialized) {
      loadDependencies();
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: false,
    characterData: false,
  });
})();
