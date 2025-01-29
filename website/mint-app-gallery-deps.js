// Load jQuery if not present
if (typeof jQuery === "undefined") {
  const jqueryScript = document.createElement("script");
  jqueryScript.src =
    "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js";
  document.head.appendChild(jqueryScript);

  jqueryScript.onload = function () {
    // Load Chosen after jQuery
    const chosenScript = document.createElement("script");
    chosenScript.src =
      "https://cdnjs.cloudflare.com/ajax/libs/chosen/1.8.7/chosen.jquery.min.js";
    document.head.appendChild(chosenScript);

    // Add Chosen CSS
    const chosenStyles = document.createElement("link");
    chosenStyles.rel = "stylesheet";
    chosenStyles.href =
      "https://cdnjs.cloudflare.com/ajax/libs/chosen/1.8.7/chosen.min.css";
    document.head.appendChild(chosenStyles);

    // Initialize Chosen when everything is loaded
    chosenScript.onload = function () {
      initializeGallerySelect();
    };
  };
}

function getTagsFromURL() {
  const searchParams = new URLSearchParams(window.location.search);
  const tags = searchParams.get("tags");
  return tags ? tags.split(",") : [];
}

// Function to initialize Chosen on gallery select
function initializeGallerySelect() {
  // Add flag to check if already initialized
  const selectElement = $(".examples-gallery-container .tag-filter");
  if (selectElement.length && !selectElement.data("chosen-initialized")) {
    selectElement
      .chosen({
        width: "100%",
        placeholder_text_multiple: "Filter by tags",
      })
      .data("chosen-initialized", true);

    // Get tags from URL and update chosen
    const urlTags = getTagsFromURL();
    if (urlTags.length > 0) {
      selectElement.val(urlTags);
      selectElement.trigger("chosen:updated");

      // Trigger the gallery:tagChange event with URL tags
      const customEvent = new CustomEvent("gallery:tagChange", {
        detail: urlTags,
      });
      document.dispatchEvent(customEvent);
    }

    // Set up change handler
    selectElement.on("change", function (evt, params) {
      const selectedValues = $(this).val() || [];
      const customEvent = new CustomEvent("gallery:tagChange", {
        detail: selectedValues,
      });
      document.dispatchEvent(customEvent);
    });
  }
}

// Debounce function
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// Debounced version of initialize
const debouncedInitialize = debounce(initializeGallerySelect, 100);

// Watch for URL changes using MutationObserver
const observer = new MutationObserver((mutations) => {
  if (window.location.pathname.includes("/use-cases")) {
    debouncedInitialize();
  }
});

observer.observe(document.body, {
  childList: true,
  subtree: true,
  attributes: false,
  characterData: false,
});

// Initialize on page load
document.addEventListener("DOMContentLoaded", function () {
  if (window.jQuery && window.jQuery.fn.chosen) {
    initializeGallerySelect();
  }
});
