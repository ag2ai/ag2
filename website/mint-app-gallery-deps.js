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

// Function to initialize Chosen on gallery select
function initializeGallerySelect() {
  setTimeout(() => {
    $(".examples-gallery-container .tag-filter")
      .chosen({
        width: "100%",
        placeholder_text_multiple: "Filter by tags",
      })
      .on("change", function (evt, params) {
        const selectedValues = $(this).val() || [];
        // Dispatch custom event with selected values
        const customEvent = new CustomEvent("gallery:tagChange", {
          detail: selectedValues,
        });
        document.dispatchEvent(customEvent);
      });
  }, 500);
}

// Initialize on page load and after dynamic content loads
document.addEventListener("DOMContentLoaded", function () {
  if (window.jQuery && window.jQuery.fn.chosen) {
    initializeGallerySelect();
  }
});
