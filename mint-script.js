const setTexts = () => {
  console.log("setTexts");
  const navigationItems = document.getElementById("navigation-items");

  const navigationH5 = navigationItems.querySelectorAll("h5");
  for (const h5 of navigationH5) {
    h5.setAttribute("data-text", h5.textContent);
  }
};

setTexts();

setInterval(() => {
  setTexts();
}, 150);
