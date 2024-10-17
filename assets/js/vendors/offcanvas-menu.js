const openButton = document.getElementById("offcanvas-open-btn");
const closeButton = document.getElementById("offcanvas-close-btn");
const offCanvas = document.getElementById("offcanvas-body");

openButton.addEventListener("click", () => {
  offCanvas.style.transform = "translateX(0)";
});

closeButton.addEventListener("click", () => {
  offCanvas.style.transform = "translateX(100%)";
});
