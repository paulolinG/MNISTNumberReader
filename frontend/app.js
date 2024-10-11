// Removes the contents of the given DOM element (equivalent to elem.innerHTML = '' but faster)
function emptyDOM(elem) {
  while (elem.firstChild) elem.removeChild(elem.firstChild);
}

// Creates a DOM element from the given HTML string
function createDOM(htmlString) {
  let template = document.createElement("template");
  template.innerHTML = htmlString.trim();
  return template.content.firstChild;
}

function initializeInputBoxes() {
  const learningRateElement = document.querySelector(".learning-rate");
  const numberOfLayersElement = document.querySelector(".number-of-layers");

  if (learningRateElement) {
    learningRateElement.addEventListener("input", (e) => {
      let value = e.target.value;

      let num = parseInt(value);

      if (!isNaN(num)) {
        if (num < 0) num = 1;
        if (num > 999) num = 999;

        e.target.value = num;
      } else {
        e.target.value = "";
      }
    });
  } else {
    console.error("Element with class 'learning-rate' not found.");
  }

  if (numberOfLayersElement) {
    numberOfLayersElement.addEventListener("input", (e) => {
      let value = e.target.value;
      let num = parseInt(value, 10);

      if (!isNaN(num)) {
        if (num < 1) num = 1;
        if (num > 10) num = 10;

        num = Math.floor(num);
        e.target.value = num;
      } else {
        e.target.value = "";
      }
    });
  } else {
    console.error("Element with class 'number-of-layers' not found.");
  }
}

function main() {
  initializeInputBoxes();
}

document.addEventListener("DOMContentLoaded", main);
