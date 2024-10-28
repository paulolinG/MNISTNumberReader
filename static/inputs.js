function initializeInputBoxes() {
    const learningRateElement = document.querySelector("#learning-rate");
    const numberOfLayersElement = document.querySelector("#number-of-layers");
    const nodesPerLayerElement = document.querySelector("#nodes-per-layer");

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

    if (nodesPerLayerElement) {
        nodesPerLayerElement.addEventListener("input", (e) => {
            let value = e.target.value;

            let num = parseInt(value);

            if (!isNaN(num)) {
                if (num < 0) num = 1;
                if (num > 64) num = 64;

                e.target.value = num;
            } else {
                e.target.value = "";
            }
        });
    } else {
        console.error("Element with class 'learning-rate' not found.");
    }
}

function initializeDrawBox(drawingGrid) {
    const gridSize = 28 * 28;

    function createGrid() {
        for (let i = 0; i < gridSize; i++) {
            const pixel = document.createElement("div");
            pixel.classList.add("pixel");
            pixel.dataset.index = i;
            drawingGrid.appendChild(pixel);
        }
    }

    let isDrawing = false;

    function togglePixel(pixel) {
        pixel.classList.add("active");
    }

    function addEventListeners() {
        drawingGrid.addEventListener("mousedown", (e) => {
            if (e.target.classList.contains("pixel")) {
                isDrawing = true;
                togglePixel(e.target);
            }
        });

        drawingGrid.addEventListener("mouseover", (e) => {
            if (isDrawing && e.target.classList.contains("pixel")) {
                togglePixel(e.target);
            }
        });

        drawingGrid.addEventListener("touchmove", (e) => {
            e.preventDefault();
            if (isDrawing) {
                const touch = e.touches[0];
                const target = document.elementsFromPoint(
                    touch.clientX,
                    touch.clientY
                );
                if (target && target.classList.contains("pixel")) {
                    togglePixel(target);
                }
            }
        });

        document.addEventListener("mouseup", () => {
            isDrawing = false;
        });

        document.addEventListener("mouseleave", () => {
            isDrawing = false;
        });
    }

    createGrid();
    addEventListeners();
}

export { initializeInputBoxes, initializeDrawBox };
