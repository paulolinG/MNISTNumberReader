import { initializeInputBoxes } from "./inputs.js";
import { initializeDrawGridButtons, trainModel } from "./button_logic.js";
import { fetchArchitecture } from "./cytoscape.js";

/*
 *  saves the neural network parameters to local storage
 *  "learning_rate" is the key for the learningRate
 *  "number_of_layers" is the key for the layer count
 */

async function saveParameters() {
    const learningRate = document.querySelector("#learning-rate").value;
    const numberOfLayers = document.querySelector("#number-of-layers").value;
    const nodesPerLayer = document.querySelector("#nodes-per-layer").value;
    const layerConfig = [];

    if (!numberOfLayers) {
        return;
    }

    for (let i = 0; i < numberOfLayers; i++) {
        layerConfig.push(nodesPerLayer);
    }

    try {
        const response = await fetch(`/api/set_parameters`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                learning_rate: learningRate,
                layer_config: layerConfig,
            }),
        });

        if (!response.ok) {
            throw new Error("Could not change network parameters");
        }

        await fetchArchitecture();
    } catch (error) {
        throw error;
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
            const touch = e.touches[0];
            const target = document.elementsFromPoint(
                touch.clientX,
                touch.clientY
            );
            if (target && target.classList.contains("pixel")) {
                togglePixel(target);
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

async function main() {
    initializeInputBoxes();
    document
        .querySelector("#save-parameters")
        .addEventListener("click", saveParameters);

    const drawingGrid = document.getElementById("drawing-grid");
    const clearButton = document.getElementById("clear-btn");
    const submitButton = document.getElementById("submit-btn");
    const trainButton = document.getElementById("train-model");

    initializeDrawBox(drawingGrid);
    initializeDrawGridButtons(clearButton, submitButton);

    trainButton.addEventListener("click", trainModel);

    await fetchArchitecture();
}

document.addEventListener("DOMContentLoaded", main);
