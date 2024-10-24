function initializeDrawGridButtons(clearButton, submitButton) {
    function clearGrid() {
        const pixels = document.querySelectorAll(".pixel");
        pixels.forEach((pixel) => pixel.classList.remove("active"));
    }

    async function submitDrawing() {
        const pixelElements = document.querySelectorAll(".pixel");
        const input = [];

        pixelElements.forEach((pixel) => {
            input.push(pixel.classList.contains("active") ? 1 : 0);
        });

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ input: input }),
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.predictions) {
                const prediction = data.predictions.indexOf(
                    Math.max(...data.predictions)
                );
                predictionResult.textContent = `Predicted Digit: ${prediction}`;
            } else if (data.error) {
                predictionResult.textContent = `Error: ${data.error}`;
            }
        } catch (error) {
            console.error("Error submitting drawing:", error);
            predictionResult.textContent = "Error submitting drawing.";
        }
    }

    clearButton.addEventListener("click", clearGrid);
    submitButton.addEventListener("click", submitDrawing);
}

export { initializeDrawGridButtons };
