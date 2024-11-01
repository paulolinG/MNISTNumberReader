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
                let predictions = [];
                for (const val of data.predictions) {
                    predictions.push(val[0]);
                }

                let maxVal = -Infinity;
                let predictionIndex = -1;
                for (let i = 0; i < predictions.length; i++) {
                    if (predictions[i] > maxVal) {
                        maxVal = predictions[i];
                        predictionIndex = i;
                    }
                }
                alert(`The number is ${predictionIndex}`);
            } else if (data.error) {
                console.log(data.error);
            }
        } catch (error) {
            console.error("Error submitting drawing:", error);
        }
    }

    clearButton.addEventListener("click", clearGrid);
    submitButton.addEventListener("click", submitDrawing);
}

async function trainModel() {
    try {
        const response = await fetch("/api/train", {
            method: "POST",
        });
        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }
        alert("Trained model sucessfully");
    } catch (error) {
        console.error(error);
    }
}

export { initializeDrawGridButtons, trainModel };
