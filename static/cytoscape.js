let cyInstance = null;

function renderArchitecture(architecture) {
    const cyContainer = document.getElementById(`neural-network-vis`);
    const elements = {
        nodes: [],
        edges: [],
    };

    const layers = architecture.layers;
    const totalLayers = layers.length;
    const neuronSpacing = 20;
    const layerSpacing = 20;

    // Iterate through each layer to create neuron nodes
    layers.forEach((layer, layerIndex) => {
        const neurons = layer.num_neurons;
        for (let neuronIndex = 0; neuronIndex < neurons; neuronIndex++) {
            const nodeId = `layer${layer.layer_number}_neuron${neuronIndex}`;
            elements.nodes.push({
                data: {
                    id: nodeId,
                    label: `L${layer.layer_number}N${neuronIndex}`,
                    type: layer.layer_type,
                    layerNumber: layer.layer_number,
                },
                position: {
                    x: layerIndex * layerSpacing,
                    y: (neuronIndex + 1) * neuronSpacing,
                },
            });
        }
    });

    for (let layerIndex = 0; layerIndex < totalLayers - 1; layerIndex++) {
        const currentLayer = layers[layerIndex];
        const nextLayer = layers[layerIndex + 1];
        const currentNeurons = currentLayer.num_neurons;
        const nextNeurons = nextLayer.num_neurons;

        const weights = currentLayer.weights;

        for (
            let currentNeuronIndex = 0;
            currentNeuronIndex < currentNeurons;
            currentNeuronIndex++
        ) {
            for (
                let nextNeuronIndex = 0;
                nextNeuronIndex < nextNeurons;
                nextNeuronIndex++
            ) {
                const sourceId = `layer${currentLayer.layer_number}_neuron${currentNeuronIndex}`;
                const targetId = `layer${nextLayer.layer_number}_neuron${nextNeuronIndex}`;
                const weightValue =
                    weights[currentNeuronIndex][nextNeuronIndex] || 0;

                elements.edges.push({
                    data: {
                        source: sourceId,
                        target: targetId,
                        weight: weightValue,
                    },
                });
            }
        }
    }

    if (cyInstance != null) {
        cyInstance.destroy();
    }

    cyInstance = cytoscape({
        container: cyContainer,
        elements: elements,
        style: [
            {
                selector: "node",
                style: {
                    label: "data(label)",
                    "text-valign": "center",
                    color: "#fff",
                    "background-color": "#007BFF",
                    width: "120px",
                    height: "60px",
                    "font-size": "14px",
                    "text-wrap": "wrap",
                    shape: "roundrectangle",
                },
            },
            {
                selector: 'node[type="Input"]',
                style: {
                    "background-color": "#ffc107", // Yellow for Input layer
                    shape: "ellipse",
                },
            },
            {
                selector: 'node[type="Output"]',
                style: {
                    "background-color": "#28a745", // Green for Output layer
                    shape: "diamond",
                },
            },
            {
                selector: "edge",
                style: {
                    width: 2,
                    "line-color": "#ccc",
                    "target-arrow-color": "#ccc",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                },
            },
        ],
        layout: {
            name: "breadthfirst",
            directed: true,
            padding: 10,
        },
    });

    cyInstance.fit();
    cyInstance.center();

    cyInstance.on("tap", "node", function (evt) {
        const node = evt.target;
        alert(`Layer: ${node.data("label")}`);
    });
}

async function fetchArchitecture() {
    try {
        const response = await fetch(`/api/architecture`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const architecture = await response.json();

        renderArchitecture(architecture);
    } catch (error) {
        console.error(`Error fetching architecture`, error);
    }
}

export { fetchArchitecture };
