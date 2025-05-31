// Tests go here; this will not be compiled when this package is used as an extension.

/**
 * Run the Example with Enhanced Testing
 */
function example() {
    game.consoleOverlay.setVisible(true);

    // Create a simple neural network with one layer
    let model = new Torch.Sequential([
        new Torch.Linear(1, 1), // Single neuron, single output
        new Torch.Linear(1,1)
    ]);

    // Generate training data
    let inputs: Torch.Tensor[] = [];
    let targets: Torch.Tensor[] = [];

    for (let i = 0; i <= 90; i++) {
        let inputValue = new Torch.Tensor([[i]]);
        let targetValue = new Torch.Tensor([[i + 2]]); // Output should be (X + 2)
        inputs.push(inputValue);
        targets.push(targetValue);
    }

    // Set training parameters
    let learningRate = 0.001;
    let epochs = 1500;

    // Train using different loss functions
    console.log("Training with MSE...");
    pause(0)
    model.train(inputs, targets, learningRate, epochs, Torch.relu, Torch.huber,true);

    // Testing trained models
    let testInputs = [new Torch.Tensor([[5]]), new Torch.Tensor([[8]]),new Torch.Tensor([[0]])];

    testInputs.forEach((testInput, index) => {
        let prediction = model.forward(testInput, Torch.relu);
        let expectedOutput = testInput.data[0][0] + 2;
        game.consoleOverlay.clear()
        console.log(`Test ${index + 1} - Input: ${testInput.data[0][0]}`);
        console.log(`Predicted Output: ${JSON.stringify(prediction.data[0][0])}`);
        console.log(`Error: ${Math.abs(prediction.data[0][0] - expectedOutput)}`);
        pause(1000)
    });
}

example();