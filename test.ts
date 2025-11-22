// Tests go here; this will not be compiled when this package is used as an extension.

/**
 * Run the Example with Enhanced Testing
 */
function example() {
    game.consoleOverlay.setVisible(true);

    // Create a simple neural network with one layer
    let model = new Torch.Sequential([
        new Torch.Linear(1, 1), // Single neuron, single output
    ]);

    // Generate training data
    let inputs: Torch.Tensor[] = [];
    let targets: Torch.Tensor[] = [];

    for (let i = -100; i <= 100; i++) {
        let inputValue = new Torch.Tensor([[i]]);
        let targetValue = new Torch.Tensor([[i*2]]); // Output should be (X*2)
        inputs.push(inputValue);
        targets.push(targetValue);
    }

    // Set training parameters
    let learningRate = 0.001;
    let epochs = 1000;

    // Train using different loss functions
    console.log("\n\n\n")
    console.log("Training with MSE...");
    pause(0)
    model.train(inputs, targets, learningRate, epochs, Torch.relu,Torch.mse,true);
    console.log(`Model Weights:${JSON.stringify((model.layers[0] as Torch.Linear).neurons[0].weights)}, Model Bias:${(model.layers[0] as Torch.Linear).neurons[0].bias}`)

    // Testing trained models
    let testInputs = [5,8,0,10,13,2];
    let errors:number[] = [];

    testInputs.forEach((testInput, index) => {
        let prediction = model.forward(new Torch.Tensor([[testInput]]),Torch.relu);
        let expectedOutput = testInput*2;
        game.consoleOverlay.clear();
        console.log("\n  \n  \n")
        console.log(`Test ${index + 1} - Input: ${testInput}`);
        console.log(`Predicted Output: ${JSON.stringify(prediction.data[0][0])}`);
        let error = Math.abs(prediction.data[0][0] - expectedOutput);
        errors.push(error)
        console.log(`Error: ${error}`);
        console.log(`Wanted: ${testInput*2}`);
        pause(1000);
    });
    let avg = new Torch.Tensor([errors]).sum() / errors.length // ahh yes cheating or not?
    pause(1000)
    game.consoleOverlay.clear();
    console.log(`Average Error:${avg}`)
}

example();
