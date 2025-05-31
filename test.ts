// tests go here; this will not be compiled when this package is used as an extension.
/** 
 * RUn the Example
*/
function example() {

    game.consoleOverlay.setVisible(true);

    // Create a basic neural network with one layer
    let model = new Torch.Sequential([
        new Torch.Linear(1, 1) // Single neuron, single output
    ]);

    // Generate training data
    let inputs: Torch.Tensor[] = [];
    let targets: Torch.Tensor[] = [];

    for (let i = 0; i <= 10; i++) {
        let inputValue = new Torch.Tensor([[i]]);
        let targetValue = new Torch.Tensor([[i]]); // Output should match input (X = X)
        inputs.push(inputValue);
        targets.push(targetValue);
    }

    // Train the model
    let learningRate = 0.1;
    let epochs = 500; // Should converge quickly since it's a simple problem
    model.train(inputs, targets, learningRate, epochs, Torch.relu);

    // Test the trained model
    let testInput = new Torch.Tensor([[5]]); // Expected output: ~5
    let prediction = model.forward(testInput, Torch.relu);
    console.log("Predicted Output: " + JSON.stringify(prediction.data));
    console.log("Error:" + (prediction.data[0][0] - 5))

}
example()