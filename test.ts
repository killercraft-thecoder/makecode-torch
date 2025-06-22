// Tests go here; this will not be compiled when this package is used as an extension.

/**
 * Run the Example with Enhanced Testing
 */
function example() {
    game.consoleOverlay.setVisible(true);

    // Create a simple neural network with one layer
    let model = new Torch.Sequential([
        new Torch.Linear(1, 1) // Single neuron, single output
    ]);

    // Generate training data
    let inputs: Torch.Tensor[] = [];
    let targets: Torch.Tensor[] = [];

    for (let i = 0; i <= 1; i++) {
        let inputValue = new Torch.Tensor([[i]]);
        let targetValue = new Torch.Tensor([[i]]); // Output should be (X)
        inputs.push(inputValue);
        targets.push(targetValue);
    }

    // Set training parameters
    let learningRate = 0.001;
    let epochs = 1500;

    // Train using different loss functions
    console.log("\n\n\n")
    console.log("Training with MSE...");
    pause(0)
    model.train(inputs, targets, learningRate, epochs, Torch.relu);

    // Testing trained models
    let testInputs = [5,8,0];

    testInputs.forEach((testInput, index) => {
        let prediction = model.forward(new Torch.Tensor([[testInput]]),Torch.relu)
        let expectedOutput = testInput;
        game.consoleOverlay.clear()
        console.log("\n  \n  \n")
        console.log(`Test ${index + 1} - Input: ${testInput}`);
        console.log(`Predicted Output: ${JSON.stringify(prediction.data[0][0])}`);
        console.log(`Error: ${Math.abs(prediction.data[0][0] - expectedOutput)}`);
        console.log(`Wanted: ${testInput}`)
        pause(1000)
    });
}

if (game.ask("Trained Model test?")) {
example();
} else if (game.ask("matrix Mul Spedd Test?")) {
    game.consoleOverlay.setVisible(true);
    let result: number[][] = [];
    let base: number[] = []
    for (let c = 0; c < 100; c++) {
        base[c] = randint(1,100); // Fill with zeros
    }
    // then Copy and Paste the allocated Array
    for (let r = 0; r < 100; r++) {
        result[r] = base.slice(0)
    }
    for (let r = 0; r < 10;r++) {
    let tensor1 = new Torch.Tensor(result)
    let tensor2 = new Torch.Tensor(result)
    let start = control.micros()
    tensor1.matmul(tensor2)
    let end = control.micros()
    console.log("Time for Matrix Multiplactation for matrixs of size 100x100 with each matrix wit hrandom numbers from 0 -> 100:")
    console.log(((end - start) / 1000) + " Milliseconds")
    }
}

