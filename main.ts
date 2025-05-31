game.consoleOverlay.setVisible(true)

// Define the neural network model using the Torch namespace
let model = new Torch.Linear(2, 1); // 2 input neurons, 1 output neuron

// Define training data (XOR-like problem)
let inputs = [
    new Torch.Tensor([[0, 0]]),
    new Torch.Tensor([[0, 1]]),
    new Torch.Tensor([[1, 0]]),
    new Torch.Tensor([[1, 1]])
];

let targets = [
    new Torch.Tensor([[0]]),
    new Torch.Tensor([[1]]),
    new Torch.Tensor([[1]]),
    new Torch.Tensor([[0]])
];

// Training the model
let learningRate = 0.1;
let epochs = 1000;
model.train(inputs, targets, learningRate, epochs);

// Testing the trained model
let testInput = new Torch.Tensor([[1, 0]]); // Expected output ~1
let prediction = model.forward(testInput, Torch.relu);
console.log("Predicted Output:" + JSON.stringify(prediction.data));
