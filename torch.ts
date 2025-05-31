// torch.ts - Extended Neural Network Library for MakeCode Arcade

namespace Torch {
    export class Tensor {
        data: number[][];

        constructor(data: number[][]) {
            this.data = data;
        }

        matmul(other: Tensor): Tensor | null {
            let rowsA = this.data.length;
            let colsA = this.data[0].length;
            let rowsB = other.data.length;
            let colsB = other.data[0].length;

            if (colsA !== rowsB) {
                return null; // Dimension mismatch
            }

            // Manually initialize the result matrix
            let result: number[][] = [];
            for (let r = 0; r < rowsA; r++) {
                let row: number[] = [];
                for (let c = 0; c < colsB; c++) {
                    row.push(0);
                }
                result.push(row);
            }

            // Optimized matrix multiplication
            for (let r = 0; r < rowsA; r++) {
                for (let i = 0; i < colsA; i++) {
                    let value = this.data[r][i]; // Reduce lookup overhead
                    for (let c = 0; c < colsB; c++) {
                        result[r][c] += value * other.data[i][c];
                    }
                }
            }

            return new Tensor(result);
        }
        
        applyFunction(func: (x: number) => number): Tensor {
            return new Tensor(this.data.map(row => row.map(func)));
        }

        add(other: Tensor): Tensor {
            let rows = Math.min(this.data.length, other.data.length);
            let cols = Math.min(this.data[0].length, other.data[0].length);

            let result: number[][] = [];

            for (let r = 0; r < rows; r++) {
                let row: number[] = [];
                for (let c = 0; c < cols; c++) {
                    row.push(this.data[r][c] + other.data[r][c]);
                }
                result.push(row);
            }

            return new Torch.Tensor(result);
        }
        sub(other: Tensor): Tensor {
            let rows = Math.min(this.data.length, other.data.length);
            let cols = Math.min(this.data[0].length, other.data[0].length);

            let result: number[][] = [];

            for (let r = 0; r < rows; r++) {
                let row: number[] = [];
                for (let c = 0; c < cols; c++) {
                    row.push(this.data[r][c] - other.data[r][c]);
                }
                result.push(row);
            }

            return new Torch.Tensor(result);
        }
        sum(): number {
            let total = 0;
            for (let row of this.data) {
                for (let value of row) {
                    total += value;
                }
            }
            return total;
        }
    }

    export class Neuron {
        weights: number[];
        bias: number;

        constructor(inputSize: number) {
            this.weights = [];
            this.bias = Math.random() * 0.2 - 0.1;

            for (let i = 0; i < inputSize; i++) {
                this.weights.push(Math.random() * 0.2 - 0.1);
            }
        }

        activate(inputs: number[], activation: (x: number) => number): number {
            let sum = inputs.reduce((acc, val, i) => acc + val * this.weights[i], this.bias);
            return activation(sum);
        }
    }

    export function activationDerivative(x: number, activation: (x: number) => number): number {
        if (activation === Torch.sigmoid) {
            let sig = Torch.sigmoid(x);
            return isNaN(sig) ? 0 : sig * (1 - sig);
        }
        if (activation === Torch.relu) return x > 0 ? 1 : 0;
        if (activation === Torch.tanh) {
            let tanhX = Torch.tanh(x);
            return isNaN(tanhX) ? 0 : 1 - tanhX * tanhX;
        }
        return 1;
    }



    export class Linear {
        neurons: Neuron[];

        constructor(inDim: number, outDim: number) {
            this.neurons = [];
            for (let i = 0; i < outDim; i++) {
                let neuron = new Neuron(inDim)
                neuron.weights = neuron.weights.map(() => Math.random() * 0.1 - 0.05);
                neuron.bias = Math.random() * 0.1 - 0.05;
                this.neurons.push(neuron);
            }
        }

        forward(input: Tensor, activation: (x: number) => number): Tensor {
            let output: number[][] = [];
            for (let row of input.data) {
                let neuronOutputs: number[] = this.neurons.map(neuron => neuron.activate(row, activation));
                output.push(neuronOutputs);
            }
            return new Tensor(output);
        }

        backward(error: Tensor, learningRate: number, activation: (x: number) => number): Tensor {
            let activatedError = error.applyFunction(x => activationDerivative(x, activation));
            let newError: number[][] = [];
            for (let i = 0; i < this.neurons.length; i++) {
                newError.push([]); // Properly initializes an empty array
            }

            this.neurons.forEach((neuron, index) => {
                neuron.weights = neuron.weights.map((w, j) => {
                    let gradient = activatedError.data[0][index] * error.data[0][index];
                    newError[j].push(gradient); // Accumulate error for next layer
                    return w + learningRate * gradient;
                });

                neuron.bias += learningRate * activatedError.data[0][index];
            });

            return new Torch.Tensor(newError);
        }



        train(inputs: Tensor[], targets: Tensor[], learningRate: number, epochs: number, activation?: (x: number) => number,disablelogging?:boolean): void {
            if (!activation) activation = Torch.sigmoid; // Default activation function

            for (let epoch = 0; epoch < epochs; epoch++) {
                let totalLoss = 0;

                for (let i = 0; i < inputs.length; i++) {
                    let input = inputs[i];
                    let target = targets[i];

                    // Forward pass with activation function
                    let prediction = this.forward(input, activation);
                    let error = target.sub(prediction);

                    if (!error || !error.data || error.data.length === 0) {
                        console.log("Error tensor is invalid. Defaulting to zero tensor.");
                        error = new Torch.Tensor([[0]]);
                    }

                    // Compute Mean Squared Error (MSE)
                    let divisor = Math.max(1, error.data.length);
                    let loss = error.applyFunction(x => x * x).sum() / divisor;
                    if (isNaN(loss)) {
                        loss = 0
                    }
                    totalLoss += loss;
                    

                    // Compute activation derivative BEFORE weight updates
                    //error = error.applyFunction(x => isNaN(x) ? 0 : x);
                    let activatedError = error.applyFunction(x => activationDerivative(x, activation));

                    // Weight update using backpropagation
                    this.neurons.forEach((neuron, index) => {
                        neuron.weights = neuron.weights.map((w, j) => {
                            let gradient = activatedError.data[0][index] * input.data[0][j];
                            return w + learningRate * gradient;
                        });
                        let value = activatedError.data[0][index]
                        if (!isNaN(value)) {
                          neuron.bias += learningRate * value;
                        }
                    });
                }

                // Learning rate decay with a safety floor
                if (epoch % 200 === 0) {
                    learningRate = Math.max(learningRate * 0.95, 0.005);
                }
                if (disablelogging != true) {
                  console.log(`Epoch ${epoch + 1}, Loss: ${totalLoss / inputs.length}`);
                }
            }
        }
    }

    export class ConvLayer {
        kernel: Tensor;
        kernelGradients: Tensor; // Store gradients for backpropagation

        constructor(size: number) {
            // Explicitly initialize the kernel with randomized weights
            let kernelArray: number[][] = [];
            for (let i = 0; i < size; i++) {
                let row: number[] = [];
                for (let j = 0; j < size; j++) {
                    row.push(Math.random() * 0.2 - 0.1);
                }
                kernelArray.push(row);
            }
            this.kernel = new Tensor(kernelArray);
            this.kernelGradients = new Tensor(kernelArray.map(row => row.map(() => 0))); // Initialize gradients
        }

        // **Forward Pass**: Apply convolution to input tensor
        forward(input: Tensor): Tensor {
            return input.matmul(this.kernel);
        }

        // **Backpropagation**: Compute gradients for weight updates
        backward(error: Tensor, learningRate: number): void {
            // Compute gradients
            this.kernelGradients = error.matmul(this.kernel);

            // Update kernel weights using gradient descent
            this.kernel.data = this.kernel.data.map((row, r) => row.map((val, c) =>
                val - learningRate * this.kernelGradients.data[r][c]));
        }

        // **Training Step**
        train(inputs: Tensor[], targets: Tensor[], learningRate: number, epochs: number): void {
            for (let epoch = 0; epoch < epochs; epoch++) {
                let totalLoss = 0;

                for (let i = 0; i < inputs.length; i++) {
                    let input = inputs[i];
                    let target = targets[i];

                    // Forward pass
                    let prediction = this.forward(input);
                    let error = target.sub(prediction);

                    // Compute Mean Squared Error (MSE) loss
                    let loss = error.applyFunction(x => x * x).sum() / Math.max(1, error.data.length);
                    totalLoss += loss;

                    // Backward pass to adjust kernel weights
                    this.backward(error, learningRate);
                }

                console.log(`Epoch ${epoch + 1}, Loss: ${totalLoss / inputs.length}`);
            }
        }
    }

    export function mae(predictions: Tensor, targets: Tensor): number {
        let errorTensor = predictions.add(targets.applyFunction(x => -x));

        // **Fix unsupported `.flat()` method by manually summing**
        let totalError = 0;
        let elementCount = 0;
        for (let row of errorTensor.data) {
            for (let val of row) {
                totalError += Math.abs(val);
                elementCount++;
            }
        }
        return totalError / elementCount;
    }

    export function relu(x: number): number {
        return Math.max(0, x);
    }

    export function sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x));
    }

    export function elu(x: number, alpha: number = 1): number {
        return x > 0 ? x : alpha * (Math.exp(x) - 1);
    }

    export function hardSigmoid(x: number): number {
        return Math.max(0, Math.min(1, (0.2 * x + 0.5)));
    }

    export function bentIdentity(x: number): number {
        return (Math.sqrt(x * x + 1) - 1) / 2 + x;
    }

    export function mish(x: number): number {
        return x * tanh(Math.log(1 + Math.exp(x)));
    }

    export function swish(x: number): number {
        return x / (1 + Math.exp(-x)); // Uses sigmoid-like smoothing
    }

    export function gelu(x: number): number {
        return x * (1 + tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))) / 2;
    }

    // Tanh (Hyperbolic Tangent)
    export function tanh(x: number): number {
        let exp2x = Math.exp(2 * x);
        return (exp2x - 1) / (exp2x + 1);
    }

    // Leaky ReLU (Improved ReLU to prevent dead neurons)
    export function leakyRelu(x: number, alpha: number = 0.01): number {
        return x > 0 ? x : alpha * x;
    }

    // Softmax (Used for classification problems)
    export function softmax(inputs: number[]): number[] {
        let expValues = inputs.map(x => Math.exp(x));
        let sumExp = expValues.reduce((a, b) => a + b, 0);
        return expValues.map(x => x / sumExp);
    }

    export class Sequential {
        layers: (Linear | ConvLayer)[];

        constructor(layers: (Linear | ConvLayer)[]) {
            this.layers = layers;
        }

        // Forward pass through all layers
        forward(input: Tensor, activation: (x: number) => number): Tensor {
            let output = input;
            for (let layer of this.layers) {
                if (layer instanceof ConvLayer) {
                    output = layer.forward(output); // ConvLayer does not use an activation
                } else {
                    output = layer.forward(output, activation);
                }
            }
            return output;
        }

        // Train the model
        train(inputs: Tensor[], targets: Tensor[], learningRate: number, epochs: number, activation?: (x: number) => number): void {
            if (!activation) activation = Torch.sigmoid;

            for (let epoch = 0; epoch < epochs; epoch++) {
                let totalLoss = 0;

                for (let i = 0; i < inputs.length; i++) {
                    let input = inputs[i];
                    let target = targets[i];

                    // Forward pass
                    let prediction = this.forward(input, activation);
                    let error = target.sub(prediction);

                    // Compute Mean Squared Error (MSE)
                    let loss = error.applyFunction(x => x * x).sum() / Math.max(1, error.data.length);
                    totalLoss += loss;

                    // Backpropagation through all layers in reverse order
                    let previousError = error;
                    let reversedLayers = this.layers.slice(); // Clone and reverse layers
                    reversedLayers.reverse(); // Do the Revserse

                    reversedLayers.forEach(layer => {
                        if (layer instanceof ConvLayer) {
                            layer.backward(previousError, learningRate); // ConvLayer only adjusts kernels
                        } else {
                            previousError = layer.backward(previousError, learningRate, activation);
                        }
                    });
                }

                // Learning rate decay with a minimum limit
                if (epoch % 200 === 0) {
                    learningRate = Math.max(learningRate * 0.95, 0.005);
                }

                console.log(`Epoch ${epoch + 1}, Loss: ${totalLoss / inputs.length}`);
            }
        }
    }





}