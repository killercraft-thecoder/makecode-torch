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
            return new Tensor(this.data.map((row, r) => row.map((val, c) => val + other.data[r][c])));
        }
        sub(other: Tensor): Tensor | null {
            let rows = this.data.length;
            let cols = this.data[0].length;

            // Check for dimension mismatch
            if (rows !== other.data.length || cols !== other.data[0].length) {
                return null; // Return null if dimensions don't match
            }

            // Perform element-wise subtraction
            let result: number[][] = [];
            for (let r = 0; r < rows; r++) {
                let row: number[] = [];
                for (let c = 0; c < cols; c++) {
                    row.push(this.data[r][c] - other.data[r][c]);
                }
                result.push(row);
            }

            return new Tensor(result);
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

    export class Linear {
        neurons: Neuron[];

        constructor(inDim: number, outDim: number) {
            this.neurons = [];
            for (let i = 0; i < outDim; i++) {
                this.neurons.push(new Neuron(inDim));
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

        train(inputs: Tensor[], targets: Tensor[], learningRate: number, epochs: number): void {
            for (let epoch = 0; epoch < epochs; epoch++) {
                let totalLoss = 0;

                for (let i = 0; i < inputs.length; i++) {
                    let input = inputs[i];
                    let target = targets[i];

                    // Forward pass
                    let prediction = this.forward(input, relu);
                    let error = target.sub(prediction); // Direct subtraction saves function calls

                    // Compute loss
                    totalLoss += mae(prediction, target);

                    // Weight update (vectorized approach)
                    this.neurons.forEach((neuron, index) => {
                        neuron.weights = neuron.weights.map((w, j) => w + learningRate * error.data[0][index] * input.data[0][j]);
                        neuron.bias += learningRate * error.data[0][index];
                    });
                }

                console.log(`Epoch ${epoch + 1}, Loss: ${totalLoss / inputs.length}`);
            }
        }
    }

    export class ConvLayer {
        kernel: Tensor;

        constructor(size: number) {
            // **Fix unsupported `.fill().map()` usage with explicit loops**
            let kernelArray: number[][] = [];
            for (let i = 0; i < size; i++) {
                let row: number[] = [];
                for (let j = 0; j < size; j++) {
                    row.push(Math.random() * 0.2 - 0.1);
                }
                kernelArray.push(row);
            }
            this.kernel = new Tensor(kernelArray);
        }

        apply(input: Tensor): Tensor {
            return input.matmul(this.kernel);
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
}