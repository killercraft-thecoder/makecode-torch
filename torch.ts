// torch.ts - Extended Neural Network Library for MakeCode Arcade

/**
 * Torch: See https://github.com/killercraft-thecoder/makecode-torch/blob/master/README.md for more detials
 */
namespace Torch {
    export interface Shape {
        rows:number;
        columns:number;
    }
    /** A Activation Function */
    type ActivationFunction = Function
    /** A Generic Number Function */
    type Function = (x: number) => number
    /** A Error Function */
    type LossFunction = (error: TensorLike) => number

    /** A 2d Matrix */
    type Matrix = number[][]

    /** A `TensorLike` Object. */
    export interface TensorLike {
        data: Matrix
        shape:Shape
        round():TensorLike;
        floor():TensorLike;
        flat():number[];
        matmul(other: TensorLike): TensorLike | null
        applyFunction(func: (x: number) => number): TensorLike
        add(other: TensorLike): TensorLike
        sub(other: TensorLike): TensorLike
        sum(): number
    }
      
    /** the Interface for Standard Models. */
    export interface Model {
        forward(input: Tensor, activation: ActivationFunction): TensorLike;
        train(inputs: TensorLike[], targets: TensorLike[], learningRate: number, epochs: number, activation?: ActivationFunction, lossFunction?: LossFunction, silent?: boolean): void
        trainDataSet(DataSet: DataSet, learningRate: number, epochs: number, activation?: ActivationFunction, lossFunction?: LossFunction, disableLogging?: boolean): void
    }
    /** the Interaface for a `Layer` in a Stanard Model */
    export interface Layer extends Model {
        neurons: Neuron[];
        decay:number;
        outputSize: number;
        inputSize: number;
    }
    /** A `CNNLayer` use nstead of `Torch.Layer` For CNN Layers Only */
    export interface CNNLayer {
        /** The convolution kernel weights. */
        kernel: TensorLike;

        /** The gradient matrix used for updating the kernel during training. */
        kernelGradients: TensorLike;

        /** Forward pass: applies the layer's convolution operation. */
        forward(input: TensorLike): TensorLike;

        /** 
         * Backward pass: computes gradients and updates kernel weights.
         * @param error - The loss error propagated to this layer.
         * @param learningRate - Scalar controlling the weight update magnitude.
         */
        backward(error: TensorLike, learningRate: number): void;

        /**
         * Trains the convolutional layer on a set of input-target pairs.
         * @param inputs - A batch of input tensors.
         * @param targets - A batch of expected output tensors.
         * @param learningRate - Scalar value for gradient descent.
         * @param epochs - Number of training iterations.
         * @param activation - Optional activation function (defaults to ReLU).
         * @param lossFunction - Optional loss function (defaults to MSE).
         */
        train(
            inputs: TensorLike[],
            targets: TensorLike[],
            learningRate: number,
            epochs: number,
            activation?: ActivationFunction,
            lossFunction?: LossFunction
        ): void;
    }

    /** 
     * A Weight Equalizer
    */
    export class Equalizer {
        /** 
         * Equalizes a Models Weights closer to 0
        */
        static equalize(model:Linear | Sequential) {
            // Step 1: Figure Out the Type
            let ml = model as Linear
            if (ml.neurons && ml.neurons.length > 0) {
                // Linear Layer Type
                ml.neurons.forEach((a) => a.weights.forEach((a) => a /= 1 + (Math.abs(a) / 16)))
            } else {
                // Sequential Layer Type
                let ms = model as Sequential
                if (ms.layers) {
                  ms.layers.forEach((a) => Equalizer.equalize(a as Linear))
                } else {
                    // Invalid Model
                    throw "Invalid Model,Model niether valid Linear or Valid Sequential"
                }
            }
        }
    }

    /** Torch's Allocater */
    export class Allocator {

        /** Do Not Create A `Allocater` , it Will only cost memory */
        constructor() { }

        /** Allocates a 2d matrix */
        static allocateMatrix(rows: number, cols: number, defaultValue: number = 0): Matrix {
            let matrix: number[][] = [];

            for (let i = 0; i < rows; i++) {
                matrix[i] = []
                for (let j = 0; j < cols; j++) {
                    matrix[i][j] = defaultValue;
                }
            }

            return matrix;
        }
        /** Allocates a 1D standard array */
        static allocateArray(rows: number, defaultValue: number = 0): number[] {
            let array: number[] = [];

            for (let i = 0; i < rows; i++) {
                array[i] = defaultValue
            }

            return array
        }
    }

    /**
    * Represents a multi-dimensional tensor for matrix computations.
    */
    export class Tensor implements TensorLike {
        shape:Shape

        /** 
         * The Data Of the Tensor
        */
        data: Matrix;
        /**
        * Creates a new tensor from a 2D `Matrix`.
        * @param data A 2D `Matrix` representing the tensor values.
        */
        constructor(data: Matrix) {
            this.data = data;
            this.shape = {rows:data.length,columns:data[0].length}
        }

        /**
         * Returns a new tensor with each element rounded to the nearest integer.
         *
         * @returns {TensorLike} A tensor with rounded values.
         */
        round():TensorLike {
            return this.applyFunction((x) => Math.round(x))
        }

        /**
         * Returns a new tensor with each element rounded down to the nearest whole number.
         *
         * @returns {TensorLike} A tensor with floored values.
         */
        floor():TensorLike {
            return this.applyFunction((x) => Math.floor(x))
        }

        /**
         * Flattens the 2D tensor into a 1D array in row-major order.
         *
         * @returns {number[]} A flat array containing all elements of the tensor.
         */
        flat():number[] {
            let data:number[] = []
            this.data.forEach((a) => a.forEach((b) => data.push(b)))
            return data
        }

        /**
        * Performs matrix multiplication (A * B) and returns the resulting tensor.
        * @param other The tensor to multiply with.
        * @returns The resulting tensor, or `null` if dimensions do not match.
        */
        matmul(other: TensorLike): TensorLike | null {
            let temp1 = this.data; // Ensure a true copy
            let temp2 = other.data; // Prevent referencing original tensor
            let rowsA = temp1.length;
            let colsA = temp1[0].length;
            let rowsB = temp2.length;
            let colsB = temp2[0].length;

            if (colsA !== rowsB) {
                return null; // Dimension mismatch
            }

            let result: number[][] = [];

            // Optimized matrix multiplication
            for (let r = 0; r < rowsA; r++) { // Process row-wise first
                for (let i = 0; i < colsA; i++) {
                    let temp3 = temp1[r][i]; // Store lookup value for row
                    for (let c = 0; c < colsB; c++) {
                        result[r][c] += temp3 * temp2[i][c]; // Perform multiplication
                    }
                }
            }
            return new Tensor(result);
        }
        /**
        * Applies a function to every element in the tensor and returns a new transformed tensor.
        * @param func The function to apply to each tensor element.
        * @returns A new tensor with transformed values.
        */
        applyFunction(func: (x: number) => number): TensorLike {
            let data = this.data;
            let result = data.map(row => row.map(func)); // Direct transformation without extra storage
            return new Torch.Tensor(result);
        }

        /**
        * Adds another tensor element-wise and returns the resulting tensor.
        * @param other The tensor to add.
        * @returns The resulting tensor after addition.
        */
        add(other: TensorLike): TensorLike {
            let rows = Math.min(this.data.length, other.data.length);
            let cols = Math.min(this.data[0].length, other.data[0].length);

            // Manual preallocation to prevent dynamic resizing overhead
            let result: number[][] = [];
            let data1 = this.data;
            let data2 = other.data;
            // Optimized addition loop
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    result[r][c] = data1[r][c] + data2[r][c]; // Direct assignment avoids push overhead
                }
            }

            return new Torch.Tensor(result);
        }

        /**
        *Subtracts another tensor element-wise and returns the resulting tensor.
        * @param other The tensor to subtract.
        * @returns The resulting tensor after subtraction.
        */
        sub(other: TensorLike): TensorLike {
            let rows = Math.min(this.data.length, other.data.length);
            let cols = Math.min(this.data[0].length, other.data[0].length);

            // Manual array allocation without `new Array()`
            let result: number[][] = [];
            for (let r = 0; r < rows; r++) {
                result[r] = [];  // No `new Array()`, just an empty array
                for (let c = 0; c < cols; c++) {
                    result[r][c] = this.data[r][c] - other.data[r][c];
                }
            }

            return new Torch.Tensor(result);
        }

        /**
        * Computes the sum of all elements in the tensor.
        * @returns The sum of all tensor elements.
        */
        sum(): number {
            return this.data.reduce((acc: number, row: number[]) => acc + row.reduce((rowAcc: number, value: number) => rowAcc + value, 0), 0);
        }
    }
    /**
     * a Matrix of Inputs to a NN
     */
    type Inputs = Matrix
    /** 
     * A Matrix of Outputs from a NN
    */
    type Outputs = Matrix
    /** 
     * A DataSet of I/O pairs
    */
    export class DataSet {
        /** the `Inputs` of the `DataSet` */
        inputs: Inputs
        /** the `Outputs` of the `DataSet` */
        outputs: Outputs
        /** 
         * Create a new `DataSet`
        */
        constructor(inputs: Inputs, outputs: Outputs) {
            this.inputs = inputs
            this.outputs = outputs
        }
        /** 
         * Pushes a new Data Pair to the `DataSet`
        */
        push(dataPair: { input: number[], output: number[] }) {
            this.inputs.push(dataPair.input)
            this.outputs.push(dataPair.output)
        }
        /**
         * Pops the last data Pair from the `DataSet`
         */
        pop(): { input: number[], output: number[] } {
            return { input: this.inputs.pop(), output: this.outputs.pop() }
        }
        /** 
         * Concats a DataSet with this `DataSet` directly and returns the updated `DataSet`
        */
        concat(other: DataSet): DataSet {
            this.inputs = this.inputs.concat(other.inputs)
            this.outputs = this.outputs.concat(other.outputs)
            return this
        }
        /**
         * Returns a Part of the `DataSet` Starting from `start` and optinally ending at `end`
         */
        slice(start: number, end?: number): DataSet {
            return new DataSet(this.inputs.slice(start, end), this.outputs.slice(start, end))
        }
    }
    /** A Singluar `Neuron` */
    export class Neuron {
        weights: number[];
        bias: number;
        /** Create a new `Neuron` */
        constructor(inputSize: number) {
            this.bias = Math.random() * 0.3 - 0.1;
            this.weights = []; // Must explicitly declare it as an empty array

            for (let i = 0; i < inputSize; i++) {
                this.weights[i] = Math.random() * 0.3 - 0.1; // Assign values directly
            }
        }
        /** Activate this `Neuron` */
        activate(inputs: number[], activation: (x: number) => number): number {
            let sum = this.bias;
            let len = inputs.length; // Cache length
            for (let i = 0; i < len; i++) {
                sum += inputs[i] * this.weights[i];
            }
            return activation(sum);
        }

        weight_decay(strength: number = 0.999) {
            this.weights.forEach((a) => a *= strength)
        }
    }
    export class Constants {
        static sqrt2pi = Math.sqrt(2 / Math.PI);
        static coeff = 0.044715;
    }
    export function activationDerivative(x: number, activation: (x: number) => number): number {
        if (activation === Torch.sigmoid) {
            let sig = Torch.sigmoid(x);
            return isNaN(sig) ? 0 : sig * (1 - sig);
        }
        if (activation === Torch.relu) return x > 0 ? 1 : 0;
        if (activation === Torch.leakyRelu) return x > 0 ? 1 : 0.01; // Default α=0.01
        if (activation === Torch.tanh) {
            let tanhX = Torch.tanh(x);
            return isNaN(tanhX) ? 0 : 1 - tanhX * tanhX;
        }
        if (activation === Torch.elu) {
            return x > 0 ? 1 : Torch.elu(x) + 1; // ELU's derivative
        }
        if (activation === Torch.swish) {
            let sig = Torch.sigmoid(x);
            return sig + x * sig * (1 - sig); // Swish derivative
        }
        if (activation === Torch.mish) {
            let omega = 1 + Math.exp(x);
            let delta = omega + omega * omega;
            return (Math.exp(x) * delta) / (delta * delta); // Mish derivative
        }
        if (activation === Torch.gelu) {
            let x3 = x * x * x;
            let tanhTerm = tanh(Constants.sqrt2pi * (x + Constants.coeff * x3));
            return 0.5 * (1 + tanhTerm + x * Constants.sqrt2pi * (1 - tanhTerm * tanhTerm) * (1 + 3 * Constants.coeff * x * x));
        }

        return 1; // Default case (linear activation)
    }
    /**
    * Represents a fully connected layer (Linear layer) in a neural network.
    */
    export class Linear implements Layer {
        neurons: Neuron[];
        outputSize:number;
        inputSize:number;
        private _decay = 0.999;

        /**
        * Initializes a linear layer with random weights and biases.
        * @param inDim The number of input dimensions (neurons in the previous layer).
        * @param outDim The number of output dimensions (neurons in this layer).
        */
        constructor(inDim: number, outDim: number) {
            this.neurons = [];
            this.inputSize = inDim;
            this.outputSize = outDim;
             
            for (let i = 0; i < outDim; i++) {
                let neuron = new Neuron(inDim)
                neuron.weights = neuron.weights.map(() => Math.random() * 0.5 - 0.25);
                neuron.bias = Math.random() * 0.5 - 0.25;
                this.neurons.push(neuron);
            }
        }
        /** 
         * Set the Weight Decay to a Value to have each neurons weights be multplied by
        */
        set decay(x: number) {
            if (x > 1) {
                return;
            }
            this._decay = x
        }
        /**
        * Get the Weight Decay to a Value Which will have each neurons weights be multplied by
        */
        get decay(): number {
            return this._decay
        }

        /**
        * Performs a forward pass, computing the output of the layer given an input tensor.
        * @param input The input tensor.
        * @param activation The activation function to apply to each neuron's output.
        * @returns The resulting tensor after applying the layer transformation.
        */
        forward(input: TensorLike, activation: (x: number) => number): Tensor {
            let output: number[][] = [];
            for (let row of input.data) {
                let neuronOutputs: number[] = this.neurons.map(neuron => neuron.activate(row, activation));
                output.push(neuronOutputs);
            }
            return new Tensor(output);
        }
        /**
        * Performs backpropagation to adjust weights based on the error tensor.
        * @param error The error tensor (difference between actual and predicted values).
        * @param learningRate The rate at which weights are adjusted.
        * @param activation The activation function to apply to the error.
        * @param lossFunction Optional loss function to modify weight updates.
        * @returns A tensor containing the propagated error for the previous layer.
        */
        backward(error: TensorLike, learningRate: number, activation: (x: number) => number,
            lossFunction?: (error: TensorLike) => number): TensorLike {
            if (error.data.length !== this.outputSize) {
                error.data = error.data.slice(0,this.outputSize); // ReShape:
            }


            // Apply activation derivative
            let activatedError = error.applyFunction(x => activationDerivative(x, activation));
            let newError: number[][] = [];

            for (let i = 0; i < this.neurons.length; i++) {
                newError.push([0]); // Properly initializes an empty array
            }
            this.neurons.forEach((a) => a.weight_decay(this._decay))
            this.neurons.forEach((neuron, index) => {
                neuron.weights = neuron.weights.map((w, j) => {
                    let gradient = activatedError.data[0][index] * error.data[0][index];

                    // Apply loss function if provided
                    if (lossFunction && gradient) {
                        gradient *= lossFunction(error);
                    }
                    if (gradient) {
                        newError[j].push(gradient); // Accumulate error for next layer
                    } else {
                        newError[j].push(0);
                    }
                    return w + learningRate * gradient;
                });

                neuron.bias += learningRate * activatedError.data[0][index];
            });

            return new Torch.Tensor(newError);
        }
        /**
       * Trains the layer using multiple input-target pairs over several epochs.
       * @param inputs Array of input tensors.
       * @param targets Array of expected output tensors.
       * @param learningRate The learning rate for weight adjustments.
       * @param epochs Number of training iterations.
       * @param activation Activation function for training (default: sigmoid).
       * @param lossFunction Loss function to optimize weight updates (default: MSE).
       * @param disableLogging If true, disables console logs for training progress.
       */
        train(inputs: TensorLike[], targets: TensorLike[], learningRate: number, epochs: number,
            activation?: ActivationFunction, lossFunction?: LossFunction, disableLogging?: boolean): void {

            if (!activation) activation = Torch.sigmoid; // Default activation function
            if (!lossFunction) lossFunction = (error) => error.applyFunction(x => x * x).sum() / Math.max(1, error.data.length); // Default to MSE

            for (let epoch = 0; epoch < epochs; epoch++) {
                let totalLoss = 0;
                this.neurons.forEach((a) => a.weight_decay(this._decay))
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

                    // Compute loss using the provided function
                    let loss = lossFunction(error);
                    if (isNaN(loss)) {
                        loss = 0;
                    }
                    totalLoss += loss;

                    // Compute activation derivative BEFORE weight updates
                    let activatedError = error.applyFunction(x => activationDerivative(x, activation));

                    // Weight update using backpropagation
                    this.neurons.forEach((neuron, index) => {
                        neuron.weights = neuron.weights.map((w, j) => {
                            let gradient = activatedError.data[0][index] * input.data[0][j];
                            return w + learningRate * gradient;
                        });
                        let value = activatedError.data[0][index];
                        if (!isNaN(value)) {
                            neuron.bias += learningRate * value;
                        }
                    });
                }

                // Learning rate decay with a safety floor
                if (epoch % 200 === 0) {
                    learningRate = Math.max(learningRate * 0.95, 0.005);
                }

                if (!disableLogging) {
                    console.log(`Epoch ${epoch + 1}, Loss: ${totalLoss / inputs.length}`);
                }
            }
        }
        /** 
         * `Internal Use Only`
         * Tensorizes a 2d matrix -> Tensor array
        */
        protected tensorize(data: number[][]): Tensor[] {
            let tensors: Tensor[] = []
            data.forEach((a) => tensors.push(new Tensor([a])))
            return tensors
        }
        /**
        * Trains a dataset using multiple input-target pairs over several epochs.
        * @param DataSet The dataset containing input-output pairs.
        * @param learningRate The learning rate for weight adjustments.
        * @param epochs Number of training iterations.
        * @param activation Optional activation function for training (default: sigmoid).
        * @param lossFunction Optional loss function for optimizing weight updates (default: MSE).
        * @param disableLogging If true, disables console logs for training progress.
        */
        trainDataSet(DataSet: DataSet, learningRate: number, epochs: number, activation?: ActivationFunction, lossFunction?: LossFunction, disableLogging?: boolean) {
            this.train(this.tensorize(DataSet.inputs), this.tensorize(DataSet.outputs), learningRate, epochs, activation, lossFunction, disableLogging)
        }
    }

    export function arrayToTensor1D(data: number[]): Tensor {
        return new Tensor([data])
    }

    export class ConvLayer implements CNNLayer {
        kernel: Tensor;
        kernelGradients: Tensor; // Store gradients for backpropagation
        private _decay = 0.999;

        /** Decay Not Supported for ConvLayer */
        get decay():number {return -1}
        /** Decay Not Supported for ConvLayer */
        set decay(x:number) {}

        constructor(size: number) {

            // Explicitly initialize the kernel with randomized weights
            let kernelArray: number[][] = [];
            for (let i = 0; i < size; i++) {
                let row: number[] = [];
                for (let j = 0; j < size; j++) {
                    row.push(Math.random() * 0.5 - 0.25);
                }
                kernelArray.push(row);
            }
            this.kernel = new Tensor(kernelArray);
            this.kernelGradients = new Tensor(kernelArray.map(row => row.map(() => 0))); // Initialize gradients
        }

        // **Forward Pass**: Apply convolution to input tensor
        forward(input: TensorLike): TensorLike {
            return input.matmul(this.kernel);
        }

        // **Backpropagation**: Compute gradients for weight updates
        backward(error: TensorLike, learningRate: number): void {
            // Compute gradients
            this.kernelGradients = error.matmul(this.kernel);

            // Update kernel weights using gradient descent
            this.kernel.data = this.kernel.data.map((row, r) => row.map((val, c) =>
                val - learningRate * this.kernelGradients.data[r][c]));
        }

        // **Training Step**
        train(inputs: TensorLike[], targets: TensorLike[], learningRate: number, epochs: number, activation?: ActivationFunction, lossFunction?: LossFunction): void {
            if (!activation) activation = relu; // Default to ReLU for feature mapping
            if (!lossFunction) lossFunction = mse; // Default to MSE if none provided

            for (let epoch = 0; epoch < epochs; epoch++) {
                let totalLoss = 0;

                for (let i = 0; i < inputs.length; i++) {
                    let input = inputs[i];
                    let target = targets[i];

                    // Forward pass with activation
                    let prediction = activation ? input.matmul(this.kernel).applyFunction(activation) : input.matmul(this.kernel);
                    let error = target.sub(prediction);

                    // Compute loss
                    let loss = lossFunction(error);
                    totalLoss += loss;

                    // Backward pass to adjust kernel weights
                    this.backward(error, learningRate);
                }

                console.log(`Epoch ${epoch + 1}, Loss: ${totalLoss / inputs.length}`);
            }
        }
    }

    // Mean Squared Error (MSE)
    export function mse(error: TensorLike): number {
        return error.applyFunction(x => x * x).sum() / Math.max(1, error.data.length);
    }

    // Mean Cubed Error (MCE)
    export function mce(error: TensorLike): number {
        return error.applyFunction(x => x * x * x).sum() / Math.max(1, error.data.length);
    }

    // Mean Error (ME) - Identity function (X = X)
    export function me(error: TensorLike): number {
        return error.sum() / Math.max(1, error.data.length);
    }

    // Root Mean Squared Error (RMSE)
    export function rmse(error: TensorLike): number {
        return Math.sqrt(error.applyFunction(x => x * x).sum() / Math.max(1, error.data.length));
    }

    // Huber Loss (Delta = 1)
    export function huber(error: TensorLike): number {
        return error.applyFunction(x => Math.abs(x) <= 1 ? 0.5 * x * x : Math.abs(x) - 0.5).sum() / Math.max(1, error.data.length);
    }

    export function al(error: TensorLike): number {
        let meanAbsError = error.applyFunction(x => Math.abs(x)).sum() / Math.max(1, error.data.length);

        // Dynamic switching based on error scale
        if (meanAbsError < 0.05) {
            return rmse(error); // RMSE for small errors (stability)
        } else if (meanAbsError > 10) {
            return huber(error); // Huber Loss for large errors (robustness)
        } else {
            return mce(error); // MCE for precision optimization
        }
    }


    export function mae(predictions: TensorLike, targets: TensorLike): number {
        let errorTensor = predictions.add(targets.applyFunction(x => -x));
        let totalError = errorTensor.data.reduce((sum, row) => sum + row.reduce((rSum, val) => rSum + Math.abs(val), 0), 0);
        let elementCount = predictions.data.length * predictions.data[0].length;
        return totalError / elementCount;
    }
    /**
    * Applies the Rectified Linear Unit (ReLU) activation function.
    * @param x Input value.
    * @returns `x` if positive, otherwise `0`.
    */
    export function relu(x: number): number {
        return Math.max(0, x);
    }
    /**
    * Applies the Sigmoid activation function, useful for binary classification.
    * @param x Input value.
    * @returns A value between 0 and 1.
    */
    export function sigmoid(x: number): number {
        let expX = Math.exp(-x);
        return 1 / (1 + expX);
    }
    /**
    * Applies the Exponential Linear Unit (ELU) activation function.
    * @param x Input value.
    * @param alpha Scaling factor for negative values (default: 1).
    * @returns `x` if positive, otherwise `alpha * (exp(x) - 1)`.
    */
    export function elu(x: number, alpha: number = 1): number {
        return x > 0 ? x : alpha * (Math.exp(x) - 1);
    }
    /**
    * Applies the Hard Sigmoid activation function, a computationally efficient alternative.
    * @param x Input value.
    * @returns A clipped linear approximation of the sigmoid function, constrained between 0 and 1.
    */
    export function hardSigmoid(x: number): number {
        return Math.max(0, Math.min(1, (0.2 * x + 0.5)));
    }
    /**
    * Applies the Bent Identity activation function, preserving identity while introducing slight curvature.
    * @param x Input value.
    * @returns `(sqrt(x² + 1) - 1) / 2 + x`.
    */
    export function bentIdentity(x: number): number {
        return (Math.sqrt(x * x + 1) - 1) / 2 + x;
    }
    /**
    * Applies the Mish activation function, a smooth alternative to ReLU.
    * @param x Input value.
    * @returns `x * tanh(ln(1 + exp(x)))`.
    */
    export function mish(x: number): number {
        return x * tanh(Math.log(1 + Math.exp(x)));
    }
    /**
    * Applies the Swish activation function, a sigmoid-weighted version of ReLU.
    * @param x Input value.
    * @returns `x / (1 + exp(-x))`.
    */
    export function swish(x: number): number {
        return x / (1 + Math.exp(-x)); // Uses sigmoid-like smoothing
    }
    /**
    * Applies the Gaussian Error Linear Unit (GELU) activation function.
    * @param x Input value.
    * @returns `x * (1 + tanh(sqrt2pi * (x + coeff * x³))) / 2`.
    */
    export function gelu(x: number): number {
        let x3 = x * x * x;
        return x * (1 + tanh(Constants.sqrt2pi * (x + Constants.coeff * x3))) / 2;
    }

    /**
    * Applies the Hyperbolic Tangent (Tanh) activation function.
    * @param x Input value.
    * @returns A value between -1 and 1.
    */
    export function tanh(x: number): number {
        let exp2x = Math.exp(2 * x);
        return (exp2x - 1) / (exp2x + 1);
    }

    /**
    * Applies the Leaky ReLU activation function, preventing dead neurons.
    * @param x Input value.
    * @param alpha Slope for negative values (default: 0.01).
    * @returns `x` if positive, otherwise `alpha * x`.
    */

    export function leakyRelu(x: number, alpha: number = 0.01): number {
        let v = x > 0 ? x : alpha * x;
        if (isNaN(v)) {
            v = 0
        }
        return v
    }

    // Softmax (Used for classification problems)
    export function softmax(inputs: number[]): number[] {
        let maxInput = inputs[0]; // Initialize max value
        for (let i = 1; i < inputs.length; i++) {
            if (inputs[i] > maxInput) maxInput = inputs[i];
        }

        let expValues: number[] = [];
        let sumExp = 0;

        for (let i = 0; i < inputs.length; i++) {
            let expVal = Math.exp(inputs[i] - maxInput); // Normalize for stability
            expValues.push(expVal);
            sumExp += expVal;
        }

        let softmaxOutput: number[] = [];
        for (let i = 0; i < expValues.length; i++) {
            softmaxOutput.push(expValues[i] / sumExp);
        }

        return softmaxOutput;
    }
    /**
    * Represents a sequential model consisting of multiple layers (Linear or ConvLayer).
    */
    export class Sequential implements Model {
        /** Array of layers included in the model. */
        layers: (Linear | ConvLayer)[];
        private _decay = 0.999;

        /**
        * Initializes a sequential model with the given layers.
        * @param layers An array of `Linear` or `ConvLayer` instances.
        */
        constructor(layers: (Linear | ConvLayer)[]) {
            this.layers = layers;
        }

        /** 
        * Set the Weight Decay to a Value to have each neurons weights be multplied by
        */
        set decay(x: number) {
            this._decay = x
            this.layers.forEach((a) => a.decay = x)
        }
        /**
        * Get the Weight Decay to a Value Which will have each neurons weights be multplied by
        */
        get decay(): number {
            return this._decay
        }


        // Forward pass through all layers
        /**
        * Performs a forward pass through all layers in the sequential model.
        * @param input The input tensor.
        * @param activation The activation function to use (applied to linear layers).
        * @returns The resulting tensor after passing through all layers.
        */
        forward(input: Tensor, activation: ActivationFunction): TensorLike {
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
        /**
        * Trains the sequential model using backpropagation.
        * @param inputs An array of input tensors used for training.
        * @param targets An array of target tensors (expected outputs).
        * @param learningRate The learning rate for updating weights.
       * @param epochs The number of training iterations.
       * @param activation The activation function to apply (default: sigmoid).
       * @param lossFunction Optional loss function to optimize weight updates (default: MSE).
       * @param silent If `true`, disables console logging for training progress.
       */
        train(inputs: TensorLike[], targets: TensorLike[], learningRate: number, epochs: number,
            activation?: ActivationFunction, lossFunction?: LossFunction, silent?: boolean): void {

            if (!activation) activation = Torch.sigmoid; // Default activation function
            if (!lossFunction) lossFunction = (error) => error.applyFunction(x => x * x).sum() / Math.max(1, error.data.length); // Default to MSE

            for (let epoch = 0; epoch < epochs; epoch++) {
                let totalLoss = 0;

                for (let i = 0; i < inputs.length; i++) {
                    let input = inputs[i];
                    let target = targets[i];

                    // Forward pass
                    let prediction = this.forward(input, activation);
                    let error = target.sub(prediction);

                    error.applyFunction((x) => Math.clamp(x,-10,20))

                    // Compute loss using the provided function
                    let loss = lossFunction(error);
                    totalLoss += loss;

                    // Backpropagation through all layers in reverse order
                    let previousError = error;
                    let reversedLayers = this.layers.slice();
                    
                    reversedLayers.reverse();

                    reversedLayers.forEach(layer => {
                        if (layer instanceof ConvLayer) {
                            layer.backward(previousError, learningRate); // ConvLayer only adjusts kernels
                        } else {
                            previousError = layer.backward(previousError, learningRate, activation, lossFunction);
                        }
                    });

                }

                // Learning rate decay with a minimum limit
                if (epoch % 200 === 0) {
                    learningRate = Math.max(learningRate * 0.95, 0.005);
                }

                if (silent !== true) {
                    console.log(`Epoch ${epoch + 1}, Loss: ${totalLoss / inputs.length}`);
                }
            }
        }
        /** 
        * `Internal Use Only`
        * Tensorizes a 2d matrix -> Tensor array
        */
        protected tensorize(data: number[][]): TensorLike[] {
            let tensors: Tensor[] = []
            data.forEach((a) => tensors.push(new Tensor([a])))
            return tensors
        }
        /**
        * Trains a dataset using multiple input-target pairs over several epochs.
        * @param DataSet The dataset containing input-output pairs.
        * @param learningRate The learning rate for weight adjustments.
        * @param epochs Number of training iterations.
        * @param activation Optional activation function for training (default: sigmoid).
        * @param lossFunction Optional loss function for optimizing weight updates (default: MSE).
        * @param disableLogging If true, disables console logs for training progress.
        */
        trainDataSet(DataSet: DataSet, learningRate: number, epochs: number, activation?: ActivationFunction, lossFunction?: LossFunction, disableLogging?: boolean) {
            this.train(this.tensorize(DataSet.inputs), this.tensorize(DataSet.outputs), learningRate, epochs, activation, lossFunction, disableLogging)
        }
    }
}