import torch
import json

def convert_model(model_path: str, output_path: str):
    # Load the PyTorch model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    layer_data = []  # Store structured layer info

    # Loop through all layers and extract details
    for name, param in model.named_parameters():
        layer_info = {
            "name": name,
            "weights": param.tolist(),  # Convert tensors to lists
        }
        layer_data.append(layer_info)

    # Generate MakeCode Arcade-compatible output
    output_code = f"""// Auto-generated Torch model loader
const layerData = {json.dumps(layer_data, indent=2)};

function loadModel() {{
    let layers = layerData.map(layer => {{
        if (layer.name.includes("conv")) {{
            return new ConvLayer(layer.weights.length); // Adjust for kernel size
        }} else {{
            return new LinearLayer(layer.weights.length);
        }}
    }});
    return new Sequential(layers);
}}

// Import the model into TORCH - makecode arcade TS version
const model = loadModel();
"""

    # Save the output to a file
    with open(output_path, 'w') as f:
        f.write(output_code)

    print(f"Conversion successful! TypeScript model saved to {output_path}")

# Example usage
if __name__ == "__main__":
    model_path = input("Enter path to saved PyTorch model: ")
    output_path = "converted_model.ts"
    convert_model(model_path, output_path)
