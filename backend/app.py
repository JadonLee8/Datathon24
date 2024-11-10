from flask import Flask, request, jsonify, send_file
import torch
import sys
from flask_cors import CORS

# sys.path.append('/python/network.py')
from backend_net import UNet, png_to_prediction

app = Flask(__name__)
CORS(app)

@app.route("/process-image", methods=["POST"])
def process():
    print("Processing image...")
    data = request.get_json()
    if not data or "bounds" not in data:
        return jsonify(error="Invalid input"), 400
    bounds = data["bounds"]
    print(f"Received bounds: {bounds}")


    # Define paths for input image and output
    input_image_path = "1.png"  # Replace with the path to your input image
    output_image_path = "output_overlay.png"  # Path to save the overlay image
    model_path = "trained_model.pth"  # Path to the saved model parameters

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and load trained parameters
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model parameters from {model_path}")

    # Run the prediction and save the overlay image
    red_area_proportion = png_to_prediction(
        model=model,
        file_path=input_image_path,
        device=device,
        threshold=0.5,
        hue_color=(1, 0, 0),
        opacity=0.9,
        save_path=output_image_path
    )

    # Calculate and print the percentage of the red overlay
    red_area_percentage = red_area_proportion * 100
    print(f"Percentage of red area in overlay image: {red_area_percentage:.2f}%")

    return jsonify(bounds=f"{bounds}",
                   image_url=f"http://127.0.0.1:5000/get-image")

@app.route("/get-image", methods=["GET"])
def get_image():
    return send_file("output_overlay.png", mimetype="image/png")
