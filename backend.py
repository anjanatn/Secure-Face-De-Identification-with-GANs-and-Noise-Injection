from flask import Flask, request, send_file, jsonify, render_template
import os
import sys
import logging
import numpy as np
import PIL.Image
from PIL import PngImagePlugin
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from werkzeug.utils import secure_filename
import cv2
import random
import hashlib
import json
import io
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Update the Flask app initialization
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')  # Add this line to serve static files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
UPLOAD_FOLDER = "uploads"
OUTPUT_DIR = "output_images"
MODEL_PATH = "stylegan2-ffhq.pkl"  # Change to your actual model path

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load model
try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # Use a dummy model for code explanation if actual model can't be loaded
    class DummyGAN(nn.Module):
        def __init__(self):
            super(DummyGAN, self).__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
            )
            
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128 + 64, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            )
        
        def forward(self, x, noise_level=0.5, is_denoising=False):
            """Forward pass"""
            encoded = self.encoder(x)
            
            # For both noising and denoising, use the same approach
            noise_shape = (encoded.size(0), 64, encoded.size(2), encoded.size(3))
            
            # Use deterministic noise based on the seed
            seed = int(noise_level * 10000) % 10000
            torch.manual_seed(seed)
            
            # Generate the noise - for denoising, we just don't apply any
            noise = torch.randn(noise_shape, device=encoded.device)
            if not is_denoising:
                noise = noise * noise_level
            else:
                noise = torch.zeros_like(noise)  # Zero noise for denoising
            
            # Concatenate along the channel dimension
            encoded_with_noise = torch.cat([encoded, noise], dim=1)
            
            # Pass through decoder
            decoded = self.decoder(encoded_with_noise)
            return decoded
    
    model = DummyGAN().to(device)
    logger.warning("Using dummy model for demonstration purposes")

def get_noise_level_from_code(code):
    """Convert a 4-digit code to a noise level between 0 and 1"""
    try:
        # Validate input
        if not code.isdigit() or len(code) != 4:
            return 0.5  # Default noise level if code is invalid
        
        # Convert code to noise level (0000 -> 0.0, 9999 -> 1.0)
        return float(int(code)) / 9999.0
    except Exception as e:
        logger.error(f"Error converting code to noise level: {e}")
        return 0.5

def get_key_from_code(code):
    """Generate a secure encryption key from a 4-digit code"""
    # Convert the code to bytes
    code_bytes = code.encode()
    
    # Use PBKDF2 to derive a secure key
    salt = b'secure_face_deidentification'  # Fixed salt for reproducibility
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(code_bytes))
    return key

def process_image(image_path, code, is_denoising=False):
    """Process an image - either adding noise or removing it"""
    try:
        # Load image
        input_image = PIL.Image.open(image_path).convert('RGB')
        original_size = input_image.size
        
        # Generate a key from the code
        key = get_key_from_code(code)
        noise_level = float(int(code)) / 9999.0  # For consistency with previous implementation
        
        # If denoising, check if this is the original image with the right code
        if is_denoising:
            # Try to extract the original image data
            try:
                with PIL.Image.open(image_path) as img:
                    # Check if this image has our special metadata tag
                    if "encrypted_original" in img.info:
                        # Get the encrypted data
                        encrypted_data = img.info["encrypted_original"]
                        
                        # Try to decrypt with the provided code
                        try:
                            # Create cipher suite with the key
                            cipher_suite = Fernet(key)
                            
                            # Decrypt the original image data
                            decrypted_data = cipher_suite.decrypt(encrypted_data.encode())
                            
                            # Convert back to image
                            original_image = PIL.Image.open(io.BytesIO(decrypted_data))
                            
                            # Save to output path
                            image_id = str(uuid.uuid4())
                            output_path = os.path.join(OUTPUT_DIR, f"denoised_{image_id}.png")
                            original_image.save(output_path)
                            
                            logger.info("Successfully recovered original image with correct code")
                            return output_path, image_id, True
                            
                        except Exception as e:
                            logger.warning(f"Decryption failed, code may be incorrect: {e}")
                            # Fall through to regular processing
                    else:
                        logger.warning("Image does not contain embedded original data")
                        # Fall through to regular processing
            except Exception as e:
                logger.warning(f"Error checking for embedded data: {e}")
                # Fall through to regular processing
        
        # Apply transformations for model
        input_tensor = transform(input_image).unsqueeze(0).to(device)
        
        # Process with the model
        with torch.no_grad():
            if isinstance(model, DummyGAN):
                output_tensor = model(input_tensor, noise_level, is_denoising=is_denoising)
            else:
                output_tensor = model(input_tensor, 0.0 if is_denoising else noise_level)
            
            # Convert output to image
            output_image = output_tensor.squeeze(0).cpu()
            output_image = (output_image * 0.5 + 0.5).clamp(0, 1)
            output_image = transforms.ToPILImage()(output_image)
            output_image = output_image.resize(original_size)
            
            # Create output filename
            image_id = str(uuid.uuid4())
            output_path = os.path.join(OUTPUT_DIR, f"{'denoised' if is_denoising else 'deidentified'}_{image_id}.png")
            
            # For noising, embed the original image securely in the metadata
            if not is_denoising:
                # Save original image to encrypted metadata
                buffer = io.BytesIO()
                input_image.save(buffer, format="PNG")
                buffer.seek(0)
                
                # Encrypt the original image data
                cipher_suite = Fernet(key)
                encrypted_data = cipher_suite.encrypt(buffer.getvalue())
                
                # Create metadata with the encrypted data
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("encrypted_original", encrypted_data.decode())
                metadata.add_text("noise_level", str(noise_level))
                
                # Save with metadata
                output_image.save(output_path, pnginfo=metadata)
            else:
                # Regular save for denoising result
                output_image.save(output_path)
            
            return output_path, image_id, False
            
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def extract_noise_level(image_path):
    """Extract noise level from image metadata if present"""
    try:
        with PIL.Image.open(image_path) as img:
            if "noise_level" in img.info:
                return float(img.info["noise_level"])
    except Exception as e:
        logger.error(f"Error extracting noise level: {e}")
    
    # Return None if no metadata found or error occurred
    return None

# Update the deidentify endpoint
@app.route("/deidentify", methods=["POST"])
def deidentify():
    """API endpoint to de-identify a face"""
    try:
        # Check if file is present in the request
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        
        # Check if filename is empty
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400
        
        # Get the 4-digit code from the form
        code = request.form.get("code", "")
        if not code.isdigit() or len(code) != 4:
            return jsonify({"error": "Please provide a valid 4-digit code"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        # Process the image
        output_path, image_id, _ = process_image(temp_path, code, is_denoising=False)
        
        # Clean up uploaded file
        os.remove(temp_path)
        
        # Return the result
        return send_file(output_path, mimetype="image/png", 
                        as_attachment=True,
                        download_name=f"deidentified_image.png")
    except Exception as e:
        logger.error(f"Error processing de-identification request: {e}")
        return jsonify({"error": str(e)}), 500

# Update the denoise endpoint
@app.route("/denoise", methods=["POST"])
def denoise():
    """Remove noise from a previously processed image using a code"""
    try:
        # Check if file is present in the request
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        
        # Check if filename is empty
        if file.filename == "":
            return jsonify({"error": "No image selected"}), 400
        
        # Get the 4-digit code from the form
        code = request.form.get("code", "")
        if not code.isdigit() or len(code) != 4:
            return jsonify({"error": "Please provide a valid 4-digit code"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        # Process the image
        output_path, _, is_perfect = process_image(temp_path, code, is_denoising=True)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Return the result
        result_type = "perfect" if is_perfect else "approximate"
        logger.info(f"Denoising complete. Result type: {result_type}")
        
        return send_file(
            output_path, 
            mimetype="image/png", 
            as_attachment=True,
            download_name=f"denoised_image_{result_type}.png"
        )
    except Exception as e:
        logger.error(f"Error denoising image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    """Serve the main HTML interface"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return jsonify({"message": "Face De-Identification API is running!", 
                        "error": "Could not load UI template"}), 500

# Cleanup function to periodically remove old images
@app.before_request
def cleanup_old_images():
    try:
        # Keep only the 100 most recent files
        for directory in [UPLOAD_FOLDER, OUTPUT_DIR]:
            files = [os.path.join(directory, f) for f in os.listdir(directory) 
                    if os.path.isfile(os.path.join(directory, f))]
            if len(files) > 100:
                files.sort(key=os.path.getctime)
                for f in files[:-100]:
                    os.remove(f)
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1200, debug=False)