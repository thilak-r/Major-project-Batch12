import os
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import models, transforms
import cv2
import matplotlib
#matplotlib.use('Agg')  # Prevents tkinter errors

load_dotenv()

# Loading vector store
try:
    print("Attempting to load vector store...")
    # absolute path for the HospitalDB 
    db_path = os.path.abspath("HospitalDB")
    print(f"Looking for vector store in: {db_path}")
    
    # List all files in the directory
    if os.path.exists(db_path):
        print(f"Directory contents of {db_path}:")
        for file in os.listdir(db_path):
            file_path = os.path.join(db_path, file)
            size = os.path.getsize(file_path)
            print(f"- {file} ({size} bytes)")
    else:
        print(f"Directory does not exist: {db_path}")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Vector store directory '{db_path}' not found")
    
    # Verify required files exist
    required_files = ['index.faiss', 'index.pkl']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(db_path, f))]
    if missing_files:
        raise FileNotFoundError(f"Missing required files in {db_path}: {', '.join(missing_files)}")
    
    print(f"Found vector store files in {db_path}")
    
    # Initialize embedding model
    print("Initializing embedding model...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    print("Found Google API key")
    
    # Load vector store
    print("Loading FAISS index...")
    try:
        vectorstore = FAISS.load_local(
            embedding_model,
        )
        print("Vector store loaded successfully")
        
        # Verify the vector store is working
        print("Testing vector store with a sample query...")
        test_docs = vectorstore.similarity_search("test", k=1)
        print(f"Successfully retrieved {len(test_docs)} test documents")
        
    except Exception as load_error:
        print(f"Error during FAISS loading: {str(load_error)}")
        print(f"Error type: {type(load_error).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise
    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load models
def load_efficientnet_model():
    try:
        model = models.efficientnet_b0(weights=None)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading EfficientNet model: {e}")
        return None

def load_densenet_model():
    try:
        model = models.densenet121(weights="IMAGENET1K_V1")
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading DenseNet model: {e}")
        return None

def load_resnet_model():
    try:
        model = models.resnet50(weights=None)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading ResNet model: {e}")
        return None

def load_xception_model():
    try:
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading Xception model: {e}")
        return None


# function to initialize models
def initialize_models():
    try:
        # Load EfficientNe
        # Load DenseNet
        models_dict['DenseNet']['model'] = load_densenet_model()
        if models_dict['DenseNet']['model']:
            print("DenseNet model loaded successfully")
        
        # Load ResNet
        models_dict['ResNet']['model'] = load_resnet_model()
        if models_dict['ResNet']['model']:
            print("ResNet model loaded successfully")
        
        # Load Xception
        models_dict['Xception']['model'] = load_xception_model()
        if models_dict['Xception']['model']:
            xception_model = models_dict['Xception']['model']
            target_layer = None
            
            
            if hasattr(xception_model, 'act4'):
            else:
                for name, module in xception_model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)) and not isinstance(module, nn.Linear):
                        target_layer = module
                        print(f"Using fallback layer '{name}' for Xception")
                        break
            
            if target_layer is not None:
            else:
                print("Could not find appropriate target layer for Xception model")
        
    except Exception as e:
        print(f"Error initializing models: {e}")

# Call initialize_models when the app starts
with app.app_context():
    initialize_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(image_path, model_name):
    try:
        # Open and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get the model
        model_info = models_dict[model_name]
        model = model_info['model']
        
        # Check if model is loaded
        if model is None:
            return {'error': f"{model_name} model is not loaded properly"}
        

        
        # Generate Grad-CAM visualization
        cam = model_info['cam']
        if cam is None:
            return {'error': f"GradCAM for {model_name} is not initialized properly"}
            
        grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(predicted_class)])
        grayscale_cam = grayscale_cam[0, :]
        
        # Convert the image for visualization using PIL instead of OpenCV
        img_np = np.array(image.resize((224, 224)))
        
        # Use show_cam_on_image from pytorch_grad_cam.utils.image instead of OpenCV
        visualization = show_cam_on_image(img_np / 255.0, grayscale_cam, use_rgb=True, colormap=2)  # 2 corresponds to COLORMAP_JET
        
        # Convert visualization to base64 for web display
        pil_img = Image.fromarray(visualization)
        buf = io.BytesIO()
        pil_img.save(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return result
    
    except Exception as e:
        print(f"Error in prediction with {model_name}: {e}")
        return {'error': str(e)}

# Initialize the Gemini model
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please add it to your .env file.")
    
    print("Chat model initialized successfully")
except Exception as e:
    print(f"Error initializing chat model: {str(e)}")
    chat_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        results = {}
        
        # Process with each model
        for model_name in models_dict.keys():
            results[model_name] = predict_image(file_path, model_name)
        
        return jsonify({
            'filename': filename,
            'file_path': file_path,
            'results': results
        })
    
    return jsonify({'error': 'Invalid file type'})

# Remove the redundant /analyze endpoint since /predict is being used

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if chat_model is None:
            print("Error: Chat model is not initialized")
            return jsonify({
                'error': 'Chat model is not initialized. Please check your API key configuration.'
            }), 500

        if vectorstore is None:
            print("Error: Vector store is not initialized")
            return jsonify({
                'error': 'Vector store is not initialized. Please check the HospitalDB directory.'
            }), 500

        data = request.get_json()
        if not data:
            print("Error: No JSON data in request")
            return jsonify({'error': 'No JSON data provided'}), 400

        user_message = data.get('message', '')
        if not user_message:
            print("Error: Empty message")
            return jsonify({'error': 'No message provided'}), 400

        print(f"Processing message: {user_message}")

        try:
            print("Successfully generated response")
            return jsonify({'response': response.content})

        except Exception as inner_e:
            return jsonify({
                'error': f'Error in RAG processing: {str(inner_e)}'
            }), 500

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == '__main__':
    app.run(debug=True)
