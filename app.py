import os
import io
import time
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from urllib.request import urlopen, Request

# Suppress noisy TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, static_folder='static', template_folder='templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_images')
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Cache for loaded models
MODELS = {}

MODEL_CONFIGS = {
    'resnet50': {
        'id': 'resnet50',
        'name': 'ResNet-50',
        'architecture': 'Residual Network (50 layers)',
        'target_size': (224, 224),
        'params': '25.6M',
        'depth': '50 Layers',
        'specialty': 'Deep residual skip-connections preventing vanishing gradients.',
        'paper_url': 'https://arxiv.org/abs/1512.03385',
        'extract_layers': ['avg_pool', 'conv5_block3_out', 'conv4_block6_out']
    },
    'vgg16': {
        'id': 'vgg16',
        'name': 'VGG-16',
        'architecture': 'Visual Geometry Group (16 layers)',
        'target_size': (224, 224),
        'params': '138.4M',
        'depth': '16 Weight Layers',
        'specialty': 'Homogeneous 3x3 convolution kernels with deep sequential blocks.',
        'paper_url': 'https://arxiv.org/abs/1409.1556',
        'extract_layers': ['fc1', 'fc2', 'block5_pool']
    },
    'vgg19': {
        'id': 'vgg19',
        'name': 'VGG-19',
        'architecture': 'Visual Geometry Group (19 layers)',
        'target_size': (224, 224),
        'params': '143.7M',
        'depth': '19 Weight Layers',
        'specialty': 'Extended 19-layer architecture with dense representations.',
        'paper_url': 'https://arxiv.org/abs/1409.1556',
        'extract_layers': ['fc1', 'fc2', 'block5_pool']
    },
    'inceptionv3': {
        'id': 'inceptionv3',
        'name': 'Inception-v3',
        'architecture': 'GoogLeNet Inception-v3',
        'target_size': (299, 299),
        'params': '23.8M',
        'depth': '48 Layers',
        'specialty': 'Multi-scale factorized convolutions (1x1, 3x3, 5x5) and asymmetric filters.',
        'paper_url': 'https://arxiv.org/abs/1512.00567',
        'extract_layers': ['avg_pool', 'mixed10', 'mixed9']
    }
}

def get_model(model_id):
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_id}")
    
    if model_id not in MODELS:
        print(f"Loading weights for {model_id}...")
        if model_id == 'resnet50':
            from keras.applications.resnet50 import ResNet50
            MODELS['resnet50'] = ResNet50(weights='imagenet', include_top=True)
        elif model_id == 'vgg16':
            from keras.applications.vgg16 import VGG16
            MODELS['vgg16'] = VGG16(weights='imagenet', include_top=True)
        elif model_id == 'vgg19':
            from keras.applications.vgg19 import VGG19
            MODELS['vgg19'] = VGG19(weights='imagenet', include_top=True)
        elif model_id == 'inceptionv3':
            from keras.applications.inception_v3 import InceptionV3
            MODELS['inceptionv3'] = InceptionV3(weights='imagenet', include_top=True)
        print(f"Model {model_id} loaded successfully!")
        
    return MODELS[model_id]

def preprocess_for_model(img, model_id):
    cfg = MODEL_CONFIGS[model_id]
    target_size = cfg['target_size']
    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    if arr.ndim == 2: # Grayscale to RGB
        arr = np.stack((arr,)*3, axis=-1)
    elif arr.shape[-1] == 4: # RGBA to RGB
        arr = arr[..., :3]
        
    arr = np.expand_dims(arr, axis=0)
    
    if model_id == 'resnet50':
        from keras.applications.resnet50 import preprocess_input
        return preprocess_input(arr)
    elif model_id == 'vgg16':
        from keras.applications.vgg16 import preprocess_input
        return preprocess_input(arr)
    elif model_id == 'vgg19':
        from keras.applications.vgg19 import preprocess_input
        return preprocess_input(arr)
    elif model_id == 'inceptionv3':
        from keras.applications.inception_v3 import preprocess_input
        return preprocess_input(arr)

def decode_for_model(preds, model_id, top=5):
    if model_id == 'resnet50':
        from keras.applications.resnet50 import decode_predictions
    elif model_id == 'vgg16':
        from keras.applications.vgg16 import decode_predictions
    elif model_id == 'vgg19':
        from keras.applications.vgg19 import decode_predictions
    elif model_id == 'inceptionv3':
        from keras.applications.inception_v3 import decode_predictions
        
    decoded = decode_predictions(preds, top=top)[0]
    results = []
    for class_id, label, prob in decoded:
        results.append({
            'class_id': class_id,
            'label': label.replace('_', ' ').title(),
            'probability': float(prob),
            'percentage': round(float(prob) * 100, 2)
        })
    return results

def load_image_from_source(source_type, source_data):
    """Loads a PIL image from 'preset', 'url', or 'base64'"""
    if source_type == 'preset':
        filename = f"{source_data}.jpg"
        filepath = os.path.join(SAMPLE_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preset image not found: {filename}")
        return Image.open(filepath).convert('RGB')
        
    elif source_type == 'url':
        req = Request(source_data, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urlopen(req, timeout=10) as resp:
            data = resp.read()
        return Image.open(io.BytesIO(data)).convert('RGB')
        
    elif source_type == 'base64':
        if ',' in source_data:
            source_data = source_data.split(',', 1)[1]
        raw_bytes = base64.b64decode(source_data)
        return Image.open(io.BytesIO(raw_bytes)).convert('RGB')
        
    raise ValueError(f"Unknown image source type: {source_type}")

@app.route('/')
def index():
    return render_template('index.html', configs=MODEL_CONFIGS)

@app.route('/api/models', methods=['GET'])
def api_models():
    return jsonify({'success': True, 'models': MODEL_CONFIGS})

@app.route('/api/sample-images/<filename>')
def get_sample_image(filename):
    return send_from_directory(SAMPLE_DIR, filename)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json or {}
    model_id = data.get('model', 'resnet50')
    source_type = data.get('source_type', 'preset')
    source_data = data.get('source_data', 'elephant')
    top_k = int(data.get('top_k', 5))
    
    try:
        pil_img = load_image_from_source(source_type, source_data)
        
        # Check if user requested comparison across all 4 models
        if model_id == 'all':
            comparison = {}
            for mid in ['resnet50', 'vgg16', 'vgg19', 'inceptionv3']:
                t0 = time.time()
                m = get_model(mid)
                x = preprocess_for_model(pil_img, mid)
                preds = m.predict(x, verbose=0)
                dur = round((time.time() - t0) * 1000, 1)
                comparison[mid] = {
                    'predictions': decode_for_model(preds, mid, top=top_k),
                    'latency_ms': dur,
                    'config': MODEL_CONFIGS[mid]
                }
            return jsonify({
                'success': True,
                'mode': 'comparison',
                'results': comparison
            })
            
        # Single model prediction
        t0 = time.time()
        model = get_model(model_id)
        x = preprocess_for_model(pil_img, model_id)
        preds = model.predict(x, verbose=0)
        dur = round((time.time() - t0) * 1000, 1)
        
        results = decode_for_model(preds, model_id, top=top_k)
        
        return jsonify({
            'success': True,
            'mode': 'single',
            'model_id': model_id,
            'model_name': MODEL_CONFIGS[model_id]['name'],
            'predictions': results,
            'latency_ms': dur
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/extract', methods=['POST'])
def api_extract():
    """Extract intermediate features from a selected layer."""
    data = request.json or {}
    model_id = data.get('model', 'inceptionv3')
    layer_name = data.get('layer', 'avg_pool')
    source_type = data.get('source_type', 'preset')
    source_data = data.get('source_data', 'sharpener')
    
    try:
        from keras.models import Model
        base_model = get_model(model_id)
        
        # Verify layer exists
        layer = base_model.get_layer(layer_name)
        extractor = Model(inputs=base_model.input, outputs=layer.output)
        
        pil_img = load_image_from_source(source_type, source_data)
        x = preprocess_for_model(pil_img, model_id)
        
        t0 = time.time()
        features = extractor.predict(x, verbose=0)
        dur = round((time.time() - t0) * 1000, 1)
        
        # Flatten feature vector if multidimensional
        flat_feats = features.flatten()
        total_dim = int(flat_feats.size)
        
        # Sample or compute statistical distribution
        mean_val = float(np.mean(flat_feats))
        std_val = float(np.std(flat_feats))
        max_val = float(np.max(flat_feats))
        min_val = float(np.min(flat_feats))
        sparsity = float(np.mean(flat_feats == 0) * 100)
        
        # Top 16 strongest activations
        top_indices = np.argsort(flat_feats)[::-1][:16].tolist()
        top_activations = [{'index': idx, 'value': round(float(flat_feats[idx]), 4)} for idx in top_indices]
        
        # First 64 values for sparkline / heatmap preview
        sample_size = min(64, total_dim)
        preview_values = [round(float(v), 4) for v in flat_feats[:sample_size]]
        
        return jsonify({
            'success': True,
            'model_id': model_id,
            'layer_name': layer_name,
            'shape': list(features.shape),
            'total_dimensions': total_dim,
            'stats': {
                'mean': round(mean_val, 4),
                'std': round(std_val, 4),
                'max': round(max_val, 4),
                'min': round(min_val, 4),
                'sparsity_pct': round(sparsity, 1)
            },
            'top_activations': top_activations,
            'preview_values': preview_values,
            'latency_ms': dur
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/model-layers/<model_id>')
def api_model_layers(model_id):
    try:
        model = get_model(model_id)
        layers_info = []
        for i, layer in enumerate(model.layers):
            try:
                out_shape = str(layer.output.shape)
            except Exception:
                out_shape = "Dynamic"
            layers_info.append({
                'index': i,
                'name': layer.name,
                'class_name': layer.__class__.__name__,
                'output_shape': out_shape,
                'params': int(layer.count_params())
            })
        return jsonify({'success': True, 'layers': layers_info, 'count': len(layers_info)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("Pre-warming ResNet-50 for instant response...")
    try:
        get_model('resnet50')
    except Exception as e:
        print("Note on startup warm-up:", e)
    print("Serving CNN Vision Hub at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
