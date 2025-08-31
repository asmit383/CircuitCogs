import os
import sys
import json
import tempfile
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import the FlameAnalyzer from test5.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import FlameAnalyzer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
temp_dir = tempfile.mkdtemp()

# Initialize the analyzer
analyzer = FlameAnalyzer()

# Format API response function
def format_api_response(results):
    """Format analysis results for API response"""
    if not results:
        return {'fire_detected': False, 'error': 'Analysis failed'}
        
    # Check if fire was detected
    fire_detected = len(results.get('boxes', [])) > 0
    
    if not fire_detected:
        return {'fire_detected': False}
        
    # Get LLaMA results
    llama_results = results.get('llama_results', {})
    average_result = llama_results.get('average_result', {})
    
    # Get top material
    top_material = None
    top_confidence = 0
    recommended_extinguisher = "Unknown"
    
    all_materials = []
    
    if isinstance(average_result, dict) and 'top_materials' in average_result:
        materials = average_result.get('top_materials', [])
        
        for material in materials:
            if isinstance(material, dict):
                material_name = material.get('name', 'Unknown')
                similarity = material.get('similarity', 0)
                extinguishers = material.get('extinguishers', ['Unknown'])
                
                all_materials.append({
                    'material': material_name,
                    'similarity': similarity
                })
                
                if similarity > top_confidence:
                    top_confidence = similarity
                    top_material = material_name
                    if extinguishers and len(extinguishers) > 0:
                        recommended_extinguisher = extinguishers[0]
    
    # Get AI reasoning
    ai_reasoning = ""
    if isinstance(average_result, dict) and 'reasoning' in average_result:
        ai_reasoning = average_result.get('reasoning', '')
    
    # Format the response according to the specified format
    response = {
        'fire_detected': fire_detected,
        'top_material': top_material or "Unknown",
        'confidence': top_confidence,
        'extinguisher': recommended_extinguisher,
        'all_results': all_materials,
        'ai_reasoning': ai_reasoning
    }
    
    return response

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(temp_dir, filename)
            file.save(filepath)
            
            # Run analysis
            results = analyzer.run_full_analysis(filepath)
            
            if results is None:
                return jsonify({'error': 'Analysis failed'}), 500
            
            # Format JSON response
            response_data = format_api_response(results)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/demo', methods=['GET'])
def api_demo():
    return jsonify({
        'status': 'success',
        'message': 'Fire Analysis API is running',
        'endpoints': {
            '/api/analyze': 'POST - Upload an image for material analysis',
            '/api/demo': 'GET - Test endpoint'
        }
    })

if __name__ == '__main__':
    print("Starting Fire Analysis API Server...")
    print("Loading models and database...")
    print("Server running at http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)