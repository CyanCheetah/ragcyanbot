from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
from chatbot import CyanBot
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize chatbot
chatbot = CyanBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        response = chatbot.get_response(message)
        return jsonify({'response': response})
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        files = request.files.getlist('file')
        if not files:
            return jsonify({'error': 'No files selected'}), 400
        
        uploaded_files = []
        for file in files:
            if file.filename:
                # Save file
                file_path = os.path.join('documents', file.filename)
                file.save(file_path)
                uploaded_files.append(file.filename)
        
        if uploaded_files:
            # Update both vector stores
            chatbot.load_documents('documents')
            return jsonify({
                'message': f'Successfully uploaded {len(uploaded_files)} files',
                'files': uploaded_files
            })
        
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'No search query provided'}), 400
        
        # Search documents using OpenSearch
        results = chatbot.opensearch_client.search_documents(query)
        return jsonify({'results': results})  # Results should be a list
    
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}", exc_info=True)
        return jsonify({'results': [], 'error': str(e)})

@app.route('/graph', methods=['GET'])
def get_graph():
    try:
        # Generate keyword graph
        graph_data = chatbot.get_keyword_graph()
        if graph_data:
            return jsonify(graph_data)
        return jsonify({'error': 'No graph data available'}), 404
    
    except Exception as e:
        logger.error(f"Error in graph endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000) 