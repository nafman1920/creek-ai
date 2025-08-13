from flask import Flask, request, jsonify, send_file, render_template
import io
from backend_ai import generate_text_func, generate_image_func, generate_voice_func, scrape_surface_web_func

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate-text', methods=['POST'])
def generate_text_route():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    text = generate_text_func(query)
    return jsonify({'result': text})

@app.route('/generate-image', methods=['POST'])
def generate_image_route():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    image = generate_image_func(query)
    if image is None:
        return jsonify({'error': 'Image generation failed'}), 500
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/generate-voice', methods=['POST'])
def generate_voice_route():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    output_path = generate_voice_func(text)
    return send_file(output_path, mimetype='audio/wav', as_attachment=True, download_name='output.wav')

@app.route('/scrape', methods=['POST'])
def scrape_route():
    data = request.json
    query = data.get('query')
    engine = data.get('engine', 'duckduckgo')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    results = scrape_surface_web_func(query, engine)
    return jsonify({'results': results})

if __name__ == "__main__":
    app.run(debug=True)
