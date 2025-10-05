from flask import Flask, render_template, request, jsonify, send_file
import cv2
import pytesseract
from PIL import Image
import os
import re
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set Tesseract path (adjust for your system)
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.isfile(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def auto_preprocess(gray):
    """Automatic preprocessing for best OCR results"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return processed

def post_process_text(text):
    """Clean up extracted text"""
    if not text.strip():
        return text
    
    text = re.sub(r'\s+', ' ', text)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if len(re.findall(r'[a-zA-Z0-9]', line)) >= 2:
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract_text():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        language = request.form.get('language', 'eng')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and process image
        image = cv2.imread(filepath)
        if image is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to load image'}), 400
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed = auto_preprocess(gray)
        
        # Save preprocessed image
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_processed.png')
        cv2.imwrite(temp_path, processed)
        
        # Perform OCR
        custom_config = f'-l {language} --oem 3 --psm 6'
        raw_text = pytesseract.image_to_string(Image.open(temp_path), config=custom_config)
        text = post_process_text(raw_text)
        
        # Get confidence scores
        data = pytesseract.image_to_data(Image.open(temp_path), output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Clean up
        os.remove(filepath)
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'text': text,
            'confidence': round(avg_confidence, 1),
            'dimensions': f"{width}x{height}",
            'word_count': len(text.split())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['POST'])
def download_text():
    try:
        text = request.json.get('text', '')
        
        # Create text file in memory
        buffer = BytesIO()
        buffer.write(text.encode('utf-8'))
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name='extracted_text.txt',
            mimetype='text/plain'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)