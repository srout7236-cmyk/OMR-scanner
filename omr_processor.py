import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Allow requests from your Base44 website

def download_image(url):
    """Download image from URL"""
    response = requests.get(url, timeout=30)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def detect_filled_bubbles(image, num_questions):
    """
    Detect which bubbles are filled using OpenCV
    Returns: list of answers with positions
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours (circles/bubbles)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find bubbles
    bubbles = []
    height, width = image.shape[:2]
    
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio (should be ~1 for circles)
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Filter: circles in answer section (bottom 50% of image)
        if (0.7 <= aspect_ratio <= 1.3 and 
            150 < area < 2500 and 
            y > height * 0.35):  # Answer section
            
            bubbles.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'center_x': x + w // 2,
                'center_y': y + h // 2,
                'area': area
            })
    
    # Sort bubbles by Y coordinate (top to bottom)
    bubbles.sort(key=lambda b: b['center_y'])
    
    # Group bubbles into rows (questions)
    questions = []
    tolerance = 30  # pixels
    
    i = 0
    while i < len(bubbles):
        current_row = [bubbles[i]]
        current_y = bubbles[i]['center_y']
        
        # Find all bubbles in same row
        j = i + 1
        while j < len(bubbles):
            if abs(bubbles[j]['center_y'] - current_y) <= tolerance:
                current_row.append(bubbles[j])
                j += 1
            else:
                break
        
        # Valid question has exactly 4 bubbles
        if len(current_row) == 4:
            # Sort by X coordinate (left to right)
            current_row.sort(key=lambda b: b['center_x'])
            questions.append(current_row)
        
        i = j if j > i + 1 else i + 1
    
    # Detect which bubble is filled for each question
    answers = []
    
    for q_num, question_bubbles in enumerate(questions[:num_questions], 1):
        # Calculate fill percentage for each bubble
        fill_percentages = []
        
        for bubble in question_bubbles:
            x, y, w, h = bubble['x'], bubble['y'], bubble['w'], bubble['h']
            
            # Extract bubble region from threshold image
            bubble_region = thresh[y:y+h, x:x+w]
            
            # Calculate percentage of white pixels (filled area in inverted image)
            white_pixels = np.sum(bubble_region == 255)
            total_pixels = w * h
            fill_pct = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            
            fill_percentages.append(fill_pct)
        
        # Find bubble with highest fill percentage
        if fill_percentages:
            max_fill = max(fill_percentages)
            
            # Threshold: must be at least 30% filled
            if max_fill > 30:
                position = fill_percentages.index(max_fill) + 1
            else:
                position = 0  # No answer
        else:
            position = 0
        
        answers.append({
            'question_number': q_num,
            'position': position,
            'fill_percentages': [round(p, 1) for p in fill_percentages]
        })
    
    # Fill remaining questions if fewer detected
    while len(answers) < num_questions:
        answers.append({
            'question_number': len(answers) + 1,
            'position': 0,
            'fill_percentages': []
        })
    
    return answers, len(bubbles), len(questions)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "OMR Service is running"})

@app.route('/process-omr', methods=['POST'])
def process_omr():
    """Process OMR sheet"""
    try:
        # Get data from request
        data = request.json
        image_url = data.get('imageUrl')
        num_questions = data.get('numberOfQuestions', 20)
        
        if not image_url:
            return jsonify({
                'success': False,
                'error': 'Image URL is required'
            }), 400
        
        print(f"Processing OMR: {image_url}")
        print(f"Number of questions: {num_questions}")
        
        # Download image
        image = download_image(image_url)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Failed to download or decode image'
            }), 400
        
        print(f"Image downloaded: {image.shape}")
        
        # Process OMR sheet
        answers, total_bubbles, total_questions = detect_filled_bubbles(image, num_questions)
        
        print(f"Detected {total_bubbles} bubbles, {total_questions} question rows")
        print(f"First 5 answers: {answers[:5]}")
        
        return jsonify({
            'success': True,
            'answers': answers,
            'total_bubbles_detected': total_bubbles,
            'total_questions_detected': total_questions,
            'studentName': 'Unknown',
            'studentCode': ''
        })
        
    except Exception as e:
        print(f"Error processing OMR: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Run the server
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

---

