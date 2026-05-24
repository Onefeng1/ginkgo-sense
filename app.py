# -*- coding: utf-8 -*-
"""
GinkgoSense Web应用
白果智能识别系统 - Flask Web服务
"""
import os
import uuid
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image

from config import WEB_CONFIG, MODEL_CONFIG
from model.classifier import GinkgoClassifier

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = WEB_CONFIG['max_upload_size']

# 初始化分类器
classifier = None

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_classifier():
    global classifier
    if classifier is None:
        weights_path = os.path.join('weights', 'best_model.pth')
        if os.path.exists(weights_path):
            classifier = GinkgoClassifier(weights=weights_path)
        else:
            # 使用随机初始化的模型（演示模式）
            classifier = GinkgoClassifier()
            print("注意：未找到训练好的模型权重，使用演示模式")
    return classifier


@app.route('/')
def index():
    return render_template('index.html', classes=MODEL_CONFIG['class_names_cn'])


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    # 验证文件类型
    ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if ext not in WEB_CONFIG['allowed_extensions']:
        return jsonify({'error': f'不支持的文件格式: .{ext}'}), 400

    try:
        # 保存临时文件
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        # 图片预处理验证
        img = Image.open(filepath)
        img.verify()

        # 预测
        model = get_classifier()
        start_time = time.time()
        result = model.predict(filepath)
        elapsed = time.time() - start_time

        result['elapsed_ms'] = round(elapsed * 1000, 1)
        result['image_size'] = f"{img.size[0]}x{img.size[1]}"

        # 清理临时文件
        os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'识别失败: {str(e)}'}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    if 'files' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    files = request.files.getlist('files')
    model = get_classifier()
    results = []

    for file in files:
        if file.filename == '':
            continue
        try:
            ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
            filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(UPLOAD_DIR, filename)
            file.save(filepath)

            result = model.predict(filepath)
            result['filename'] = file.filename
            results.append(result)

            os.remove(filepath)
        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})

    return jsonify({'results': results, 'total': len(results)})


@app.route('/info')
def info():
    return jsonify({
        'project': 'GinkgoSense',
        'version': '1.0.0',
        'model': MODEL_CONFIG['backbone'],
        'classes': MODEL_CONFIG['class_names_cn'],
        'description': '白果智能识别系统',
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  GinkgoSense - 白果智能识别系统")
    print("  访问 http://localhost:5000")
    print("=" * 50)
    app.run(
        host=WEB_CONFIG['host'],
        port=WEB_CONFIG['port'],
        debug=WEB_CONFIG['debug'],
    )
