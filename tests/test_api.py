# -*- coding: utf-8 -*-
"""
Web API接口测试
"""
import os
import sys
import io
import json
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app


@pytest.fixture
def client():
    """创建Flask测试客户端"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_jpeg():
    """生成JPEG格式测试图片"""
    img = Image.fromarray(__import__('numpy').random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf


@pytest.fixture
def sample_png():
    """生成PNG格式测试图片"""
    img = Image.fromarray(__import__('numpy').random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


class TestIndexPage:
    """首页测试"""

    def test_index_returns_200(self, client):
        response = client.get('/')
        assert response.status_code == 200

    def test_index_contains_title(self, client):
        response = client.get('/')
        assert b'GinkgoSense' in response.data


class TestInfoEndpoint:
    """信息接口测试"""

    def test_info_returns_json(self, client):
        response = client.get('/info')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['project'] == 'GinkgoSense'
        assert 'classes' in data
        assert 'model' in data


class TestPredictEndpoint:
    """预测接口测试"""

    def test_predict_no_file(self, client):
        response = client.post('/predict')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_predict_empty_filename(self, client):
        data = {'file': (io.BytesIO(b''), '')}
        response = client.post('/predict', data=data, content_type='multipart/form-data')
        assert response.status_code == 400

    def test_predict_invalid_extension(self, client):
        data = {'file': (io.BytesIO(b'test'), 'test.exe')}
        response = client.post('/predict', data=data, content_type='multipart/form-data')
        assert response.status_code == 400

    def test_predict_valid_image(self, client, sample_jpeg):
        data = {'file': (sample_jpeg, 'test.jpg')}
        response = client.post('/predict', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'label' in result
        assert 'confidence' in result
        assert 'elapsed_ms' in result

    def test_predict_valid_png(self, client, sample_png):
        data = {'file': (sample_png, 'test.png')}
        response = client.post('/predict', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'label' in result

    def test_predict_result_structure(self, client, sample_jpeg):
        data = {'file': (sample_jpeg, 'test.jpg')}
        response = client.post('/predict', data=data, content_type='multipart/form-data')
        result = json.loads(response.data)
        expected_keys = {'label', 'label_cn', 'confidence', 'probabilities'}
        assert expected_keys.issubset(set(result.keys()))
        assert result['label'] in ['intact', 'cracked', 'broken']
        assert isinstance(result['probabilities'], dict)


class TestBatchPredictEndpoint:
    """批量预测接口测试"""

    def test_batch_predict_no_files(self, client):
        response = client.post('/batch_predict')
        assert response.status_code == 400

    def test_batch_predict_multiple_images(self, client, sample_jpeg, sample_png):
        data = {
            'files': [
                (sample_jpeg, 'test1.jpg'),
                (sample_png, 'test2.png'),
            ]
        }
        response = client.post('/batch_predict', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        result = json.loads(response.data)
        assert 'results' in result
        assert result['total'] == 2
