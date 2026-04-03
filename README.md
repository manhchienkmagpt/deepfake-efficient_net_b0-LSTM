# Deepfake Detection Using EfficientNet-B0 and LSTM

Một dự án hoàn chỉnh để phát hiện deepfake sử dụng mô hình CNN-LSTM kết hợp EfficientNet-B0 backbone và LSTM sequence modeling.

## Giới thiệu

Dự án này sử dụng mô hình học sâu (Deep Learning) để phát hiện các video deepfake. Mô hình kết hợp:
- **CNN Backbone**: EfficientNet-B0 để trích xuất đặc trưng không gian từ từng frame
- **LSTM**: Để mô hình hóa mối quan hệ thời gian giữa các frame
- **Face Detection**: MTCNN để phát hiện và cắt vùng khuôn mặt

## Yêu cầu Hệ Thống

- Python >= 3.8
- GPU NVIDIA (CUDA 11.8 trở lên) - Khuyến nghị
- RAM >= 16GB
- Ổ cứng >= 100GB (để chứa dataset)

## Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd deepfake2
```

### 2. Tạo Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Chuẩn bị Dataset

Dự án sử dụng FaceForensics++ dataset. Bạn cần:

1. **Tải dataset**: Tải video từ [FaceForensics++](https://github.com/ondyari/FaceForensics)
2. **Tạo CSV files** cho mỗi loại video:
   - `original.csv` - Videos gốc
   - `Deepfakes.csv` - Deepfakes
   - `Face2Face.csv` - Face2Face
   - `FaceShifter.csv` - FaceShifter
   - `FaceSwap.csv` - FaceSwap
   - `NeuralTextures.csv` - NeuralTextures

3. **Format CSV file**:
   ```
   Unnamed: 0,File Path,Label,Frame Count,Width,Height,Codec
   0,path/to/video1.mp4,REAL,300,1920,1080,h264
   1,path/to/video2.mp4,FAKE,300,1920,1080,h264
   ```

## Cách Chạy

### 1. Mở Jupyter Notebook

```bash
jupyter notebook deepfake_efficientnetb0_LSTM.ipynb
```

### 2. Cấu hình Parameters

Trong cell đầu tiên (Section 1. Configuration), cập nhật các đường dẫn:

```python
ROOT_DIR = r"path/to/your/video_folder"

ORIGINAL_CSV = r"path/to/original.csv"
DEEPFAKES_CSV = r"path/to/Deepfakes.csv"
FACE2FACE_CSV = r"path/to/Face2Face.csv"
FACESHIFTER_CSV = r"path/to/FaceShifter.csv"
FACESWAP_CSV = r"path/to/FaceSwap.csv"
NEURALTEXTURES_CSV = r"path/to/NeuralTextures.csv"
```

### 3. Các Hyperparameters quan trọng

```python
IMG_SIZE = 380                  # Kích thước input (EfficientNet-B0)
NUM_FRAMES = 32                # Số frame lấy từ mỗi video
BATCH_SIZE = 8                 # Batch size cho training
EPOCHS = 30                    # Số epoch training
LEARNING_RATE = 1e-4           # Learning rate
PATIENCE = 5                   # Early stopping patience
MODEL_HIDDEN_SIZE = 256        # LSTM hidden size
MODEL_NUM_LAYERS = 1           # Số LSTM layers
MODEL_DROPOUT = 0.3            # Dropout rate
```

### 4. Chạy Training

Chạy các cell theo thứ tự từ trên xuống dưới:

1. **Cell 1-2**: Configuration & Imports
2. **Cell 3-4**: Transforms & Utils
3. **Cell 5-6**: Dataset & DataLoader
4. **Cell 7**: Model Architecture
5. **Cell 8**: Training Functions
6. **Cell 9**: Main Training Loop
7. **Cell 10**: Evaluation & Testing

### 5. Sau khi Training

Model sẽ được lưu tại đường dẫn được định nghĩa trong `SAVE_PATH`:
```python
SAVE_PATH = "best_efficientb0.pth"
```

## Các Tính năng Chính

✅ **Face Detection**: Sử dụng MTCNN để tự động phát hiện và cắt khuôn mặt
✅ **Data Augmentation**: Áp dụng augmentation để tăng kích thước dataset
✅ **Multi-method Fake**: Hỗ trợ nhiều phương pháp tạo deepfake (Face2Face, Deepfakes, FaceShifter, FaceSwap, NeuralTextures)
✅ **Temporal Modeling**: Sử dụng LSTM để mô hình hóa thông tin thời gian
✅ **Metrics**: ROC-AUC, Precision, Recall, F1-score

## Output & Metrics

Sau khi training, bạn sẽ nhận được:

- **Model Checkpoint**: `best_efficientb4.pth`
- **Metrics**:
  - ROC-AUC Score
  - Precision, Recall, F1-Score
  - Training/Validation Loss
  - Confusion Matrix

## Tối ưu Hóa & Tuning

### GPU Memory

Nếu bị out-of-memory, hãy giảm:
- `BATCH_SIZE` từ 8 → 4 hoặc 2
- `NUM_FRAMES` từ 32 → 16
- `IMG_SIZE` từ 380 → 224

### Tốc độ Training

- Tắt face detection nếu không cần: `use_face_detection=False`
- Giảm `NUM_WORKERS` nếu bị disk I/O bottleneck

## Troubleshooting

### 1. "CUDA out of memory"
```python
BATCH_SIZE = 4  # Giảm batch size
torch.cuda.empty_cache()
```

### 2. "Video file not found"
- Kiểm tra đường dẫn trong CSV files
- Đảm bảo `ROOT_DIR` đúng

### 3. "MTCNN face not detected"
- Video chứa khuôn mặt quay hết đó không rõ ràng
- Hãy đặt `use_face_detection=False` để sử dụng toàn bộ frame

### 4. "CUDA not available"
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Nếu False, cài đặt CUDA
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Kết Quả Kỳ Vọng

Với đầy đủ training:
- **Accuracy**: 95%+
- **ROC-AUC**: 0.98+
- **F1-Score**: 0.95+

## File Structure

```
deepfake2/
├── deepfake_efficientnetb0_LSTM.ipynb  # Main notebook
├── requirements.txt                    # Python dependencies
├── README.md                          # Documentation (this file)
└── best_efficientb0.pth              # Trained model (sau khi training)
```

## References

- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [MTCNN Face Detection](https://arxiv.org/abs/1604.02878)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

MIT License - Xem file LICENSE để chi tiết

## Contact & Support

Nếu có bất kỳ câu hỏi hoặc vấn đề, hãy tạo Issue trên repository.

---

**Lưu ý**: Dự án này chỉ dùng cho mục đích nghiên cứu và giáo dục. Hãy tuân thủ các quy định pháp luật khi sử dụng.
