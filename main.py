from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import torch
from torchvision import models
import os
from gtts import gTTS
import base64
from pydantic import BaseModel
import io

app = FastAPI()

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Định nghĩa class names
class_names = ["010000", "020000", "050000", "100000", "200000", "500000"]

# Thêm device config cho PyTorch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_model():
    # Thay thế VGG16 của TensorFlow bằng VGG16 của PyTorch
    model = models.vgg16_bn()
    model.classifier[6] = torch.nn.Linear(4096, len(class_names))
    return model

# Cập nhật phần load model
try:
    model = get_model()
    weights_path = './best_model.pth'  # Đổi đuôi file weights
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file weights tại: {weights_path}")
    
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Đã load model và weights thành công!")
    
except Exception as e:
    print(f"Lỗi khi load model: {str(e)}")
    raise HTTPException(status_code=500, detail="Không thể load model")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc ảnh
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Thêm các bước tiền xử lý
        # 1. Chuẩn hóa độ sáng và màu sắc
        # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(lab)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        # cl = clahe.apply(l)
        # limg = cv2.merge((cl,a,b))
        # image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # # 2. Giảm nhiễu
        # image = cv2.GaussianBlur(image, (5,5), 0)
        
        # # 3. Resize và chuẩn hóa
        # image = cv2.resize(image, (112, 112))
        # image = image[:, :, ::-1]  # BGR to RGB
        # image = (image.transpose((2, 0, 1)) - 127.5) * 0.007843137
        
        # # Chuyển sang tensor
        # image = np.expand_dims(image, axis=0)
        # image = torch.from_numpy(image.astype(np.float32))
        # image = image.to(device)
        
        # # Resize hình ảnh
        image = cv2.resize(image, (112, 112))

        # Chuyển đổi từ BGR sang RGB
        image = image[:, :, ::-1]

        # Chuẩn hóa giá trị pixel
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.007843137

        # Chuyển hình ảnh sang tensor
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image.astype(np.float32))

        # Đưa hình ảnh lên thiết bị (GPU hoặc CPU)
        image = image.to(device)
        # Dự đoán với threshold cao hơn
        with torch.no_grad():
            predictions = model(image)
            predictions = torch.nn.Softmax(dim=1)(predictions)
            predictions = predictions.cpu().detach().numpy()[0]
        
        confidence = float(np.max(predictions))
        predicted_class = class_names[np.argmax(predictions)]
        
        # Debug - in thông tin để kiểm tra
        print(f'Raw prediction values: {predictions}')
        print(f'Predicted class: {predicted_class} with confidence: {confidence*100:.2f}%')

        formatted_value = f"{int(predicted_class):,}"
        return {
            "denomination": formatted_value,
            "confidence": f"{confidence:.2f}"
        }
        # # Tăng ngưỡng confidence
        # if confidence > 0.85:  # Tăng từ 0.7 lên 0.85
        # else:
        #     return {
        #         "denomination": "Unknown",
        #         "confidence": f"{confidence:.2f}"
        #     }
            
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Thêm model cho request TTS
class TTSRequest(BaseModel):
    text: str

@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    try:
        tts = gTTS(text=request.text, lang='vi', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_content = base64.b64encode(audio_buffer.getvalue()).decode()
        return {"audio": audio_content}
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "API nhận diện mệnh giá tiền Việt Nam"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)