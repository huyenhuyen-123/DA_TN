// Import các components con
import axios from 'axios';
import { CameraCapturedPicture, CameraView, useCameraPermissions } from 'expo-camera';
import React, { useRef, useState } from 'react';
import { View, ViewStyle } from 'react-native';
import { CameraViewComponent } from './components/CameraView';
import { PermissionRequest } from './components/PermissionRequest';
import { ResultView } from './components/ResultView';

const API_URL = 'http://192.168.1.32:8000';  // Thay API_URL bang ip hien tai

type Props = {
  style?: ViewStyle;
};
const MoneyDetection: React.FC<Props> = ({ style}) => {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<'front' | 'back'>('back');
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const [denomination, setDenomination] = useState<string>('');
  const [confidence, setConfidence] = useState<number>(0);

  // Hàm xử lý khi nhấn nút chụp
  const handleCaptureCallback  = async (photo: CameraCapturedPicture) => {
    if (!cameraRef.current) return;

    try {
      setIsLoading(true);
      setCapturedImage(photo?.uri || '');

      // Tạo form data
      const formData = new FormData();
      formData.append('file', {
        uri: photo?.uri,
        type: 'image/jpeg',
        name: 'money.jpg',
      } as any);

      // Gửi ảnh lên server
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000,
      });

      // Hiển thị kết quả
      const { denomination: denom, confidence: conf } = response.data;
      setDenomination(denom);
      setConfidence(conf);

    } catch (error) {
      console.error('Capture error:', error);
      setDenomination('');
      setConfidence(0);
    } finally {
      setIsLoading(false);
    }
  };

  // Hàm thử lại
  const handleRetry = () => {
    setCapturedImage(null);
    setDenomination('');
    setConfidence(0);
  };

  // Kiểm tra quyền camera
  if (!permission) return <View />;
  if (!permission.granted) {
    return <PermissionRequest onRequestPermission={requestPermission} />;
  }

  // Hiển thị kết quả nếu đã chụp ảnh
  if (capturedImage) {
    return (
      <>
        <ResultView
          capturedImage={capturedImage}
          isLoading={isLoading}
          denomination={denomination}
          confidence={confidence}
          onRetry={handleRetry}
          API_URL={API_URL}
        />
        
      </>
    );
  }

  // Hiển thị camera
  return (
    <CameraViewComponent
      cameraRef={cameraRef}
      facing={facing}
      onCapture={handleCaptureCallback}
    />
  );
};

export default MoneyDetection;
