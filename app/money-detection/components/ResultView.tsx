import { Ionicons } from '@expo/vector-icons';
import axios from 'axios';
import { Audio } from 'expo-av';
import { BlurView } from 'expo-blur';
import * as Speech from 'expo-speech';
import { useEffect } from 'react';
import { ActivityIndicator, Image, Platform, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

interface ResultViewProps {
  capturedImage: string;
  isLoading: boolean;
  denomination: string;
  confidence: number;
  onRetry: () => void;
  API_URL: string;
}

export const ResultView = ({
  capturedImage,
  isLoading,
  denomination,
  confidence = 0,
  onRetry,
  API_URL
}: ResultViewProps) => {
  const playAudio = async (base64Audio: string) => {
    try {
      // Unload audio cũ nếu có
      const soundObject = new Audio.Sound();

      // Tạo URI từ base64
      const uri = `data:audio/mp3;base64,${base64Audio}`;

      // Load và phát audio
      await soundObject.loadAsync({ uri });
      const status = await soundObject.getStatusAsync();

      if (status.isLoaded) {
        await soundObject.playAsync();
        // Đợi audio phát xong
        soundObject.setOnPlaybackStatusUpdate(async (playbackStatus) => {
          if (playbackStatus.isLoaded && playbackStatus.didJustFinish) {
            await soundObject.unloadAsync();
          }
        });
      }
    } catch (error) {
      console.error('Error playing audio:', error);
      // Log chi tiết lỗi
      if (error instanceof Error) {
        console.error('Error details:', error.message);
      }
    }
  };

  const speakText = async (text: string) => {
    try {
      const response = await axios.post(`${API_URL}/api/tts`,
        { text },
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: 10000,
        }
      );

      if (response.data.audio) {
        await playAudio(response.data.audio);
      } else {
        throw new Error('No audio data received');
      }
    } catch (error) {
      console.error('Error with TTS:', error);
      // Fallback to expo-speech
      try {
        await Speech.speak(text, {
          language: 'vi-VN',
          pitch: 1,
          rate: 0.8
        });
      } catch (speechError) {
        console.error('Fallback speech error:', speechError);
      }
    }
  };

  useEffect(() => {
    const initAudio = async () => {
      try {
        // Yêu cầu quyền audio nếu cần
        await Audio.requestPermissionsAsync();
        // Cấu hình audio
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: false,
          playsInSilentModeIOS: true,
          shouldDuckAndroid: true,
          playThroughEarpieceAndroid: false,
          staysActiveInBackground: false,
        });
      } catch (error) {
        console.error('Error initializing audio:', error);
      }
    };

    initAudio();
  }, []);

  useEffect(() => {
    if (!isLoading && denomination) {
      const amount = denomination.replace(/[^0-9]/g, '');
      const formattedMoney = Number(amount).toLocaleString('vi-VN');
      const textToSpeak = `${formattedMoney} đồng`;

      speakText(textToSpeak);
    }
  }, [isLoading, denomination]);

  // Hàm format số tiền và độ tin cậy
  const formatResult = (denomination: string, confidence: number) => {
    // Format số tiền theo định dạng tiền tệ Việt Nam
    const amount = denomination.replace(/[^0-9]/g, '');
    const formattedMoney = Number(amount).toLocaleString('vi-VN');
    // Format độ tin cậy với 1 số thập phân
    const formattedConfidence = (confidence * 100).toFixed(1);

    return {
      money: `${formattedMoney} VNĐ`,
      confidence: `Độ tin cậy: ${formattedConfidence}%`
    };
  };

  const result = formatResult(denomination, confidence);

  return (
    <View style={styles.container}>
      <Image
        source={{ uri: capturedImage }}
        style={styles.blurredBackground}
        blurRadius={10}
      />

      <Image
        source={{ uri: capturedImage }}
        style={styles.previewImage}
      />

      <BlurView intensity={50} tint="dark" style={styles.header}>
        <Ionicons name="image" size={24} color="#fff" />
        <Text style={styles.headerText}>Kết quả nhận diện</Text>
      </BlurView>

      {isLoading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2196F3" />
          <Text style={styles.loadingText}>Đang nhận diện...</Text>
        </View>
      ) : (
        <>
          <View style={styles.controlsContainer}>
            <View style={styles.predictionContainer}>
              <Text style={styles.predictionLabel}>Kết quả nhận diện:</Text>
              <Text style={styles.predictionText}>{result.money}</Text>
              <Text style={styles.confidenceText}>{result.confidence}</Text>
            </View>
            <View style={styles.buttonGroup}>
              <TouchableOpacity
                  style={[styles.button, styles.retryButton]}
                  onPress={onRetry}
                >
                <Ionicons name="camera" size={48} color="white" />
                <Text style={styles.buttonText}>Chụp lại</Text>
              </TouchableOpacity>
            </View>
          </View>
        </>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  blurredBackground: {
    ...StyleSheet.absoluteFillObject,
    width: '100%',
    height: '100%',
  },
  previewImage: {
    flex: 1,
    resizeMode: 'contain',
  },
  retryButton: {
    backgroundColor: "#2196F3",
  },
  header: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    paddingTop: Platform.OS === 'ios' ? 50 : 16,
  },
  headerText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 8,
  },
  loadingContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.3)',
  },
  loadingText: {
    color: '#fff',
    fontSize: 16,
    marginTop: 10,
  },
  resultBox: {
    marginBottom: 20,
    alignItems: 'center',
  },
  predictionText: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 5,
  },
  confidenceText: {
    fontSize: 16,
    color: '#fff',
    opacity: 0.8,
    textAlign: 'center',
  },
  controlsContainer: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: 20,
    paddingBottom: Platform.OS === 'ios' ? 30 : 20,
  },
  buttonGroup: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    gap: 20,
  },
  button: {
    flexDirection: "row",
    alignItems: "center",
    padding: 48,
    borderRadius: 25,
    minWidth: 120,
    justifyContent: "center",
    width: '100%'
  },
  detailButton: {
    backgroundColor: 'rgba(76, 175, 80, 0.9)',
  },
  buttonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
    marginLeft: 8,
  },
  predictionContainer: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
    alignItems: 'center',
  },
  predictionLabel: {
    fontSize: 16,
    color: '#fff',
    marginBottom: 5,
    opacity: 0.8,
  }
}); 