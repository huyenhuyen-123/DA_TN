import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
interface LibraryPermissionRequestProps {
  onRequestPermission: () => void;
}

export const LibraryPermissionRequest = ({
  onRequestPermission,
}: LibraryPermissionRequestProps) => {
  return (
    <View style={styles.permissionContainer}>
      <Text style={styles.permissionTitle}>
        Cấp quyền truy cập thư viện ảnh
      </Text>
      <Text style={styles.permissionDesc}>
        Ứng dụng cần quyền truy cập thư viện để lưu ảnh bạn đã chụp.
      </Text>
      <TouchableOpacity
        style={[styles.button, styles.permissionButton]}
        onPress={onRequestPermission}
      >
        <Text style={styles.buttonText}>Cho phép truy cập</Text>
      </TouchableOpacity>
    </View>
  );
};

interface PermissionRequestProps {
  onRequestPermission: () => void;
}

export const PermissionRequest = ({ onRequestPermission }: PermissionRequestProps) => {
  return (
    <View style={styles.permissionContainer}>
      <Text style={styles.permissionTitle}>
        Cấp quyền truy cập Camera
      </Text>
      <Text style={styles.permissionDesc}>
        Ứng dụng cần quyền truy cập camera để nhận diện mệnh giá tiền
      </Text>
      <TouchableOpacity
        style={[styles.button, styles.permissionButton]}
        onPress={onRequestPermission}
      >
        <Text style={styles.buttonText}>Cho phép truy cập</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#fff',
  },
  permissionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 20,
    textAlign: 'center',
    color: '#333',
  },
  permissionDesc: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 30,
    color: '#666',
    lineHeight: 24,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 15,
    borderRadius: 8,
    minWidth: 200,
    justifyContent: 'center',
  },
  permissionButton: {
    backgroundColor: '#2196F3',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
}); 