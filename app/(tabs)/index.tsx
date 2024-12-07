import MoneyDetection from '@/app/money-detection';
import { ThemedView } from '@/components/ThemedView';
import { Platform, StatusBar, StyleSheet } from 'react-native';

export default function HomeScreen() {
  return (
    <ThemedView style={styles.container}>
      <MoneyDetection style={styles.moneyDetection} />
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
  container: {
    flex: 1,
    width: '100%',
  },
  moneyDetection: {
    flex: 1,
    width: '100%',
  }
});
