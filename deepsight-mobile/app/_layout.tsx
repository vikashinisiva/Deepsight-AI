// import '../global.css';
import { Stack } from 'expo-router';
import 'react-native-reanimated';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <Stack
        screenOptions={{
          headerStyle: {
            backgroundColor: '#1e3a8a',
          },
          headerTintColor: '#ffffff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen name="camera" options={{ title: 'Camera' }} />
        <Stack.Screen name="analysis" options={{ title: 'Analysis Results' }} />
        <Stack.Screen name="history" options={{ title: 'Analysis History' }} />
        <Stack.Screen name="settings" options={{ title: 'Settings' }} />
        <Stack.Screen name="+not-found" />
      </Stack>
      <StatusBar style="auto" />
    </SafeAreaProvider>
  );
}
