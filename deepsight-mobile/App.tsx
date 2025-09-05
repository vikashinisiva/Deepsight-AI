import React from 'react';
import { Stack } from 'expo-router';

// Import the test script
import './services/temp_test.ts';

export default function App() {
  return (
    <Stack>
      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
      <Stack.Screen name="+not-found" />
    </Stack>
  );
}
