import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { router } from 'expo-router';

export default function NotFoundScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      
      <View style={styles.content}>
        <Ionicons name="help-circle-outline" size={80} color="#64748b" />
        
        <Text style={styles.title}>Page Not Found</Text>
        <Text style={styles.subtitle}>
          The page you're looking for doesn't exist or has been moved.
        </Text>
        
        <TouchableOpacity 
          style={styles.homeButton} 
          onPress={() => router.replace('/(tabs)')}
        >
          <Ionicons name="home" size={20} color="white" />
          <Text style={styles.homeButtonText}>Go to Home</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 32,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1e293b',
    marginTop: 24,
    marginBottom: 12,
  },
  subtitle: {
    fontSize: 16,
    color: '#64748b',
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 32,
  },
  homeButton: {
    backgroundColor: '#3b82f6',
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    gap: 8,
  },
  homeButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});
