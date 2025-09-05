/**
 * Permission Manager Component
 * Handles comprehensive permission requests for camera, media library, and microphone
 */

import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, Alert, StyleSheet, Platform, Linking } from 'react-native';
import { useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as MediaLibrary from 'expo-media-library';
import { Audio } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';

interface PermissionState {
  camera: boolean | null;
  media: boolean | null;
  audio: boolean | null;
}

interface PermissionManagerProps {
  onPermissionsGranted?: () => void;
  onPermissionsDenied?: (deniedPermissions: string[]) => void;
  showSkipOption?: boolean;
  children?: React.ReactNode;
}

export default function PermissionManager({ 
  onPermissionsGranted, 
  onPermissionsDenied, 
  showSkipOption = true,
  children 
}: PermissionManagerProps) {
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [mediaPermission, requestMediaPermission] = ImagePicker.useMediaLibraryPermissions();
  const [permissions, setPermissions] = useState<PermissionState>({
    camera: null,
    media: null,
    audio: null
  });
  const [isLoading, setIsLoading] = useState(true);
  const [permissionsChecked, setPermissionsChecked] = useState(false);

  useEffect(() => {
    checkAllPermissions();
  }, []);

  const checkAllPermissions = async () => {
    setIsLoading(true);
    try {
      // Check audio permission
      const audioStatus = await Audio.getPermissionsAsync();
      
      // Check media library permission
      const mediaStatus = await MediaLibrary.getPermissionsAsync();

      const currentPermissions = {
        camera: cameraPermission?.granted || false,
        media: mediaPermission?.granted || mediaStatus.granted || false,
        audio: audioStatus.granted || false
      };

      setPermissions(currentPermissions);
      setPermissionsChecked(true);

      // Check if all permissions are granted
      const allGranted = Object.values(currentPermissions).every(Boolean);
      if (allGranted && onPermissionsGranted) {
        onPermissionsGranted();
      }
    } catch (error) {
      console.warn('Failed to check permissions:', error);
      setPermissions({ camera: false, media: false, audio: false });
      setPermissionsChecked(true);
    } finally {
      setIsLoading(false);
    }
  };

  const requestAllPermissions = async () => {
    const results = {
      camera: false,
      media: false,
      audio: false
    };

    try {
      setIsLoading(true);

      // Request camera permission
      if (!cameraPermission?.granted) {
        const cameraResult = await requestCameraPermission();
        results.camera = cameraResult.granted;
      } else {
        results.camera = true;
      }

      // Request media library permission
      if (!mediaPermission?.granted) {
        const mediaResult = await requestMediaPermission();
        results.media = mediaResult.granted;
        
        // Also request MediaLibrary permission for better compatibility
        if (!mediaResult.granted) {
          const mediaLibResult = await MediaLibrary.requestPermissionsAsync();
          results.media = mediaLibResult.granted;
        }
      } else {
        results.media = true;
      }

      // Request audio permission
      if (!permissions.audio) {
        const audioResult = await Audio.requestPermissionsAsync();
        results.audio = audioResult.granted;
      } else {
        results.audio = true;
      }

      setPermissions(results);

      // Check results
      const granted = Object.values(results).filter(Boolean).length;
      const total = Object.keys(results).length;
      const deniedPermissions = [];

      if (!results.camera) deniedPermissions.push('Camera');
      if (!results.media) deniedPermissions.push('Media Library');
      if (!results.audio) deniedPermissions.push('Microphone');

      if (granted === total) {
        Alert.alert(
          '✅ All Permissions Granted',
          'Great! All permissions have been granted. You can now use all features of DeepSight AI.',
          [{ text: 'Continue', onPress: onPermissionsGranted }]
        );
      } else if (granted > 0) {
        Alert.alert(
          '⚠️ Some Permissions Granted',
          `${granted}/${total} permissions granted. Denied: ${deniedPermissions.join(', ')}.\n\nSome features may be limited. You can grant the remaining permissions later in Settings.`,
          [
            { text: 'Continue Anyway', onPress: onPermissionsGranted },
            { text: 'Open Settings', onPress: () => Linking.openSettings() }
          ]
        );
      } else {
        Alert.alert(
          '❌ Permissions Required',
          `All permissions were denied: ${deniedPermissions.join(', ')}.\n\nDeepSight AI requires these permissions to function properly. Please grant them in Settings.`,
          [
            { text: 'Try Again', onPress: requestAllPermissions },
            { text: 'Open Settings', onPress: () => Linking.openSettings() }
          ]
        );
        if (onPermissionsDenied) {
          onPermissionsDenied(deniedPermissions);
        }
      }
    } catch (error) {
      console.error('Error requesting permissions:', error);
      Alert.alert('Error', 'Failed to request permissions. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const openSettings = () => {
    Alert.alert(
      'Open Settings',
      'To grant permissions, go to your device Settings > Apps > DeepSight AI > Permissions, then enable Camera, Storage, and Microphone.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Open Settings', onPress: () => Linking.openSettings() }
      ]
    );
  };

  // Show loading state
  if (isLoading || !permissionsChecked) {
    return (
      <View style={styles.container}>
        <Ionicons name="shield-checkmark-outline" size={80} color="#3b82f6" />
        <Text style={styles.title}>Checking Permissions...</Text>
        <Text style={styles.description}>
          Please wait while we verify your device permissions.
        </Text>
      </View>
    );
  }

  // If all permissions are granted, show children or success message
  const allPermissionsGranted = Object.values(permissions).every(Boolean);
  if (allPermissionsGranted) {
    return children ? <>{children}</> : (
      <View style={styles.container}>
        <Ionicons name="shield-checkmark" size={80} color="#10b981" />
        <Text style={styles.title}>All Permissions Granted! ✅</Text>
        <Text style={styles.description}>
          DeepSight AI is ready to use with full functionality.
        </Text>
      </View>
    );
  }

  // Show permission request UI
  const missingPermissions = [];
  if (!permissions.camera) missingPermissions.push({ icon: '📷', name: 'Camera', key: 'camera' });
  if (!permissions.media) missingPermissions.push({ icon: '📁', name: 'Media Library', key: 'media' });
  if (!permissions.audio) missingPermissions.push({ icon: '🎤', name: 'Microphone', key: 'audio' });

  return (
    <View style={styles.container}>
      <Ionicons name="shield-outline" size={80} color="#ef4444" />
      <Text style={styles.title}>Permissions Required</Text>
      <Text style={styles.description}>
        DeepSight AI needs the following permissions to provide full functionality:
      </Text>
      
      <View style={styles.permissionList}>
        {missingPermissions.map((permission, index) => (
          <View key={index} style={styles.permissionItem}>
            <Text style={styles.permissionText}>
              {permission.icon} {permission.name}
            </Text>
            <Ionicons 
              name={permissions[permission.key as keyof PermissionState] ? "checkmark-circle" : "close-circle"} 
              size={24} 
              color={permissions[permission.key as keyof PermissionState] ? "#10b981" : "#ef4444"} 
            />
          </View>
        ))}
      </View>

      <View style={styles.featureDescription}>
        <Text style={styles.featureText}>
          • <Text style={styles.bold}>Camera:</Text> Capture photos and videos for AI analysis
        </Text>
        <Text style={styles.featureText}>
          • <Text style={styles.bold}>Media Library:</Text> Select existing photos and videos
        </Text>
        <Text style={styles.featureText}>
          • <Text style={styles.bold}>Microphone:</Text> Record audio with videos
        </Text>
      </View>

      <TouchableOpacity style={styles.primaryButton} onPress={requestAllPermissions}>
        <Ionicons name="shield-checkmark" size={24} color="white" style={styles.buttonIcon} />
        <Text style={styles.primaryButtonText}>Grant Permissions</Text>
      </TouchableOpacity>

      <TouchableOpacity style={styles.secondaryButton} onPress={openSettings}>
        <Ionicons name="settings" size={20} color="#64748b" style={styles.buttonIcon} />
        <Text style={styles.secondaryButtonText}>Open Settings</Text>
      </TouchableOpacity>

      {showSkipOption && (
        <TouchableOpacity 
          style={styles.skipButton} 
          onPress={() => {
            Alert.alert(
              'Limited Functionality',
              'Without these permissions, some features may not work properly. You can grant permissions later in Settings.',
              [
                { text: 'Continue Anyway', onPress: onPermissionsGranted },
                { text: 'Grant Permissions', onPress: requestAllPermissions }
              ]
            );
          }}
        >
          <Text style={styles.skipButtonText}>Continue with Limited Features</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
    backgroundColor: '#f8fafc',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1e293b',
    marginTop: 24,
    marginBottom: 16,
    textAlign: 'center',
  },
  description: {
    fontSize: 16,
    color: '#64748b',
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 32,
  },
  permissionList: {
    width: '100%',
    marginBottom: 24,
  },
  permissionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    paddingHorizontal: 16,
    marginVertical: 4,
    backgroundColor: 'white',
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#3b82f6',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  permissionText: {
    fontSize: 16,
    color: '#1e293b',
    fontWeight: '500',
  },
  featureDescription: {
    width: '100%',
    backgroundColor: '#f1f5f9',
    padding: 16,
    borderRadius: 8,
    marginBottom: 24,
  },
  featureText: {
    fontSize: 14,
    color: '#475569',
    lineHeight: 20,
    marginVertical: 2,
  },
  bold: {
    fontWeight: 'bold',
    color: '#1e293b',
  },
  primaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#3b82f6',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
    marginBottom: 12,
    shadowColor: '#3b82f6',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 4,
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#cbd5e1',
    marginBottom: 12,
  },
  secondaryButtonText: {
    color: '#64748b',
    fontSize: 14,
    fontWeight: '500',
  },
  skipButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  skipButtonText: {
    color: '#94a3b8',
    fontSize: 12,
    textAlign: 'center',
  },
  buttonIcon: {
    marginRight: 8,
  },
});
