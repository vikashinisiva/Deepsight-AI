import React, { useState, useRef } from 'react';
import { View, Text, TouchableOpacity, Alert, StyleSheet, Platform, Linking } from 'react-native';
import { Camera, CameraView, CameraType, FlashMode, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as MediaLibrary from 'expo-media-library';
import { Audio } from 'expo-av';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { router, useLocalSearchParams } from 'expo-router';

export default function CaptureScreen() {
  const { mode } = useLocalSearchParams<{ mode?: string }>();
  const [facing, setFacing] = useState<CameraType>('back');
  const [flash, setFlash] = useState<FlashMode>('off');
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [permission, requestPermission] = useCameraPermissions();
  const [mediaPermission, requestMediaPermission] = ImagePicker.useMediaLibraryPermissions();
  const [audioPermission, setAudioPermission] = useState<boolean | null>(null);
  const [permissionsChecked, setPermissionsChecked] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const recordingTimer = useRef<NodeJS.Timeout | null>(null);

  // Determine if we're in deepfake or video mode
  const isVideoMode = mode === 'deepfake' || mode === 'video';

  // Check all permissions when component mounts
  React.useEffect(() => {
    checkAllPermissions();
    return () => {
      if (recordingTimer.current) {
        clearInterval(recordingTimer.current);
      }
    };
  }, []);

  const checkAllPermissions = async () => {
    try {
      // Check audio permission for video recording
      const audioStatus = await Audio.getPermissionsAsync();
      setAudioPermission(audioStatus.granted);
      setPermissionsChecked(true);
    } catch (error) {
      console.warn('Failed to check audio permissions:', error);
      setAudioPermission(false);
      setPermissionsChecked(true);
    }
  };

  const requestAllPermissions = async () => {
    const results = {
      camera: false,
      media: false,
      audio: false
    };

    try {
      // Request camera permission
      if (!permission?.granted) {
        const cameraResult = await requestPermission();
        results.camera = cameraResult.granted;
      } else {
        results.camera = true;
      }

      // Request media library permission
      if (!mediaPermission?.granted) {
        const mediaResult = await requestMediaPermission();
        results.media = mediaResult.granted;
      } else {
        results.media = true;
      }

      // Request audio permission for video recording
      if (!audioPermission) {
        const audioResult = await Audio.requestPermissionsAsync();
        results.audio = audioResult.granted;
        setAudioPermission(audioResult.granted);
      } else {
        results.audio = true;
      }

      // Show results to user
      const granted = Object.values(results).filter(Boolean).length;
      const total = Object.keys(results).length;

      if (granted === total) {
        Alert.alert(
          '✅ Permissions Granted', 
          'All permissions have been granted. You can now use all camera and media features.',
          [{ text: 'OK' }]
        );
      } else {
        const deniedPermissions = [];
        if (!results.camera) deniedPermissions.push('Camera');
        if (!results.media) deniedPermissions.push('Media Library');
        if (!results.audio) deniedPermissions.push('Microphone');

        Alert.alert(
          '⚠️ Some Permissions Denied',
          `The following permissions were denied: ${deniedPermissions.join(', ')}.\n\nSome features may not work properly. You can grant these permissions later in your device settings.`,
          [
            { text: 'OK' },
            { text: 'Open Settings', onPress: () => Linking.openSettings() }
          ]
        );
      }
    } catch (error) {
      console.error('Error requesting permissions:', error);
      Alert.alert('Error', 'Failed to request permissions. Please try again.');
    }
  };

  const showPermissionDeniedAlert = (permissionType: string) => {
    Alert.alert(
      `${permissionType} Permission Required`,
      `DeepSight AI needs ${permissionType.toLowerCase()} access to function properly. Please grant permission in your device settings.`,
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Open Settings', onPress: () => Linking.openSettings() }
      ]
    );
  };

  // Show loading state while checking permissions
  if (!permissionsChecked) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <Ionicons name="shield-checkmark-outline" size={80} color="#3b82f6" />
          <Text style={styles.permissionTitle}>Checking Permissions...</Text>
          <Text style={styles.permissionText}>
            Please wait while we check your device permissions.
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  // Show comprehensive permission request screen if any permission is missing
  if (!permission?.granted || !mediaPermission?.granted || !audioPermission) {
    const missingPermissions = [];
    if (!permission?.granted) missingPermissions.push('📷 Camera');
    if (!mediaPermission?.granted) missingPermissions.push('📁 Media Library');
    if (!audioPermission) missingPermissions.push('🎤 Microphone');

    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <Ionicons name="shield-outline" size={80} color="#ef4444" />
          <Text style={styles.permissionTitle}>Permissions Required</Text>
          <Text style={styles.permissionText}>
            DeepSight AI needs the following permissions to provide full functionality:
          </Text>
          
          <View style={styles.permissionList}>
            {missingPermissions.map((permission, index) => (
              <View key={index} style={styles.permissionItem}>
                <Text style={styles.permissionItemText}>{permission}</Text>
                <Ionicons name="close-circle" size={24} color="#ef4444" />
              </View>
            ))}
          </View>

          <View style={styles.permissionDescription}>
            <Text style={styles.permissionDescText}>
              • <Text style={styles.bold}>Camera:</Text> Capture photos and videos for analysis
            </Text>
            <Text style={styles.permissionDescText}>
              • <Text style={styles.bold}>Media Library:</Text> Select existing photos and videos
            </Text>
            <Text style={styles.permissionDescText}>
              • <Text style={styles.bold}>Microphone:</Text> Record audio with videos
            </Text>
          </View>

          <TouchableOpacity style={styles.permissionButton} onPress={requestAllPermissions}>
            <Ionicons name="shield-checkmark" size={24} color="white" style={styles.buttonIcon} />
            <Text style={styles.permissionButtonText}>Grant Permissions</Text>
          </TouchableOpacity>

          <TouchableOpacity 
            style={styles.skipButton} 
            onPress={() => {
              Alert.alert(
                'Limited Functionality',
                'Without these permissions, some features may not work properly. You can grant permissions later in Settings.',
                [
                  { text: 'Continue Anyway', onPress: () => setPermissionsChecked(true) },
                  { text: 'Grant Permissions', onPress: requestAllPermissions }
                ]
              );
            }}
          >
            <Text style={styles.skipButtonText}>Skip for Now</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  if (!mediaPermission) {
    return <View />;
  }

  if (!mediaPermission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <Ionicons name="images-outline" size={80} color="#64748b" />
          <Text style={styles.permissionTitle}>Media Library Access Required</Text>
          <Text style={styles.permissionText}>
            DeepSight AI needs access to your media library to select images and videos for analysis.
          </Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestMediaPermission}>
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.permissionContainer}>
          <Ionicons name="camera-outline" size={80} color="#64748b" />
          <Text style={styles.permissionTitle}>Camera Access Required</Text>
          <Text style={styles.permissionText}>
            DeepSight AI needs access to your camera to capture images for analysis.
          </Text>
          <TouchableOpacity style={styles.permissionButton} onPress={requestPermission}>
            <Text style={styles.permissionButtonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  function toggleCameraType() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  function toggleFlash() {
    setFlash(current => 
      current === 'off' ? 'on' : 'off'
    );
  }

  async function takePicture() {
    if (!permission?.granted) {
      Alert.alert(
        'Camera Permission Required',
        'To take photos, please grant camera permission.',
        [
          { text: 'Cancel', style: 'cancel' },
          { 
            text: 'Grant Permission', 
            onPress: async () => {
              const result = await requestPermission();
              if (result.granted) {
                takePicture(); // Retry taking picture
              } else {
                showPermissionDeniedAlert('Camera');
              }
            }
          }
        ]
      );
      return;
    }

    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: false,
        });
        
        // Navigate to analysis screen with the photo
        router.push({
          pathname: '/analysis',
          params: { imageUri: photo.uri }
        });
      } catch (error) {
        Alert.alert('Error', 'Failed to take picture');
      }
    }
  }

  async function pickFromGallery() {
    console.log('📱 Picking from gallery...');
    
    if (!mediaPermission?.granted) {
      Alert.alert(
        'Media Library Permission Required',
        'To select photos and videos, please grant media library permission.',
        [
          { text: 'Cancel', style: 'cancel' },
          { 
            text: 'Grant Permission', 
            onPress: async () => {
              const result = await requestMediaPermission();
              if (result.granted) {
                pickFromGallery(); // Retry picking
              } else {
                showPermissionDeniedAlert('Media Library');
              }
            }
          }
        ]
      );
      return;
    }

    try {
      const mediaTypes = isVideoMode 
        ? 'videos' 
        : 'images';
        
      console.log(`🎥 Launching ${isVideoMode ? 'video' : 'image'} picker...`);
      
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes,
        allowsEditing: false, // Disable editing to avoid issues
        quality: 1.0, // Higher quality
        videoMaxDuration: 60, // Increase to 60 seconds
      });

      console.log('📋 Picker result:', result);

      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        console.log('📄 Selected asset:', {
          uri: asset.uri,
          type: asset.type,
          duration: asset.duration,
          fileSize: asset.fileSize
        });

        // Navigate to analysis immediately
        router.push({
          pathname: '/analysis',
          params: { 
            imageUri: asset.uri,
            mediaType: asset.type || (isVideoMode ? 'video' : 'image'),
            isVideo: isVideoMode ? 'true' : 'false'
          }
        });
      } else {
        console.log('❌ No media selected or operation canceled');
      }
    } catch (error) {
      console.error('❌ Gallery picker error:', error);
      Alert.alert(
        'Error', 
        `Failed to pick ${isVideoMode ? 'video' : 'image'}: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  async function startVideoRecording() {
    console.log('🎬 Starting video recording...');
    
    // Check audio permission before recording
    if (!audioPermission) {
      Alert.alert(
        'Microphone Permission Required',
        'To record video with audio, please grant microphone permission.',
        [
          { text: 'Cancel', style: 'cancel' },
          { 
            text: 'Grant Permission', 
            onPress: async () => {
              const result = await Audio.requestPermissionsAsync();
              if (result.granted) {
                setAudioPermission(true);
                startVideoRecording(); // Retry recording
              } else {
                showPermissionDeniedAlert('Microphone');
              }
            }
          }
        ]
      );
      return;
    }
    
    if (cameraRef.current && !isRecording) {
      try {
        setIsRecording(true);
        setRecordingDuration(0);
        
        // Start the timer
        recordingTimer.current = setInterval(() => {
          setRecordingDuration(prev => {
            const newDuration = prev + 1;
            // Auto-stop at 30 seconds
            if (newDuration >= 30) {
              stopVideoRecording();
              return 30;
            }
            return newDuration;
          });
        }, 1000) as unknown as NodeJS.Timeout;

        console.log('📹 Recording video...');
        const video = await cameraRef.current.recordAsync({
          maxDuration: 30, // 30 seconds max
        });
        
        console.log('✅ Video recorded:', video);
        
        if (video) {
          router.push({
            pathname: '/analysis',
            params: { 
              imageUri: video.uri,
              mediaType: 'video',
              isVideo: 'true'
            }
          });
        }
      } catch (error) {
        console.error('❌ Recording error:', error);
        Alert.alert('Error', `Failed to record video: ${error instanceof Error ? error.message : 'Unknown error'}`);
        setIsRecording(false);
        if (recordingTimer.current) {
          clearInterval(recordingTimer.current);
        }
      }
    }
  }

  async function stopVideoRecording() {
    if (cameraRef.current && isRecording) {
      try {
        await cameraRef.current.stopRecording();
        setIsRecording(false);
        setRecordingDuration(0);
        if (recordingTimer.current) {
          clearInterval(recordingTimer.current);
        }
      } catch (error) {
        console.warn('Error stopping video recording:', error);
      }
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />
      
      <CameraView style={styles.camera} facing={facing} flash={flash} ref={cameraRef}>
        {/* Header Controls */}
        <View style={styles.header}>
          <TouchableOpacity style={styles.headerButton} onPress={toggleFlash}>
            <Ionicons 
              name={flash === 'off' ? "flash-off" : "flash"} 
              size={24} 
              color="white" 
            />
          </TouchableOpacity>
          
          <TouchableOpacity style={styles.headerButton} onPress={() => router.back()}>
            <Ionicons name="close" size={24} color="white" />
          </TouchableOpacity>
        </View>

        {/* Recording Timer */}
        {isVideoMode && isRecording && (
          <View style={styles.recordingTimer}>
            <View style={styles.recordingIndicator} />
            <Text style={styles.recordingText}>
              REC {Math.floor(recordingDuration / 60)}:{(recordingDuration % 60).toString().padStart(2, '0')}
            </Text>
          </View>
        )}

        {/* Bottom Controls */}
        <View style={styles.bottomControls}>
          <TouchableOpacity style={styles.galleryButton} onPress={pickFromGallery}>
            <Ionicons name={isVideoMode ? "videocam" : "images"} size={28} color="white" />
          </TouchableOpacity>

          {isVideoMode ? (
            <TouchableOpacity 
              style={[styles.captureButton, isRecording && styles.recordingButton]} 
              onPress={isRecording ? stopVideoRecording : startVideoRecording}
              disabled={isRecording && recordingDuration < 2} // Prevent immediate stop for 2s
            >
              <View style={[styles.captureButtonInner, isRecording && styles.recordingButtonInner]} />
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={styles.captureButton} onPress={takePicture}>
              <View style={styles.captureButtonInner} />
            </TouchableOpacity>
          )}

          <TouchableOpacity style={styles.flipButton} onPress={toggleCameraType}>
            <Ionicons name="camera-reverse" size={28} color="white" />
          </TouchableOpacity>
        </View>
      </CameraView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  camera: {
    flex: 1,
  },
  permissionContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
    backgroundColor: '#f8fafc',
  },
  permissionTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1e293b',
    marginTop: 24,
    marginBottom: 16,
    textAlign: 'center',
  },
  permissionText: {
    fontSize: 16,
    color: '#64748b',
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 32,
  },
  permissionButton: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
  },
  permissionButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingTop: 60,
    paddingBottom: 20,
  },
  headerButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    padding: 12,
    borderRadius: 24,
  },
  bottomControls: {
    position: 'absolute',
    bottom: 60,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 60,
  },
  galleryButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    padding: 16,
    borderRadius: 24,
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'white',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#3b82f6',
  },
  flipButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    padding: 16,
    borderRadius: 24,
  },
  recordingTimer: {
    position: 'absolute',
    top: 80,
    left: 20,
    backgroundColor: 'rgba(255, 0, 0, 0.8)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    flexDirection: 'row',
    alignItems: 'center',
    zIndex: 10,
  },
  recordingIndicator: {
    width: 8,
    height: 8,
    backgroundColor: 'white',
    borderRadius: 4,
    marginRight: 8,
  },
  recordingText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  recordingButton: {
    backgroundColor: '#ef4444',
  },
  recordingButtonInner: {
    backgroundColor: 'white',
    width: 40,
    height: 40,
    borderRadius: 4,
  },
  permissionList: {
    width: '100%',
    marginVertical: 20,
  },
  permissionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    paddingHorizontal: 16,
    marginVertical: 4,
    backgroundColor: '#fee2e2',
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#ef4444',
  },
  permissionItemText: {
    fontSize: 16,
    color: '#7f1d1d',
    fontWeight: '500',
  },
  permissionDescription: {
    width: '100%',
    backgroundColor: '#f1f5f9',
    padding: 16,
    borderRadius: 8,
    marginVertical: 16,
  },
  permissionDescText: {
    fontSize: 14,
    color: '#475569',
    lineHeight: 20,
    marginVertical: 2,
  },
  bold: {
    fontWeight: 'bold',
    color: '#1e293b',
  },
  buttonIcon: {
    marginRight: 8,
  },
  skipButton: {
    marginTop: 16,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#cbd5e1',
    backgroundColor: 'transparent',
  },
  skipButtonText: {
    color: '#64748b',
    fontSize: 14,
    fontWeight: '500',
    textAlign: 'center',
  },
});
