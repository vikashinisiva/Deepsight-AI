import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { router } from 'expo-router';
import { useFocusEffect } from '@react-navigation/native';
import { storageService, AppStatistics } from '@/services/storageService';
import { aiService } from '@/services/aiService';

export default function HomeScreen() {
  const [stats, setStats] = useState<AppStatistics | null>(null);
  const [recentAnalyses, setRecentAnalyses] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Load statistics when screen comes into focus
  useFocusEffect(
    React.useCallback(() => {
      loadData();
    }, [])
  );

  const loadData = async () => {
    try {
      setIsLoading(true);
      const [statistics, analyses] = await Promise.all([
        storageService.getStatistics(),
        storageService.getAnalyses()
      ]);
      setStats(statistics);
      // Get the 3 most recent analyses
      setRecentAnalyses(analyses.slice(0, 3));
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTimeAgo = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return `${Math.floor(diffMins / 1440)}d ago`;
  };

  const testAPIConnection = async () => {
    try {
      const isHealthy = await aiService.healthCheck();
      Alert.alert(
        isHealthy ? 'Connection Successful' : 'Connection Failed',
        isHealthy ? 'The AI server is online and ready.' : 'Could not connect to the AI server. Check your network and server status.',
        [{ text: 'OK' }]
      );
    } catch (error) {
      Alert.alert('Error', error instanceof Error ? error.message : 'Failed to test connection', [{ text: 'OK' }]);
    }
  };

  const features = [
    {
      icon: 'shield-checkmark',
      title: 'Deepfake Detection',
      description: 'Detect manipulated images and deepfakes',
      action: () => router.push('/capture?mode=deepfake'),
    },
    {
      icon: 'videocam',
      title: 'Video Analysis',
      description: 'Analyze videos for deepfakes and manipulations',
      action: () => router.push('/capture?mode=video'),
    },
    {
      icon: 'camera',
      title: 'Image Analysis',
      description: 'General AI-powered image analysis',
      action: () => router.push('/capture'),
    },
    {
      icon: 'analytics',
      title: 'Results History',
      description: 'View your previous analysis results',
      action: () => router.push('/results'),
    },
    {
      icon: 'cloud-upload',
      title: 'Batch Processing',
      description: 'Process multiple images at once',
      action: () => {},
    },
    {
      icon: 'server',
      title: 'Test API',
      description: 'Test connection to local AI server',
      action: testAPIConnection,
    },
    {
      icon: 'settings',
      title: 'Settings',
      description: 'Customize your experience',
      action: () => router.push('/profile'),
    },
  ];

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      
      {/* Header */}
      <LinearGradient
        colors={['#1e3a8a', '#3b82f6']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <View>
            <Text style={styles.headerTitle}>DeepSight AI</Text>
            <Text style={styles.headerSubtitle}>Intelligent Image Analysis</Text>
          </View>
          <TouchableOpacity 
            style={styles.profileButton}
            onPress={() => router.push('/profile')}
          >
            <Ionicons name="person" size={24} color="white" />
          </TouchableOpacity>
        </View>
      </LinearGradient>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Quick Stats */}
        <View style={styles.statsSection}>
          <Text style={styles.sectionTitle}>Quick Overview</Text>
          <View style={styles.statsRow}>
            <View style={styles.statCard}>
              <Text style={styles.statValue}>
                {isLoading ? '--' : stats?.totalAnalyses || 0}
              </Text>
              <Text style={styles.statLabel}>Images Analyzed</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statValueGreen}>
                {isLoading ? '--' : stats ? Math.round((stats.successfulAnalyses / Math.max(stats.totalAnalyses, 1)) * 100) + '%' : '0%'}
              </Text>
              <Text style={styles.statLabel}>Success Rate</Text>
            </View>
          </View>
          {stats && stats.totalAnalyses > 0 && (
            <View style={styles.statsRow}>
              <View style={styles.statCard}>
                <Text style={styles.statValue}>
                  {Math.round(stats.averageProcessingTime)}s
                </Text>
                <Text style={styles.statLabel}>Avg Processing</Text>
              </View>
              <View style={styles.statCard}>
                <Text style={styles.statValue}>
                  {Object.keys(stats.mostDetectedObjects).length}
                </Text>
                <Text style={styles.statLabel}>Object Types</Text>
              </View>
            </View>
          )}
        </View>

        {/* Feature Grid */}
        <View style={styles.featuresSection}>
          <Text style={styles.sectionTitle}>Features</Text>
          <View style={styles.featuresGrid}>
            {features.map((feature, index) => (
              <TouchableOpacity
                key={index}
                onPress={feature.action}
                style={styles.featureCard}
              >
                <View style={styles.featureIcon}>
                  <Ionicons name={feature.icon as any} size={20} color="#3b82f6" />
                </View>
                <Text style={styles.featureTitle}>{feature.title}</Text>
                <Text style={styles.featureDescription}>{feature.description}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Recent Activity */}
        <View style={styles.activitySection}>
          <Text style={styles.sectionTitle}>Recent Activity</Text>
          <View style={styles.activityCard}>
            {isLoading ? (
              <View style={styles.activityItem}>
                <View style={styles.activityIconBlue}>
                  <Ionicons name="refresh" size={16} color="#3b82f6" />
                </View>
                <View style={styles.activityContent}>
                  <Text style={styles.activityTitle}>Loading...</Text>
                  <Text style={styles.activitySubtitle}>Fetching recent activity</Text>
                </View>
                <Text style={styles.activityTime}>--</Text>
              </View>
            ) : recentAnalyses.length > 0 ? (
              recentAnalyses.map((analysis, index) => (
                <View key={analysis.id} style={styles.activityItem}>
                  <View style={styles.activityIconSuccess}>
                    <Ionicons name="checkmark" size={16} color="#10b981" />
                  </View>
                  <View style={styles.activityContent}>
                    <Text style={styles.activityTitle}>Analysis Complete</Text>
                    <Text style={styles.activitySubtitle}>
                      {analysis.result.objects?.length > 0 
                        ? `${analysis.result.objects.length} objects detected`
                        : analysis.result.colors?.length > 0 
                          ? `${analysis.result.colors.length} colors found`
                          : analysis.result.prediction
                            ? `Deepfake: ${analysis.result.prediction} (${Math.round((analysis.result.fake_confidence || 0) * 100)}%)`
                            : 'Analysis completed'
                      }
                    </Text>
                  </View>
                  <Text style={styles.activityTime}>{formatTimeAgo(analysis.createdAt)}</Text>
                </View>
              ))
            ) : (
              <View style={styles.activityItem}>
                <View style={styles.activityIconBlue}>
                  <Ionicons name="camera" size={16} color="#3b82f6" />
                </View>
                <View style={styles.activityContent}>
                  <Text style={styles.activityTitle}>Start Analyzing</Text>
                  <Text style={styles.activitySubtitle}>Capture your first image to begin</Text>
                </View>
                <Text style={styles.activityTime}>--</Text>
              </View>
            )}
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8fafc',
  },
  header: {
    paddingHorizontal: 24,
    paddingVertical: 32,
    borderBottomLeftRadius: 24,
    borderBottomRightRadius: 24,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerTitle: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
  },
  headerSubtitle: {
    color: '#bfdbfe',
    fontSize: 16,
  },
  profileButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    padding: 12,
    borderRadius: 24,
  },
  content: {
    flex: 1,
    paddingHorizontal: 24,
  },
  statsSection: {
    marginTop: 24,
    marginBottom: 24,
  },
  sectionTitle: {
    color: '#374151',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 16,
    flex: 1,
    marginHorizontal: 6,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  statValue: {
    color: '#3b82f6',
    fontSize: 24,
    fontWeight: 'bold',
  },
  statValueGreen: {
    color: '#10b981',
    fontSize: 24,
    fontWeight: 'bold',
  },
  statLabel: {
    color: '#6b7280',
    fontSize: 14,
  },
  featuresSection: {
    marginBottom: 24,
  },
  featuresGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  featureCard: {
    backgroundColor: 'white',
    padding: 24,
    borderRadius: 16,
    width: '47%',
    marginBottom: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  featureIcon: {
    backgroundColor: '#eff6ff',
    padding: 12,
    borderRadius: 12,
    width: 48,
    height: 48,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  featureTitle: {
    color: '#374151',
    fontWeight: '600',
    fontSize: 16,
    marginBottom: 8,
  },
  featureDescription: {
    color: '#6b7280',
    fontSize: 14,
  },
  activitySection: {
    marginBottom: 32,
  },
  activityCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    padding: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  activityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  activityIconSuccess: {
    backgroundColor: '#dcfce7',
    padding: 8,
    borderRadius: 8,
    marginRight: 12,
  },
  activityIconBlue: {
    backgroundColor: '#dbeafe',
    padding: 8,
    borderRadius: 8,
    marginRight: 12,
  },
  activityContent: {
    flex: 1,
  },
  activityTitle: {
    color: '#374151',
    fontWeight: '500',
  },
  activitySubtitle: {
    color: '#6b7280',
    fontSize: 14,
  },
  activityTime: {
    color: '#9ca3af',
    fontSize: 14,
  },
});
