import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image, StyleSheet, RefreshControl } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { LinearGradient } from 'expo-linear-gradient';
import { storageService, StoredAnalysis } from '@/services/storageService';
import { useFocusEffect } from '@react-navigation/native';

export default function ResultsScreen() {
  const [selectedFilter, setSelectedFilter] = useState<'all' | 'completed' | 'processing'>('all');
  const [analyses, setAnalyses] = useState<StoredAnalysis[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Load analyses when screen comes into focus
  useFocusEffect(
    React.useCallback(() => {
      loadAnalyses();
    }, [])
  );

  const loadAnalyses = async () => {
    try {
      setIsLoading(true);
      const storedAnalyses = await storageService.getAnalyses();
      setAnalyses(storedAnalyses);
    } catch (error) {
      console.error('Failed to load analyses:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadAnalyses();
    setRefreshing(false);
  };

  const deleteAnalysis = async (id: string) => {
    try {
      await storageService.deleteAnalysis(id);
      await loadAnalyses(); // Reload the list
    } catch (error) {
      console.error('Failed to delete analysis:', error);
    }
  };

  const toggleFavorite = async (id: string) => {
    try {
      await storageService.toggleFavorite(id);
      await loadAnalyses(); // Reload the list
    } catch (error) {
      console.error('Failed to toggle favorite:', error);
    }
  };

  // Filter analyses based on selected filter
  const filteredResults = analyses.filter(analysis => {
    if (selectedFilter === 'all') return true;
    if (selectedFilter === 'completed') return analysis.result.objects.length > 0 || analysis.result.colors.length > 0;
    // For now, we don't have processing status, so return empty for processing
    return false;
  });

  const formatTimestamp = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return date.toLocaleDateString();
  };

  const FilterButton = ({ label, value, isSelected }: { 
    label: string; 
    value: typeof selectedFilter; 
    isSelected: boolean 
  }) => (
    <TouchableOpacity
      style={[styles.filterButton, isSelected && styles.filterButtonActive]}
      onPress={() => setSelectedFilter(value)}
    >
      <Text style={[styles.filterButtonText, isSelected && styles.filterButtonTextActive]}>
        {label}
      </Text>
    </TouchableOpacity>
  );

  const ResultCard = ({ result }: { result: StoredAnalysis }) => (
    <TouchableOpacity style={styles.resultCard}>
      <Image source={{ uri: result.imageUri }} style={styles.resultImage} />
      
      <View style={styles.resultContent}>
        <View style={styles.resultHeader}>
          <Text style={styles.resultTimestamp}>{formatTimestamp(new Date(result.createdAt))}</Text>
          <View style={styles.headerActions}>
            <TouchableOpacity 
              onPress={() => toggleFavorite(result.id)}
              style={styles.actionButton}
            >
              <Ionicons 
                name={result.isFavorite ? "star" : "star-outline"} 
                size={16} 
                color={result.isFavorite ? "#fbbf24" : "#9ca3af"} 
              />
            </TouchableOpacity>
            <TouchableOpacity 
              onPress={() => deleteAnalysis(result.id)}
              style={styles.actionButton}
            >
              <Ionicons name="trash-outline" size={16} color="#ef4444" />
            </TouchableOpacity>
          </View>
        </View>

        {/* Objects detected */}
        {result.result.objects.length > 0 && (
          <View style={styles.sectionContainer}>
            <Text style={styles.sectionLabel}>Objects detected ({result.result.objects.length}):</Text>
            <View style={styles.tagsList}>
              {result.result.objects.slice(0, 3).map((obj: any, index: number) => (
                <View key={index} style={styles.tag}>
                  <Text style={styles.tagText}>
                    {obj.label} ({Math.round(obj.confidence * 100)}%)
                  </Text>
                </View>
              ))}
              {result.result.objects.length > 3 && (
                <View style={styles.tag}>
                  <Text style={styles.tagText}>
                    +{result.result.objects.length - 3} more
                  </Text>
                </View>
              )}
            </View>
          </View>
        )}

        {/* Colors detected */}
        {result.result.colors.length > 0 && (
          <View style={styles.sectionContainer}>
            <Text style={styles.sectionLabel}>Colors ({result.result.colors.length}):</Text>
            <View style={styles.colorsList}>
              {result.result.colors.slice(0, 5).map((color: any, index: number) => (
                <View key={index} style={styles.colorItem}>
                  <View 
                    style={[styles.colorSwatch, { backgroundColor: color.hex }]}
                  />
                  <Text style={styles.colorName}>{color.name}</Text>
                </View>
              ))}
            </View>
          </View>
        )}

        {/* Scene classification */}
        {result.result.scenes && result.result.scenes.length > 0 && (
          <View style={styles.sectionContainer}>
            <Text style={styles.sectionLabel}>Scene:</Text>
            <Text style={styles.contentText}>
              {result.result.scenes[0].name} ({Math.round(result.result.scenes[0].confidence * 100)}%)
            </Text>
          </View>
        )}

        {/* Image caption */}
        {result.result.caption && (
          <View style={styles.sectionContainer}>
            <Text style={styles.sectionLabel}>Description:</Text>
            <Text style={styles.captionText}>"{result.result.caption}"</Text>
          </View>
        )}

        {/* Text recognition */}
        {result.result.text && result.result.text.length > 0 && (
          <View style={styles.sectionContainer}>
            <Text style={styles.sectionLabel}>Text found:</Text>
            <Text style={styles.contentText}>{result.result.text.join(', ')}</Text>
          </View>
        )}
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="dark" />
      
      {/* Header */}
      <LinearGradient
        colors={['#1e3a8a', '#3b82f6']}
        style={styles.header}
      >
        <Text style={styles.headerTitle}>Analysis Results</Text>
        <Text style={styles.headerSubtitle}>Your AI analysis history</Text>
      </LinearGradient>

      {/* Filter Buttons */}
      <View style={styles.filterContainer}>
        <FilterButton label="All" value="all" isSelected={selectedFilter === 'all'} />
        <FilterButton label="Completed" value="completed" isSelected={selectedFilter === 'completed'} />
        <FilterButton label="Processing" value="processing" isSelected={selectedFilter === 'processing'} />
      </View>

      {/* Results List */}
      <ScrollView 
        style={styles.resultsContainer} 
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            colors={['#3b82f6']}
            tintColor="#3b82f6"
          />
        }
      >
        {filteredResults.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons name="analytics-outline" size={64} color="#94a3b8" />
            <Text style={styles.emptyStateTitle}>No results found</Text>
            <Text style={styles.emptyStateText}>
              {selectedFilter === 'all' 
                ? 'Start analyzing images to see results here'
                : `No ${selectedFilter} analyses found`
              }
            </Text>
          </View>
        ) : (
          filteredResults.map(result => (
            <ResultCard key={result.id} result={result} />
          ))
        )}
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
  headerTitle: {
    color: 'white',
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  headerSubtitle: {
    color: '#bfdbfe',
    fontSize: 16,
  },
  filterContainer: {
    flexDirection: 'row',
    paddingHorizontal: 24,
    paddingVertical: 16,
    gap: 12,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#e2e8f0',
  },
  filterButtonActive: {
    backgroundColor: '#3b82f6',
  },
  filterButtonText: {
    color: '#64748b',
    fontSize: 14,
    fontWeight: '500',
  },
  filterButtonTextActive: {
    color: 'white',
  },
  resultsContainer: {
    flex: 1,
    paddingHorizontal: 24,
  },
  resultCard: {
    backgroundColor: 'white',
    borderRadius: 16,
    marginBottom: 16,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  resultImage: {
    width: '100%',
    height: 120,
    backgroundColor: '#e2e8f0',
  },
  resultContent: {
    padding: 16,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  resultTimestamp: {
    color: '#64748b',
    fontSize: 14,
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  statusCompleted: {
    backgroundColor: '#dcfce7',
  },
  statusProcessing: {
    backgroundColor: '#dbeafe',
  },
  statusFailed: {
    backgroundColor: '#fee2e2',
  },
  statusText: {
    fontSize: 12,
    fontWeight: '500',
  },
  statusTextCompleted: {
    color: '#16a34a',
  },
  statusTextProcessing: {
    color: '#2563eb',
  },
  statusTextFailed: {
    color: '#dc2626',
  },
  confidenceText: {
    color: '#374151',
    fontSize: 14,
    fontWeight: '500',
    marginBottom: 8,
  },
  objectsContainer: {
    marginBottom: 12,
  },
  objectsLabel: {
    color: '#6b7280',
    fontSize: 12,
    fontWeight: '500',
    marginBottom: 4,
  },
  objectsList: {
    color: '#374151',
    fontSize: 14,
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  tag: {
    backgroundColor: '#f1f5f9',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  tagText: {
    color: '#475569',
    fontSize: 12,
  },
  processingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  processingText: {
    color: '#3b82f6',
    fontSize: 14,
    fontStyle: 'italic',
  },
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 64,
  },
  emptyStateTitle: {
    color: '#374151',
    fontSize: 18,
    fontWeight: '600',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyStateText: {
    color: '#6b7280',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  actionButton: {
    padding: 4,
    borderRadius: 4,
    backgroundColor: '#f8fafc',
  },
  sectionContainer: {
    marginBottom: 12,
  },
  sectionLabel: {
    color: '#6b7280',
    fontSize: 12,
    fontWeight: '500',
    marginBottom: 4,
  },
  tagsList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  colorsList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  colorItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  colorSwatch: {
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#e5e7eb',
  },
  colorName: {
    color: '#374151',
    fontSize: 12,
  },
  contentText: {
    color: '#374151',
    fontSize: 14,
  },
  captionText: {
    color: '#6b7280',
    fontSize: 14,
    fontStyle: 'italic',
  },
});
