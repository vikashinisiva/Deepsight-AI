import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import { StatusBar } from 'expo-status-bar';
import { LinearGradient } from 'expo-linear-gradient';
import { router } from 'expo-router';

export default function ProfileScreen() {
  const userStats = {
    totalAnalyses: 124,
    successRate: 96,
    favoriteCategory: 'Deep Fakes',
    memberSince: 'September 2025',
  };

  const settingsOptions = [
    {
      icon: 'camera',
      title: 'Camera Settings',
      subtitle: 'Configure capture preferences',
      onPress: () => Alert.alert('Camera Settings', 'Camera settings would open here'),
    },
    {
      icon: 'analytics',
      title: 'Analysis Preferences',
      subtitle: 'Customize AI analysis options',
      onPress: () => Alert.alert('Analysis Preferences', 'Analysis preferences would open here'),
    },
    {
      icon: 'notifications',
      title: 'Notifications',
      subtitle: 'Manage notification settings',
      onPress: () => Alert.alert('Notifications', 'Notification settings would open here'),
    },
    {
      icon: 'cloud',
      title: 'Data & Storage',
      subtitle: 'Manage your data and storage',
      onPress: () => Alert.alert('Data & Storage', 'Data settings would open here'),
    },
    {
      icon: 'shield-checkmark',
      title: 'Privacy & Security',
      subtitle: 'Privacy and security settings',
      onPress: () => Alert.alert('Privacy & Security', 'Privacy settings would open here'),
    },
    {
      icon: 'help-circle',
      title: 'Help & Support',
      subtitle: 'Get help and contact support',
      onPress: () => Alert.alert('Help & Support', 'Help center would open here'),
    },
  ];

  const accountOptions = [
    {
      icon: 'person-circle',
      title: 'Edit Profile',
      subtitle: 'Update your profile information',
      onPress: () => Alert.alert('Edit Profile', 'Profile editing would open here'),
    },
    {
      icon: 'card',
      title: 'Subscription',
      subtitle: 'Manage your subscription',
      onPress: () => Alert.alert('Subscription', 'Subscription management would open here'),
    },
    {
      icon: 'download',
      title: 'Export Data',
      subtitle: 'Download your analysis data',
      onPress: () => Alert.alert('Export Data', 'Data export would start here'),
    },
    {
      icon: 'log-out',
      title: 'Sign Out',
      subtitle: 'Sign out of your account',
      onPress: () => Alert.alert('Sign Out', 'Are you sure you want to sign out?', [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Sign Out', style: 'destructive' },
      ]),
      isDestructive: true,
    },
  ];

  const StatCard = ({ title, value, subtitle }: { title: string; value: string | number; subtitle?: string }) => (
    <View style={styles.statCard}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statTitle}>{title}</Text>
      {subtitle && <Text style={styles.statSubtitle}>{subtitle}</Text>}
    </View>
  );

  const OptionRow = ({ 
    icon, 
    title, 
    subtitle, 
    onPress, 
    isDestructive = false 
  }: { 
    icon: string; 
    title: string; 
    subtitle: string; 
    onPress: () => void;
    isDestructive?: boolean;
  }) => (
    <TouchableOpacity style={styles.optionRow} onPress={onPress}>
      <View style={[styles.optionIcon, isDestructive && styles.optionIconDestructive]}>
        <Ionicons 
          name={icon as any} 
          size={20} 
          color={isDestructive ? '#ef4444' : '#3b82f6'} 
        />
      </View>
      <View style={styles.optionContent}>
        <Text style={[styles.optionTitle, isDestructive && styles.optionTitleDestructive]}>
          {title}
        </Text>
        <Text style={styles.optionSubtitle}>{subtitle}</Text>
      </View>
      <Ionicons name="chevron-forward" size={20} color="#94a3b8" />
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />
      
      {/* Header */}
      <LinearGradient
        colors={['#1e3a8a', '#3b82f6']}
        style={styles.header}
      >
        <View style={styles.headerContent}>
          <View style={styles.profileSection}>
            <View style={styles.avatarContainer}>
              <Ionicons name="person" size={40} color="white" />
            </View>
            <View style={styles.profileInfo}>
              <Text style={styles.profileName}>Team 227</Text>
              <Text style={styles.profileEmail}>vishal and vikashini</Text>
              <Text style={styles.memberSince}>Member since {userStats.memberSince}</Text>
            </View>
          </View>
        </View>
      </LinearGradient>

      <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
        {/* Stats Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Your Statistics</Text>
          <View style={styles.statsGrid}>
            <StatCard title="Total Analyses" value={userStats.totalAnalyses} />
            <StatCard title="Success Rate" value={`${userStats.successRate}%`} />
            <StatCard title="Favorite" value={userStats.favoriteCategory} subtitle="Category" />
            <StatCard title="This Month" value="23" subtitle="Analyses" />
          </View>
        </View>

        {/* Settings Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Settings</Text>
          <View style={styles.optionsContainer}>
            {settingsOptions.map((option, index) => (
              <OptionRow key={index} {...option} />
            ))}
          </View>
        </View>

        {/* Account Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account</Text>
          <View style={styles.optionsContainer}>
            {accountOptions.map((option, index) => (
              <OptionRow key={index} {...option} />
            ))}
          </View>
        </View>

        {/* App Info */}
        <View style={styles.appInfo}>
          <Text style={styles.appInfoText}>DeepSight AI v1.0.0</Text>
          <Text style={styles.appInfoText}>© 2024 DeepSight Technologies</Text>
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
    paddingBottom: 32,
  },
  headerContent: {
    paddingHorizontal: 24,
    paddingTop: 16,
  },
  profileSection: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatarContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  profileInfo: {
    flex: 1,
  },
  profileName: {
    color: 'white',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  profileEmail: {
    color: '#bfdbfe',
    fontSize: 16,
    marginBottom: 4,
  },
  memberSince: {
    color: '#93c5fd',
    fontSize: 14,
  },
  content: {
    flex: 1,
    paddingHorizontal: 24,
  },
  section: {
    marginTop: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1e293b',
    marginBottom: 16,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  statCard: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    minWidth: '47%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#3b82f6',
    marginBottom: 4,
  },
  statTitle: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
    textAlign: 'center',
  },
  statSubtitle: {
    fontSize: 12,
    color: '#6b7280',
    marginTop: 2,
  },
  optionsContainer: {
    backgroundColor: 'white',
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  optionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#f1f5f9',
  },
  optionIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#eff6ff',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  optionIconDestructive: {
    backgroundColor: '#fef2f2',
  },
  optionContent: {
    flex: 1,
  },
  optionTitle: {
    fontSize: 16,
    fontWeight: '500',
    color: '#1e293b',
    marginBottom: 2,
  },
  optionTitleDestructive: {
    color: '#ef4444',
  },
  optionSubtitle: {
    fontSize: 14,
    color: '#64748b',
  },
  appInfo: {
    alignItems: 'center',
    paddingVertical: 32,
  },
  appInfoText: {
    fontSize: 12,
    color: '#94a3b8',
    marginBottom: 4,
  },
});
