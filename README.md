# Physio Aware Deepfake Detection

## Overview
This project investigates a physiological signal–aware approach to deepfake and video-based spoof detection for secure e-KYC and media forensics applications. Unlike conventional methods that rely solely on spatial artifacts, this system leverages **remote photoplethysmography (rPPG)** to capture subtle blood flow patterns from facial videos and fuse them with spatio-temporal visual features.

## Motivation
Video-based identity verification systems are increasingly vulnerable to AI-generated deepfake attacks. While deepfake models can synthesize highly realistic facial textures, they often fail to reproduce **temporally consistent physiological signals** such as heartbeat-induced skin color variations. This project explores whether these signals can act as reliable liveness cues.

## Methodology
The proposed pipeline consists of:
1. Face detection and motion stabilization
2. Region-of-interest (ROI) extraction (forehead and cheeks)
3. rPPG signal recovery using green-channel analysis and frequency-domain processing
4. Spatio-temporal feature fusion using deep learning models
5. Live vs. spoof classification

## Datasets
- Celeb-DF
- DeepFake-TIMIT
- Self-recorded real videos (controlled lighting)

> Note: Datasets are not included in this repository.

## Applications
- Secure Video KYC (Banking & FinTech)
- Deepfake Media Forensics
- Cognitive Security and Disinformation Detection

