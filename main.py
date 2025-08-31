import cv2
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import io
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from skimage import color
import requests
import os
import threading
from ultralytics import YOLO
import customtkinter as ctk

# Flask imports
from flask import Flask, request, render_template, jsonify, send_file, render_template_string
import base64
import socket
import webbrowser
from werkzeug.utils import secure_filename
import tempfile
import shutil

# New imports for QR code and Ngrok
import qrcode
import subprocess
import time

# Set CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Ngrok configuration
NGROK_TOKEN = "YOUR NGROK TOKEN"

class FlameAnalyzer:
    def __init__(self):
        self.detection_model = None
        self.segmentation_model = None
        self.materials_db = None
        self.last_results = None
        self.model_loaded = False
        self.db_loaded = False
        self.current_process = "Initializing..."
        
        # Auto-load models and database on startup
        self.load_models()
        self.load_materials_db()
        
    def load_models(self):
        """Load custom YOLO models"""
        try:
            if os.path.exists('OBJ_best.pt'):
                self.detection_model = YOLO('OBJ_best.pt')
                print("Loaded custom detection model: OBJ_best.pt")
                self.model_loaded = True
            else:
                print("Error: OBJ_best.pt not found")
                return False
            
            if os.path.exists('SEG_best.pt'):
                self.segmentation_model = YOLO('SEG_best.pt')
                print("Loaded custom segmentation model: SEG_best.pt")
            else:
                print("Warning: SEG_best.pt not found - will use bounding box masks")
                
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            self.model_loaded = False
            return False
    
    def load_materials_db(self, json_path="flame_dataset.json"):
        """Load materials database from JSON file"""
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    self.materials_db = json.load(f)
                print(f"Loaded materials database: {json_path}")
                self.db_loaded = True
                return True
            except Exception as e:
                print(f"Error loading materials DB: {e}")
                return False
        else:
            print(f"Materials database not found: {json_path}")
            return False
    
    def detect_fire(self, image_bgr):
        """Detect fire regions using custom YOLO model - keep only highest confidence detection"""
        self.current_process = "Detecting fire regions..."
        if self.detection_model is None:
            return None, []
        
        try:
            results = self.detection_model(image_bgr, imgsz=640, conf=0.4, verbose=False)
            all_boxes = []
            annotated_image = image_bgr.copy()
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if cls == 0 and conf > 0.4:
                            all_boxes.append([x1, y1, x2, y2, conf])
            
            # Keep only the detection with highest confidence
            if all_boxes:
                best_box = max(all_boxes, key=lambda x: x[4])
                x1, y1, x2, y2, conf = best_box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, f'Fire: {conf:.3f}', 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                return annotated_image, [best_box]
            else:
                return annotated_image, []
                
        except Exception as e:
            print(f"Error in fire detection: {e}")
            return image_bgr, []
    
    def segment_fire(self, image_bgr, boxes):
        """Segment fire regions using custom YOLO segmentation model - keep highest confidence"""
        self.current_process = "Segmenting flame regions..."
        if self.segmentation_model is None:
            return self._create_bbox_mask(image_bgr, boxes)
        
        try:
            results = self.segmentation_model(image_bgr, imgsz=640, conf=0.4, verbose=False)
            mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            
            best_mask = None
            best_conf = 0
            
            for result in results:
                if result.masks is not None and len(result.masks) > 0:
                    masks = result.masks.data.cpu().numpy()  # [N, H, W]
                    
                    # Get corresponding confidences if available
                    if result.boxes is not None and len(result.boxes) > 0:
                        confidences = result.boxes.conf.cpu().numpy()
                    else:
                        confidences = [0.5] * len(masks)  # Default confidence
                    
                    # Find mask with highest confidence
                    for i, seg_mask in enumerate(masks):
                        conf = confidences[i] if i < len(confidences) else 0.5
                        if conf > best_conf:
                            best_conf = conf
                            best_mask = seg_mask
            
            # Process the best mask only
            if best_mask is not None:
                binary_mask = (best_mask > 0.5).astype(np.uint8)
                mask_resized = cv2.resize(binary_mask, 
                                        (image_bgr.shape[1], image_bgr.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                mask = mask_resized * 255
            
            # If no segmentation found, fallback to bbox
            if np.sum(mask) == 0 and boxes:
                mask = self._create_bbox_mask(image_bgr, boxes)
            
            return mask
            
        except Exception as e:
            print(f"Error in fire segmentation: {e}")
            return self._create_bbox_mask(image_bgr, boxes)
    
    def _create_bbox_mask(self, image, boxes):
        """Create mask from bounding boxes"""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            mask[y1:y2, x1:x2] = 255
        return mask
    
    def extract_flame_pixels(self, image, mask):
        """Extract flame pixels using the segmentation mask"""
        self.current_process = "Extracting flame pixels..."
        if mask is None or np.sum(mask) == 0:
            return np.array([]).reshape(0, 3)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        flame_pixels = image_rgb[mask > 0]
        return flame_pixels
    
    def optimal_k_selection(self, data, max_k=6):
        """Find optimal k using simplified elbow method"""
        if len(data) < 4:
            return 2
        
        # Limit computation for speed
        max_k = min(max_k, len(data) // 10, 6)  # More conservative limits
        
        try:
            # Quick silhouette-based selection
            best_k = 2
            best_score = -1
            
            for k in range(2, max_k + 1):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)  # Reduced iterations
                    labels = kmeans.fit_predict(data)
                    
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(data, labels, sample_size=min(300, len(data)))  # Sample for speed
                        if score > best_score:
                            best_score = score
                            best_k = k
                except:
                    continue
            
            return min(best_k, 4)  # Cap at 4 for performance
            
        except Exception as e:
            print(f"Optimal k selection failed: {e}, using default k=3")
            return 3
    
    def apply_clustering_techniques(self, flame_lab):
        """Apply core clustering techniques to flame pixels"""
        if len(flame_lab) < 10:
            return {'K-Means++': np.mean(flame_lab, axis=0)}  # Simple fallback
        
        # Subsample for performance if dataset is large
        if len(flame_lab) > 1000:
            indices = np.random.choice(len(flame_lab), 1000, replace=False)
            flame_sample = flame_lab[indices]
        else:
            flame_sample = flame_lab
            
        scaler = StandardScaler()
        flame_scaled = scaler.fit_transform(flame_sample)
        
        clustering_results = {}
        
        # Determine optimal k (with timeout protection)
        self.current_process = "Determining optimal clusters..."
        optimal_k = self.optimal_k_selection(flame_scaled)
        
        try:
            # 1. Variational Bayesian GMM
            self.current_process = "Applying VB-GMM..."
            vbgmm = BayesianGaussianMixture(n_components=optimal_k, random_state=42, max_iter=50)
            vbgmm.fit(flame_scaled)
            vb_centers = scaler.inverse_transform(vbgmm.means_)
            vb_weighted = np.average(vb_centers, axis=0, weights=vbgmm.weights_)
            clustering_results['VB-GMM'] = vb_weighted
            
            # 2. Standard GMM
            self.current_process = "Applying GMM..."
            gmm = GaussianMixture(n_components=optimal_k, random_state=42, max_iter=50)
            gmm.fit(flame_scaled)
            gmm_centers = scaler.inverse_transform(gmm.means_)
            gmm_weighted = np.average(gmm_centers, axis=0, weights=gmm.weights_)
            clustering_results['GMM'] = gmm_weighted
            
            # 3. K-Means++
            self.current_process = "Applying K-Means++..."
            kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=3, max_iter=100)
            kmeans.fit(flame_scaled)
            kmeans_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            kmeans_weighted = np.mean(kmeans_centers, axis=0)
            clustering_results['K-Means++'] = kmeans_weighted
            
            # 4. DBSCAN
            self.current_process = "Applying DBSCAN..."
            dbscan = DBSCAN(eps=0.5, min_samples=max(3, len(flame_sample) // 50))
            dbscan_labels = dbscan.fit_predict(flame_scaled)
            unique_labels = np.unique(dbscan_labels[dbscan_labels >= 0])
            if len(unique_labels) > 0:
                dbscan_centers = []
                for label in unique_labels:
                    cluster_points = flame_sample[dbscan_labels == label]
                    if len(cluster_points) > 0:
                        dbscan_centers.append(np.mean(cluster_points, axis=0))
                if dbscan_centers:
                    dbscan_weighted = np.mean(dbscan_centers, axis=0)
                    clustering_results['DBSCAN'] = dbscan_weighted
                else:
                    clustering_results['DBSCAN'] = np.mean(flame_sample, axis=0)
            else:
                clustering_results['DBSCAN'] = np.mean(flame_sample, axis=0)
            
            # 5. Mean-Shift
            self.current_process = "Applying Mean-Shift..."
            try:
                meanshift = MeanShift(bandwidth=0.8, max_iter=50)
                meanshift.fit(flame_scaled)
                ms_centers = scaler.inverse_transform(meanshift.cluster_centers_)
                ms_weighted = np.mean(ms_centers, axis=0)
                clustering_results['Mean-Shift'] = ms_weighted
            except:
                clustering_results['Mean-Shift'] = np.mean(flame_sample, axis=0)
            
            # 6. Agglomerative Clustering
            self.current_process = "Applying Agglomerative..."
            try:
                agg = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
                agg_labels = agg.fit_predict(flame_scaled)
                agg_centers = []
                for i in range(optimal_k):
                    cluster_points = flame_sample[agg_labels == i]
                    if len(cluster_points) > 0:
                        agg_centers.append(np.mean(cluster_points, axis=0))
                if agg_centers:
                    agg_weighted = np.mean(agg_centers, axis=0)
                    clustering_results['Agglomerative'] = agg_weighted
                else:
                    clustering_results['Agglomerative'] = np.mean(flame_sample, axis=0)
            except:
                clustering_results['Agglomerative'] = np.mean(flame_sample, axis=0)
            
        except Exception as e:
            print(f"Error in clustering: {e}")
            # Ensure at least one result
            if not clustering_results:
                clustering_results['K-Means++'] = np.mean(flame_lab, axis=0)
        
        return clustering_results
    
    def analyze_colors(self, flame_pixels):
        """Enhanced color analysis with multiple clustering techniques"""
        if len(flame_pixels) == 0:
            return None
        
        self.current_process = "Converting to LAB color space..."
        flame_rgb_norm = flame_pixels.astype(np.float32) / 255.0
        flame_lab = color.rgb2lab(flame_rgb_norm.reshape(-1, 1, 3)).reshape(-1, 3)
        
        # Simulate vertical regions
        n_pixels = len(flame_lab)
        base_end = int(n_pixels * 0.5)
        mid_end = int(n_pixels * 0.8)
        
        regions = {
            'base': flame_lab[:base_end],
            'middle': flame_lab[base_end:mid_end],
            'tip': flame_lab[mid_end:]
        }
        
        weights = {'base': 0.5, 'middle': 0.3, 'tip': 0.2}
        region_results = {}
        
        # Analyze each region with all clustering techniques
        for region_name, pixels in regions.items():
            if len(pixels) > 10:
                self.current_process = f"Analyzing {region_name} region..."
                clustering_results = self.apply_clustering_techniques(pixels)
                region_results[region_name] = {
                    'clustering_results': clustering_results,
                    'region_weight': weights[region_name]
                }
        
        # Combine all results
        final_results = {}
        individual_results = {}  # NEW: Store individual method results
        
        if region_results:
            methods = ['VB-GMM', 'GMM', 'K-Means++', 'DBSCAN', 'Mean-Shift', 'Agglomerative']
            
            for method in methods:
                weighted_lab = np.zeros(3)
                total_weight = 0
                
                for region_name, region_data in region_results.items():
                    if method in region_data['clustering_results']:
                        weight = region_data['region_weight']
                        weighted_lab += region_data['clustering_results'][method] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    weighted_lab /= total_weight
                    try:
                        rgb_norm = color.lab2rgb(weighted_lab.reshape(1, 1, 3)).reshape(3)
                        rgb = np.clip((rgb_norm * 255), 0, 255).astype(int)
                        
                        final_results[method] = {
                            'lab': weighted_lab,
                            'rgb': rgb
                        }
                        
                        # Store individual result for separate analysis
                        individual_results[method] = {
                            'lab': weighted_lab,
                            'rgb': rgb,
                            'method_name': method
                        }
                        
                    except:
                        # Fallback if color conversion fails
                        final_results[method] = {
                            'lab': weighted_lab,
                            'rgb': np.array([128, 128, 128])  # Gray fallback
                        }
                        individual_results[method] = {
                            'lab': weighted_lab,
                            'rgb': np.array([128, 128, 128]),
                            'method_name': method
                        }
        
        return {
            'region_results': region_results,
            'final_results': final_results,
            'individual_results': individual_results,  # NEW
            'raw_lab': flame_lab
        }
    
    def query_llama_individual(self, method_name, color_data):
        """Query LLaMA for individual clustering method"""
        if not self.materials_db:
            return {"reasoning": "No materials database available.", "top_materials": []}
        
        try:
            prompt = f"""
You are a flame analysis expert. Analyze the flame color data from the {method_name} clustering method and identify the most likely burning material.

FLAME COLOR DATA (Method: {method_name}):
RGB: {color_data['rgb'].tolist()}
LAB: {color_data['lab'].tolist()}

MATERIALS DATABASE:
{json.dumps(self.materials_db, indent=2)}

TASK: Compare the {method_name} clustering result with the materials database and provide:
1. Top 3 most likely materials with similarity percentages
2. Recommended extinguishers for each material
3. Brief scientific reasoning specific to {method_name} method characteristics
4. Confidence level for this specific clustering approach

REQUIRED OUTPUT FORMAT:
{{
    "method_name": "{method_name}",
    "top_materials": [
        {{"name": "Material1", "similarity": 95.5, "extinguishers": ["CO2", "Foam"]}},
        {{"name": "Material2", "similarity": 78.2, "extinguishers": ["Water"]}},
        {{"name": "Material3", "similarity": 65.1, "extinguishers": ["Dry Powder"]}}
    ],
    "reasoning": "Scientific explanation specific to {method_name} clustering...",
    "confidence": 85.0
}}

Respond with valid JSON only.
"""
            
            endpoints = [
                "http://localhost:11434/api/generate",
                "http://localhost:8080/completion",
                "http://127.0.0.1:5000/v1/completions"
            ]
            
            for endpoint in endpoints:
                try:
                    if "ollama" in endpoint:
                        response = requests.post(endpoint, json={
                            "model": "llama3.2:8b",
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 1000}
                        }, timeout=30)
                    else:
                        response = requests.post(endpoint, json={
                            "prompt": prompt,
                            "max_tokens": 1000,
                            "temperature": 0.3
                        }, timeout=15)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if "ollama" in endpoint:
                            llama_output = result.get("response", "")
                        else:
                            llama_output = result.get("choices", [{}])[0].get("text", "")
                        
                        try:
                            start_idx = llama_output.find('{')
                            end_idx = llama_output.rfind('}') + 1
                            if start_idx >= 0 and end_idx > start_idx:
                                json_str = llama_output[start_idx:end_idx]
                                parsed_json = json.loads(json_str)
                                return parsed_json
                        except Exception:
                            pass
                        
                        return {"method_name": method_name, "reasoning": llama_output, "top_materials": [], "confidence": 50.0}
                
                except requests.exceptions.RequestException:
                    continue
            
            return self._individual_fallback_analysis(method_name, color_data)
            
        except Exception as e:
            print(f"Error querying LLaMA for {method_name}: {e}")
            return self._individual_fallback_analysis(method_name, color_data)
    
    def _individual_fallback_analysis(self, method_name, color_data):
        """Fallback analysis for individual clustering method"""
        if not self.materials_db:
            return {"method_name": method_name, "reasoning": "No materials database available.", "top_materials": [], "confidence": 0.0}
        
        matches = []
        for material in self.materials_db.get('materials', []):
            if 'flame_lab' in material:
                material_lab = np.array(material['flame_lab'])
                distance = np.linalg.norm(color_data['lab'] - material_lab)
                similarity = max(0, 100 * (1 - distance / 200))
                matches.append({
                    'name': material['name'],
                    'similarity': similarity,
                    'extinguishers': material.get('possible_extinguishers', ['Unknown'])
                })
        
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        if matches:
            top_match = matches[0]
            reasoning = f"Using {method_name} clustering: Closest match to {top_match['name']} with {top_match['similarity']:.1f}% similarity."
            confidence = min(top_match['similarity'], 95.0)
        else:
            reasoning = f"{method_name} clustering completed but no material matches found in database."
            confidence = 0.0
        
        return {
            'method_name': method_name,
            'top_materials': matches[:3],
            'reasoning': reasoning,
            'confidence': confidence
        }
    
    def query_llama(self, color_features):
        """Query local LLaMA model with all clustering results AND individual results"""
        if not color_features or not self.materials_db:
            return {"individual_results": {}, "average_result": {"reasoning": "No color features or materials database available.", "top_materials": []}}
        
        self.current_process = "Querying LLaMA for material identification..."
        
        try:
            final_results = color_features['final_results']
            individual_results = color_features['individual_results']
            
            # Get individual analyses for each clustering method
            individual_analyses = {}
            for method_name, color_data in individual_results.items():
                self.current_process = f"Analyzing {method_name} individually..."
                individual_analyses[method_name] = self.query_llama_individual(method_name, color_data)
            
            # Prepare comprehensive color data for average analysis
            color_data = {}
            for method, result in final_results.items():
                color_data[method] = {
                    'RGB': result['rgb'].tolist(),
                    'LAB': result['lab'].tolist()
                }
            
            self.current_process = "Generating comprehensive analysis..."
            prompt = f"""
You are a flame analysis expert. Analyze the comprehensive flame color data from multiple clustering techniques and identify the most likely burning material.

COMPREHENSIVE FLAME COLOR DATA:
{json.dumps(color_data, indent=2)}

MATERIALS DATABASE:
{json.dumps(self.materials_db, indent=2)}

TASK: Compare all clustering results with the materials database and provide:
1. Top 3 most likely materials with similarity percentages
2. Recommended extinguishers for each material
3. Brief scientific reasoning considering all clustering methods
4. Confidence assessment based on clustering consensus

REQUIRED OUTPUT FORMAT:
{{
    "top_materials": [
        {{"name": "Material1", "similarity": 95.5, "extinguishers": ["CO2", "Foam"], "clustering_consensus": 85.0}},
        {{"name": "Material2", "similarity": 78.2, "extinguishers": ["Water"], "clustering_consensus": 72.0}},
        {{"name": "Material3", "similarity": 65.1, "extinguishers": ["Dry Powder"], "clustering_consensus": 58.0}}
    ],
    "reasoning": "Scientific explanation considering all clustering techniques...",
    "clustering_summary": "Summary of how different clustering methods contributed to the analysis..."
}}

Respond with valid JSON only.
"""
            
            endpoints = [
                "http://localhost:11434/api/generate",
                "http://localhost:8080/completion",
                "http://127.0.0.1:5000/v1/completions"
            ]
            
            average_result = None
            for endpoint in endpoints:
                try:
                    if "ollama" in endpoint:
                        response = requests.post(endpoint, json={
                            "model": "llama3.2:8b",
                            "prompt": prompt,
                            "stream": False,
                            "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 1500}
                        }, timeout=60)
                    else:
                        response = requests.post(endpoint, json={
                            "prompt": prompt,
                            "max_tokens": 1500,
                            "temperature": 0.3
                        }, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if "ollama" in endpoint:
                            llama_output = result.get("response", "")
                        else:
                            llama_output = result.get("choices", [{}])[0].get("text", "")
                        
                        try:
                            start_idx = llama_output.find('{')
                            end_idx = llama_output.rfind('}') + 1
                            if start_idx >= 0 and end_idx > start_idx:
                                json_str = llama_output[start_idx:end_idx]
                                parsed_json = json.loads(json_str)
                                average_result = parsed_json
                                break
                        except Exception:
                            pass
                        
                        average_result = {"reasoning": llama_output, "top_materials": [], "clustering_summary": "Raw LLaMA output"}
                        break
                
                except requests.exceptions.RequestException:
                    continue
            
            if average_result is None:
                average_result = self._enhanced_fallback_analysis(color_features)
            
            return {
                "individual_results": individual_analyses,
                "average_result": average_result
            }
            
        except Exception as e:
            print(f"Error querying LLaMA: {e}")
            return {
                "individual_results": {},
                "average_result": self._enhanced_fallback_analysis(color_features)
            }
    
    def _enhanced_fallback_analysis(self, color_features):
        """Enhanced fallback analysis using all clustering results"""
        if not self.materials_db:
            return {"reasoning": "No materials database available.", "top_materials": [], "clustering_summary": "Database unavailable"}
        
        final_results = color_features['final_results']
        all_matches = []
        
        for method, result in final_results.items():
            method_matches = []
            for material in self.materials_db.get('materials', []):
                if 'flame_lab' in material:
                    material_lab = np.array(material['flame_lab'])
                    distance = np.linalg.norm(result['lab'] - material_lab)
                    similarity = max(0, 100 * (1 - distance / 200))
                    method_matches.append({
                        'name': material['name'],
                        'similarity': similarity,
                        'extinguishers': material.get('possible_extinguishers', ['Unknown']),
                        'method': method
                    })
            all_matches.extend(method_matches)
        
        # Aggregate by material name
        material_scores = {}
        for match in all_matches:
            name = match['name']
            if name not in material_scores:
                material_scores[name] = {
                    'similarities': [],
                    'extinguishers': match['extinguishers'],
                    'methods': []
                }
            material_scores[name]['similarities'].append(match['similarity'])
            material_scores[name]['methods'].append(match['method'])
        
        # Calculate consensus scores
        final_matches = []
        for name, data in material_scores.items():
            avg_similarity = np.mean(data['similarities'])
            consensus = (len(data['similarities']) / len(final_results)) * 100
            final_matches.append({
                'name': name,
                'similarity': avg_similarity,
                'extinguishers': data['extinguishers'],
                'clustering_consensus': consensus
            })
        
        final_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Generate reasoning
        if final_matches:
            top_match = final_matches[0]
            reasoning = f"Multi-clustering analysis using {len(final_results)} techniques achieved {top_match['clustering_consensus']:.1f}% consensus. "
            reasoning += f"Average similarity to {top_match['name']}: {top_match['similarity']:.1f}%. "
            reasoning += f"Recommended suppression: {', '.join(top_match['extinguishers'])}."
        else:
            reasoning = "Multi-clustering analysis completed but no material matches found in database."
        
        clustering_summary = f"Applied {len(final_results)} clustering techniques: {', '.join(final_results.keys())}"
        
        return {
            'top_materials': final_matches[:3],
            'reasoning': reasoning,
            'clustering_summary': clustering_summary
        }
    
    def run_full_analysis(self, image_path):
        """Run the complete analysis pipeline with enhanced clustering"""
        try:
            self.current_process = "Loading image..."
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                return None
            
            # Detection
            detection_result, boxes = self.detect_fire(image_bgr)
            if not boxes:
                return {
                    'original_image': cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                    'detection_result': cv2.cvtColor(detection_result, cv2.COLOR_BGR2RGB),
                    'segmentation_mask': np.zeros(image_bgr.shape[:2], dtype=np.uint8),
                    'flame_pixels': np.array([]).reshape(0, 3),
                    'color_features': None,
                    'llama_results': {'individual_results': {}, 'average_result': {'reasoning': 'No fire detected.', 'top_materials': []}},
                    'boxes': [],
                    'current_process': 'Complete - No fire detected'
                }
            
            # Segmentation
            mask = self.segment_fire(image_bgr, boxes)
            
            # Extract pixels
            flame_pixels = self.extract_flame_pixels(image_bgr, mask)
            if len(flame_pixels) == 0:
                return {
                    'original_image': cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                    'detection_result': cv2.cvtColor(detection_result, cv2.COLOR_BGR2RGB),
                    'segmentation_mask': mask,
                    'flame_pixels': flame_pixels,
                    'color_features': None,
                    'llama_results': {'individual_results': {}, 'average_result': {'reasoning': 'Fire detected but no flame pixels extracted.', 'top_materials': []}},
                    'boxes': boxes,
                    'current_process': 'Complete - No flame pixels'
                }
            
            # Enhanced color analysis
            color_features = self.analyze_colors(flame_pixels)
            
            # LLaMA analysis (both individual and average)
            llama_results = self.query_llama(color_features)
            
            self.current_process = "Analysis complete!"
            
            results = {
                'original_image': cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                'detection_result': cv2.cvtColor(detection_result, cv2.COLOR_BGR2RGB),
                'segmentation_mask': mask,
                'flame_pixels': flame_pixels,
                'color_features': color_features,
                'llama_results': llama_results,
                'boxes': boxes,
                'current_process': 'Complete'
            }
            
            self.last_results = results
            return results
            
        except Exception as e:
            print(f"Error in full analysis: {e}")
            self.current_process = f"Error: {str(e)}"
            return None


class NgrokManager:
    def __init__(self):
        self.process = None
        self.public_url = None
        self.token = NGROK_TOKEN
        self.authenticated = False
        
    def authenticate(self):
        """Authenticate ngrok with token"""
        try:
            result = subprocess.run(['ngrok', 'config', 'add-authtoken', self.token], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.authenticated = True
                print("Ngrok authenticated successfully")
                return True
            else:
                print(f"Ngrok authentication failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"Error authenticating ngrok: {e}")
            return False
    
    def start_tunnel(self, port):
        """Start ngrok tunnel"""
        try:
            if not self.authenticated:
                if not self.authenticate():
                    return False, "Authentication failed"
            
            # Start ngrok tunnel
            self.process = subprocess.Popen(['ngrok', 'http', str(port)], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE)
            
            # Wait a bit for ngrok to start
            time.sleep(3)
            
            # Get public URL from ngrok API
            try:
                response = requests.get('http://localhost:4040/api/tunnels')
                if response.status_code == 200:
                    data = response.json()
                    tunnels = data.get('tunnels', [])
                    if tunnels:
                        self.public_url = tunnels[0]['public_url']
                        return True, self.public_url
                    else:
                        return False, "No tunnels found"
                else:
                    return False, "Could not get tunnel info"
            except Exception as e:
                return False, f"Error getting tunnel URL: {e}"
                
        except Exception as e:
            return False, f"Error starting tunnel: {e}"
    
    def stop_tunnel(self):
        """Stop ngrok tunnel"""
        try:
            if self.process:
                self.process.terminate()
                self.process = None
                self.public_url = None
            return True
        except Exception as e:
            print(f"Error stopping tunnel: {e}")
            return False


class FlaskWebServer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        self.server_thread = None
        self.server_port = None
        self.server_url = None
        self.temp_dir = tempfile.mkdtemp()
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string(WEB_TEMPLATE)
        
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'})
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'})
            
            if file:
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(self.temp_dir, filename)
                    file.save(filepath)
                    
                    # Run analysis
                    results = self.analyzer.run_full_analysis(filepath)
                    
                    if results is None:
                        return jsonify({'error': 'Analysis failed'})
                    
                    # Convert images to base64
                    response_data = {
                        'original_image': self.image_to_base64(results['original_image']),
                        'detection_result': self.image_to_base64(results['detection_result']),
                        'segmentation_mask': self.image_to_base64(results['segmentation_mask']),
                        'color_features': self.process_color_features(results.get('color_features')),
                        'llama_results': results.get('llama_results', {}),
                        'current_process': results.get('current_process', 'Complete')
                    }
                    
                    # Clean up
                    os.remove(filepath)
                    
                    return jsonify(response_data)
                    
                except Exception as e:
                    return jsonify({'error': f'Processing failed: {str(e)}'})
        
        @self.app.route('/api/analyze', methods=['POST'])
        def api_analyze():
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file:
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(self.temp_dir, filename)
                    file.save(filepath)
                    
                    # Run analysis
                    results = self.analyzer.run_full_analysis(filepath)
                    
                    if results is None:
                        return jsonify({'error': 'Analysis failed'}), 500
                    
                    # Format JSON response
                    response_data = self.format_api_response(results)
                    
                    # Clean up
                    os.remove(filepath)
                    
                    return jsonify(response_data)
                    
                except Exception as e:
                    return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
        @self.app.route('/api/demo', methods=['GET'])
        def api_demo():
            return jsonify({
                'status': 'success',
                'message': 'Fire Analysis API is running',
                'endpoints': {
                    '/api/analyze': 'POST - Upload an image for material analysis',
                    '/api/demo': 'GET - Test endpoint',
                    '/upload': 'POST - Web interface upload endpoint',
                    '/status': 'GET - Server status information'
                }
            })
        
        @self.app.route('/status')
        def status():
            return jsonify({
                'model_loaded': self.analyzer.model_loaded,
                'db_loaded': self.analyzer.db_loaded,
                'current_process': self.analyzer.current_process
            })
    
    def image_to_base64(self, image):
        """Convert image array to base64 string"""
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                    pass  # Already RGB
                
                pil_image = Image.fromarray(image.astype(np.uint8))
                buffer = io.BytesIO()
                pil_image.save(buffer, format='PNG')
                buffer.seek(0)
                return base64.b64encode(buffer.getvalue()).decode()
            return ""
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return ""
    
    def process_color_features(self, color_features):
        """Process color features for web response"""
        if not color_features or 'final_results' not in color_features:
            return None
        
        processed = {}
        for method, result in color_features['final_results'].items():
            processed[method] = {
                'rgb': result['rgb'].tolist() if isinstance(result['rgb'], np.ndarray) else result['rgb'],
                'lab': result['lab'].tolist() if isinstance(result['lab'], np.ndarray) else result['lab']
            }
        return processed
        
    def format_api_response(self, results):
        """Format analysis results for API response"""
        if not results:
            return {'fire_detected': False, 'error': 'Analysis failed'}
            
        # Check if fire was detected
        fire_detected = len(results.get('boxes', [])) > 0
        
        if not fire_detected:
            return {'fire_detected': False}
            
        # Get LLaMA results
        llama_results = results.get('llama_results', {})
        average_result = llama_results.get('average_result', {})
        
        # Get top material
        top_material = None
        top_confidence = 0
        recommended_extinguisher = "Unknown"
        
        all_materials = []
        
        if isinstance(average_result, dict) and 'top_materials' in average_result:
            materials = average_result.get('top_materials', [])
            
            for material in materials:
                if isinstance(material, dict):
                    material_name = material.get('name', 'Unknown')
                    similarity = material.get('similarity', 0)
                    extinguishers = material.get('extinguishers', ['Unknown'])
                    
                    all_materials.append({
                        'material': material_name,
                        'similarity': similarity
                    })
                    
                    if similarity > top_confidence:
                        top_confidence = similarity
                        top_material = material_name
                        if extinguishers and len(extinguishers) > 0:
                            recommended_extinguisher = extinguishers[0]
        
        # Get AI reasoning
        ai_reasoning = ""
        if isinstance(average_result, dict) and 'reasoning' in average_result:
            ai_reasoning = average_result.get('reasoning', '')
        
        # Format the response according to the specified format
        response = {
            'fire_detected': fire_detected,
            'top_material': top_material or "Unknown",
            'confidence': top_confidence,
            'extinguisher': recommended_extinguisher,
            'all_results': all_materials,
            'ai_reasoning': ai_reasoning
        }
        
        return response
    
    def find_free_port(self):
        """Find a free port for the server"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def start_server(self):
        """Start the Flask server in a separate thread"""
        try:
            self.server_port = 8080  # Use fixed port 8080 instead of find_free_port()
            self.server_url = f"http://localhost:{self.server_port}"
            
            def run_server():
                self.app.run(host='0.0.0.0', port=self.server_port, debug=False, use_reloader=False)
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            return True, self.server_url
        except Exception as e:
            print(f"Error starting server: {e}")
            return False, str(e)
    
    def stop_server(self):
        """Stop the Flask server"""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            # Note: Flask development server doesn't have a clean shutdown method
            # In production, you'd use a proper WSGI server like Gunicorn
            return True
        except Exception as e:
            print(f"Error stopping server: {e}")
            return False


# Enhanced Web template with FIXED JavaScript and improved UI
WEB_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Flame Material Identification</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            color: white;
            min-height: 100vh;
            animation: backgroundShift 10s ease-in-out infinite alternate;
        }
        
        @keyframes backgroundShift {
            0% { background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%); }
            100% { background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%); }
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 30px;
            backdrop-filter: blur(15px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            font-size: 2.8rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            background: linear-gradient(45deg, #FFD700, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
            font-weight: 500;
        }
        
        .upload-section {
            background: rgba(255,255,255,0.15);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            backdrop-filter: blur(15px);
            border: 2px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        }
        
        .upload-area {
            border: 3px dashed rgba(255,255,255,0.6);
            border-radius: 15px;
            padding: 50px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .upload-area::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: left 0.6s ease;
        }
        
        .upload-area:hover::before {
            left: 100%;
        }
        
        .upload-area:hover {
            border-color: rgba(255,255,255,0.9);
            background: rgba(255,255,255,0.08);
            transform: scale(1.02);
        }
        
        .upload-area.dragover {
            border-color: #4CAF50;
            background: rgba(76, 175, 80, 0.15);
            transform: scale(1.05);
        }
        
        #fileInput {
            display: none;
        }
        
        .upload-btn {
            background: linear-gradient(45deg, #4CAF50, #45a049, #66BB6A);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 30px;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            margin-top: 20px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        
        .upload-btn:hover {
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 8px 25px rgba(76, 175, 80, 0.4);
        }
        
        .status {
            text-align: center;
            margin: 20px 0;
            font-weight: bold;
            font-size: 18px;
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 30px 0;
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .spinner {
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            display: none;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        
        .result-card {
            background: rgba(255,255,255,0.15);
            border-radius: 20px;
            padding: 25px;
            backdrop-filter: blur(15px);
            border: 2px solid rgba(255,255,255,0.2);
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        
        .result-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        
        .result-card h3 {
            margin-bottom: 20px;
            color: #FFD700;
            font-size: 1.4rem;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        
        .result-image {
            max-width: 100%;
            height: auto;
            border-radius: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: transform 0.3s ease;
        }
        
        .result-image:hover {
            transform: scale(1.05);
        }
        
        .color-samples {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 20px;
            justify-content: center;
        }
        
        .color-sample {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 8px;
        }
        
        .color-box {
            width: 60px;
            height: 60px;
            border-radius: 12px;
            border: 3px solid rgba(255,255,255,0.4);
            margin-bottom: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: transform 0.3s ease;
        }
        
        .color-box:hover {
            transform: scale(1.1);
        }
        
        .color-label {
            font-size: 11px;
            text-align: center;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        
        .individual-results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .method-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
            border-radius: 20px;
            padding: 25px;
            border: 2px solid rgba(255,255,255,0.2);
            transition: all 0.4s ease;
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        
        .method-card:hover {
            transform: translateY(-10px) rotateX(5deg);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }
        
        .method-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .method-name {
            font-size: 1.2rem;
            font-weight: bold;
            color: #64B5F6;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
        
        .confidence-badge {
            background: linear-gradient(45deg, #FF6B6B, #FF8E53);
            color: white;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
            box-shadow: 0 2px 8px rgba(255, 107, 107, 0.3);
        }
        
        .method-materials {
            margin-bottom: 15px;
        }
        
        .method-reasoning {
            font-size: 0.95rem;
            line-height: 1.5;
            color: #E8E8E8;
        }
        
        .materials-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            overflow: hidden;
        }
        
        .materials-table th,
        .materials-table td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        
        .materials-table th {
            background: rgba(255,255,255,0.15);
            font-weight: bold;
            font-size: 1.1rem;
        }
        
        .materials-table tr:hover {
            background: rgba(255,255,255,0.08);
        }
        
        .analysis-text {
            background: rgba(255,255,255,0.08);
            padding: 20px;
            border-radius: 15px;
            margin-top: 20px;
            line-height: 1.7;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .section-title {
            font-size: 1.8rem;
            font-weight: bold;
            margin: 40px 0 25px 0;
            color: #4CAF50;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .error {
            color: #ff6b6b;
            text-align: center;
            margin: 20px 0;
            background: rgba(255, 107, 107, 0.1);
            padding: 20px;
            border-radius: 10px;
            border: 2px solid rgba(255, 107, 107, 0.3);
        }
        
        .sort-controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }
        
        .sort-btn {
            background: rgba(255,255,255,0.2);
            color: white;
            border: 2px solid rgba(255,255,255,0.3);
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
            font-weight: bold;
        }
        
        .sort-btn:hover {
            background: rgba(76, 175, 80, 0.3);
            border-color: #4CAF50;
            transform: translateY(-2px);
        }
        
        .sort-btn.active {
            background: #4CAF50;
            border-color: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 Enhanced Flame Material Identification</h1>
            <p>Multi-Clustering Analysis with Individual Method Results</p>
        </div>
        
        <div class="upload-section">
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <h3>📸 Upload Flame Image</h3>
                <p>Click here or drag and drop an image</p>
                <input type="file" id="fileInput" accept="image/*">
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    Select Image
                </button>
            </div>
        </div>
        
        <div class="status" id="status">🚀 Ready - Upload an image to begin analysis</div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>⚡ Analyzing flame with multiple clustering techniques...</p>
        </div>
        
        <div class="error" id="error" style="display: none;"></div>
        
        <div class="results" id="results">
            <div class="results-grid">
                <div class="result-card">
                    <h3>📷 Original Image</h3>
                    <img id="originalImage" class="result-image" alt="Original Image">
                </div>
                
                <div class="result-card">
                    <h3>🔍 Fire Detection</h3>
                    <img id="detectionImage" class="result-image" alt="Detection Result">
                </div>
                
                <div class="result-card">
                    <h3>✂️ Flame Segmentation</h3>
                    <img id="segmentationImage" class="result-image" alt="Segmentation Result">
                </div>
                
                <div class="result-card">
                    <h3>🎨 Color Analysis</h3>
                    <p>Multi-clustering flame color analysis:</p>
                    <div id="colorSamples" class="color-samples"></div>
                </div>
            </div>
            
            <div class="section-title">🧠 Individual Clustering Method Results</div>
            
            <div class="sort-controls">
                <button class="sort-btn active" onclick="sortResults('confidence', this)">Sort by Confidence</button>
                <button class="sort-btn" onclick="sortResults('similarity', this)">Sort by Similarity</button>
                <button class="sort-btn" onclick="sortResults('method', this)">Sort by Method</button>
            </div>
            
            <div id="individualResults" class="individual-results"></div>
            
            <div class="section-title">📊 Consensus Analysis</div>
            
            <div class="result-card">
                <h3>🎯 Material Identification Results (Average)</h3>
                <table id="materialsTable" class="materials-table">
                    <thead>
                        <tr>
                            <th>Material</th>
                            <th>Similarity</th>
                            <th>Extinguishers</th>
                            <th>Consensus</th>
                        </tr>
                    </thead>
                    <tbody id="materialsTableBody">
                    </tbody>
                </table>
            </div>
            
            <div class="result-card">
                <h3>🤖 AI Analysis (Consensus)</h3>
                <div id="reasoning" class="analysis-text"></div>
            </div>
            
            <div class="result-card">
                <h3>📈 Clustering Summary</h3>
                <div id="clusteringSummary" class="analysis-text"></div>
            </div>
        </div>
    </div>
    
    <script>
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.querySelector('.upload-area');
        const status = document.getElementById('status');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const error = document.getElementById('error');
        let currentResults = null;
        
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Please select an image file');
                return;
            }
            
            if (file.size > 16 * 1024 * 1024) {
                showError('File size must be less than 16MB');
                return;
            }
            
            uploadImage(file);
        }
        
        function uploadImage(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            // Show loading
            loading.style.display = 'block';
            results.style.display = 'none';
            error.style.display = 'none';
            status.textContent = 'Uploading and analyzing image...';
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                currentResults = data;
                displayResults(data);
            })
            .catch(err => {
                loading.style.display = 'none';
                showError('Upload failed: ' + err.message);
            });
        }
        
        function displayResults(data) {
            // Display images
            if (data.original_image) {
                document.getElementById('originalImage').src = 'data:image/png;base64,' + data.original_image;
            }
            if (data.detection_result) {
                document.getElementById('detectionImage').src = 'data:image/png;base64,' + data.detection_result;
            }
            if (data.segmentation_mask) {
                document.getElementById('segmentationImage').src = 'data:image/png;base64,' + data.segmentation_mask;
            }
            
            // Display color analysis
            displayColorAnalysis(data.color_features);
            
            // Display individual results
            displayIndividualResults(data.llama_results);
            
            // Display material results (average)
            displayMaterialResults(data.llama_results);
            
            // Show results
            results.style.display = 'block';
            status.textContent = data.current_process || 'Analysis complete';
        }
        
        function displayColorAnalysis(colorFeatures) {
            const colorSamples = document.getElementById('colorSamples');
            colorSamples.innerHTML = '';
            
            if (!colorFeatures) {
                colorSamples.innerHTML = '<p>No color analysis available</p>';
                return;
            }
            
            Object.entries(colorFeatures).forEach(([method, colors]) => {
                const sample = document.createElement('div');
                sample.className = 'color-sample';
                
                const colorBox = document.createElement('div');
                colorBox.className = 'color-box';
                const rgb = colors.rgb;
                colorBox.style.backgroundColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
                
                const label = document.createElement('div');
                label.className = 'color-label';
                label.textContent = method;
                
                sample.appendChild(colorBox);
                sample.appendChild(label);
                colorSamples.appendChild(sample);
            });
        }
        
        function displayIndividualResults(llamaResults) {
            const container = document.getElementById('individualResults');
            container.innerHTML = '';
            
            if (!llamaResults || !llamaResults.individual_results) {
                container.innerHTML = '<p>No individual clustering results available</p>';
                return;
            }
            
            const individualResults = llamaResults.individual_results;
            
            // Convert to array for sorting
            window.individualResultsData = Object.entries(individualResults).map(([method, result]) => ({
                method,
                ...result
            }));
            
            // Sort by confidence (default)
            sortResults('confidence', document.querySelector('.sort-btn.active'));
        }
        
        function sortResults(criteria, buttonElement) {
            if (!window.individualResultsData) return;
            
            // Update active button
            document.querySelectorAll('.sort-btn').forEach(btn => btn.classList.remove('active'));
            if (buttonElement) {
                buttonElement.classList.add('active');
            }
            
            let sortedData = [...window.individualResultsData];
            
            switch(criteria) {
                case 'confidence':
                    sortedData.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
                    break;
                case 'similarity':
                    sortedData.sort((a, b) => {
                        const aMax = a.top_materials?.[0]?.similarity || 0;
                        const bMax = b.top_materials?.[0]?.similarity || 0;
                        return bMax - aMax;
                    });
                    break;
                case 'method':
                    sortedData.sort((a, b) => a.method.localeCompare(b.method));
                    break;
            }
            
            const container = document.getElementById('individualResults');
            container.innerHTML = '';
            
            sortedData.forEach(result => {
                const card = document.createElement('div');
                card.className = 'method-card';
                
                const topMaterial = result.top_materials?.[0];
                const confidence = result.confidence || 0;
                
                card.innerHTML = `
                    <div class="method-header">
                        <div class="method-name">${result.method}</div>
                        <div class="confidence-badge">${confidence.toFixed(1)}%</div>
                    </div>
                    
                    <div class="method-materials">
                        ${result.top_materials ? result.top_materials.slice(0, 3).map(material => `
                            <div style="margin-bottom: 5px;">
                                <strong>${material.name}</strong>: ${material.similarity.toFixed(1)}%
                                <br><small>Extinguishers: ${material.extinguishers.join(', ')}</small>
                            </div>
                        `).join('') : '<div>No materials identified</div>'}
                    </div>
                    
                    <div class="method-reasoning">
                        ${result.reasoning || 'No reasoning provided'}
                    </div>
                `;
                
                container.appendChild(card);
            });
        }
        
        function displayMaterialResults(llamaResults) {
            const tableBody = document.getElementById('materialsTableBody');
            const reasoning = document.getElementById('reasoning');
            const clusteringSummary = document.getElementById('clusteringSummary');
            
            // Clear previous results
            tableBody.innerHTML = '';
            
            const averageResult = llamaResults?.average_result;
            
            if (!averageResult || !averageResult.top_materials) {
                tableBody.innerHTML = '<tr><td colspan="4">No consensus material identification results</td></tr>';
            } else {
                averageResult.top_materials.forEach(material => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${material.name || 'Unknown'}</td>
                        <td>${(material.similarity || 0).toFixed(1)}%</td>
                        <td>${(material.extinguishers || ['Unknown']).join(', ')}</td>
                        <td>${(material.clustering_consensus || 0).toFixed(1)}%</td>
                    `;
                    tableBody.appendChild(row);
                });
            }
            
            // Display reasoning and summary
            reasoning.textContent = averageResult?.reasoning || 'No consensus reasoning available';
            clusteringSummary.textContent = averageResult?.clustering_summary || 'No clustering summary available';
        }
        
        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
            status.textContent = 'Error occurred';
        }
        
        // Check server status on load
        fetch('/status')
        .then(response => response.json())
        .then(data => {
            let statusText = 'Server Status: ';
            statusText += data.model_loaded ? 'Models loaded' : 'Models not found';
            statusText += ' | ';
            statusText += data.db_loaded ? 'Database loaded' : 'Database not found';
            status.textContent = statusText;
        })
        .catch(() => {
            status.textContent = 'Server connection failed';
        });
    </script>
</body>
</html>
'''


class FlameAnalyzerGUI:
    def __init__(self):
        self.analyzer = FlameAnalyzer()
        self.flask_server = FlaskWebServer(self.analyzer)
        self.ngrok_manager = NgrokManager()
        self.root = ctk.CTk()
        self.root.title("Enhanced Flame Material Identification Pipeline")
        self.root.geometry("1800x1200")
        self.image_path = None
        self.server_running = False
        self.ngrok_running = False
        self.local_mode = True
        self.setup_gui()
        
        # Start process monitor
        self.update_process_status()
        
    def update_process_status(self):
        """Update current process status"""
        if hasattr(self, 'status_label'):
            if self.local_mode:
                self.status_label.configure(text=self.analyzer.current_process)
            else:
                self.status_label.configure(text="Server Mode - Local functionality disabled")
        self.root.after(500, self.update_process_status)  # Update every 500ms
        
    def setup_gui(self):
        """Set up the enhanced CustomTkinter GUI with server functionality"""
        # Set colorful appearance for local UI
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")
        
        # Main container with gradient-like effect
        main_frame = ctk.CTkFrame(self.root, fg_color=["#2B2B2B", "#1F1F1F"])
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header with colorful title and status
        header_frame = ctk.CTkFrame(main_frame, fg_color=["#3B4252", "#2E3440"])
        header_frame.pack(fill="x", pady=(0, 20))
        
        # Colorful Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="Spectral Flame Analyzer",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=["#FF6B6B", "#4ECDC4"]  # Gradient-like colors
        )
        title_label.pack(pady=15)
        
        # Status panel with colors
        status_frame = ctk.CTkFrame(header_frame, fg_color=["#4C566A", "#3B4252"])
        status_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        model_status = "Models loaded" if self.analyzer.model_loaded else "Models not found"
        db_status = "Database loaded" if self.analyzer.db_loaded else "Database not found"
        
        status_info = ctk.CTkLabel(
            status_frame,
            text=f"Status: {model_status} | {db_status}",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=["#A3BE8C", "#88C0D0"]
        )
        status_info.pack(pady=12)
        
        # Replace the entire control panel section (starting from "# Control panel with colorful server controls in rows")
        # with this compact single-row version:

        # Compact control panel - all buttons in one row
        control_frame = ctk.CTkFrame(main_frame, fg_color=["#434C5E", "#2E3440"])
        control_frame.pack(fill="x", pady=(0, 20))

        # Single row for all controls
        button_row_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_row_frame.pack(fill="x", padx=20, pady=15)

        # Local analysis buttons
        local_label = ctk.CTkLabel(button_row_frame, text="Local:", 
                                font=ctk.CTkFont(size=12, weight="bold"),
                                text_color=["#A3BE8C", "#8FB572"])
        local_label.pack(side="left", padx=(0, 5))

        self.load_btn = ctk.CTkButton(
            button_row_frame,
            text="Load",
            command=self.load_image,
            width=60,
            height=28,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=["#BF616A", "#A54A5A"],
            hover_color=["#D08770", "#B85A50"]
        )
        self.load_btn.pack(side="left", padx=2)

        self.analyze_btn = ctk.CTkButton(
            button_row_frame,
            text="Analyze",
            command=self.run_analysis,
            width=70,
            height=28,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=["#A3BE8C", "#8FB572"],
            hover_color=["#EBCB8B", "#D4AC6A"]
        )
        self.analyze_btn.pack(side="left", padx=(2, 15))

        # Server buttons
        server_label = ctk.CTkLabel(button_row_frame, text="Server:", 
                                font=ctk.CTkFont(size=12, weight="bold"),
                                text_color=["#88C0D0", "#5E81AC"])
        server_label.pack(side="left", padx=(0, 5))

        self.start_server_btn = ctk.CTkButton(
            button_row_frame,
            text="Start",
            command=self.start_server,
            width=55,
            height=28,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=["#8FBCBB", "#7CA8A7"],
            hover_color=["#A3BE8C", "#8FB572"]
        )
        self.start_server_btn.pack(side="left", padx=2)

        self.stop_server_btn = ctk.CTkButton(
            button_row_frame,
            text="Stop",
            command=self.stop_server,
            width=50,
            height=28,
            font=ctk.CTkFont(size=10, weight="bold"),
            state="disabled",
            fg_color=["#BF616A", "#A54A5A"],
            hover_color=["#D08770", "#B85A50"]
        )
        self.stop_server_btn.pack(side="left", padx=2)

        self.open_browser_btn = ctk.CTkButton(
            button_row_frame,
            text="Browser",
            command=self.open_browser,
            width=65,
            height=28,
            font=ctk.CTkFont(size=10, weight="bold"),
            state="disabled",
            fg_color=["#EBCB8B", "#D4AC6A"],
            hover_color=["#D08770", "#B85A50"]
        )
        self.open_browser_btn.pack(side="left", padx=(2, 15))

        # Ngrok buttons
        ngrok_label = ctk.CTkLabel(button_row_frame, text="Online:", 
                                font=ctk.CTkFont(size=12, weight="bold"),
                                text_color=["#B48EAD", "#8B6F9B"])
        ngrok_label.pack(side="left", padx=(0, 5))

        self.start_ngrok_btn = ctk.CTkButton(
            button_row_frame,
            text="Start",
            command=self.start_ngrok,
            width=55,
            height=28,
            font=ctk.CTkFont(size=10, weight="bold"),
            state="disabled",
            fg_color=["#A3BE8C", "#8FB572"],
            hover_color=["#EBCB8B", "#D4AC6A"]
        )
        self.start_ngrok_btn.pack(side="left", padx=2)

        self.stop_ngrok_btn = ctk.CTkButton(
            button_row_frame,
            text="Stop",
            command=self.stop_ngrok,
            width=50,
            height=28,
            font=ctk.CTkFont(size=10, weight="bold"),
            state="disabled",
            fg_color=["#BF616A", "#A54A5A"],
            hover_color=["#D08770", "#B85A50"]
        )
        self.stop_ngrok_btn.pack(side="left", padx=2)

        self.qr_btn = ctk.CTkButton(
            button_row_frame,
            text="QR",
            command=self.show_qr_code,
            width=40,
            height=28,
            font=ctk.CTkFont(size=10, weight="bold"),
            state="disabled",
            fg_color=["#D08770", "#B85A50"],
            hover_color=["#EBCB8B", "#D4AC6A"]
        )
        self.qr_btn.pack(side="left", padx=2)

        # Status label - right aligned with more space
        self.status_label = ctk.CTkLabel(
            button_row_frame,
            text="Ready - Load an image to start analysis",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=["#ECEFF4", "#D8DEE9"]
        )
        self.status_label.pack(side="right", padx=10)
        
        # Results notebook with colorful tabs
        self.notebook = ctk.CTkTabview(main_frame, 
                                      segmented_button_fg_color=["#5E81AC", "#4C566A"],
                                      segmented_button_selected_color=["#88C0D0", "#6B8CAE"],
                                      segmented_button_selected_hover_color=["#A3BE8C", "#8FB572"])
        self.notebook.pack(fill="both", expand=True)
        
        self.create_tabs()
        
    def create_tabs(self):
        """Create enhanced GUI tabs with individual clustering results"""
        # Original Image tab
        self.notebook.add("Original Image")
        original_tab = self.notebook.tab("Original Image")
        original_frame = ctk.CTkFrame(original_tab, fg_color=["#3B4252", "#2E3440"])
        original_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.image_label = ctk.CTkLabel(original_frame, text="No image loaded", 
                                       font=ctk.CTkFont(size=16),
                                       text_color=["#D8DEE9", "#ECEFF4"])
        self.image_label.pack(expand=True)
        
        # Detection tab
        self.notebook.add("Fire Detection")
        detection_tab = self.notebook.tab("Fire Detection")
        detection_frame = ctk.CTkFrame(detection_tab, fg_color=["#434C5E", "#3B4252"])
        detection_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.detection_label = ctk.CTkLabel(detection_frame, text="No detection results", 
                                           font=ctk.CTkFont(size=16),
                                           text_color=["#D8DEE9", "#ECEFF4"])
        self.detection_label.pack(expand=True)
        
        # Segmentation tab
        self.notebook.add("Segmentation")
        segmentation_tab = self.notebook.tab("Segmentation")
        segmentation_frame = ctk.CTkFrame(segmentation_tab, fg_color=["#4C566A", "#434C5E"])
        segmentation_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.segmentation_label = ctk.CTkLabel(segmentation_frame, text="No segmentation results", 
                                              font=ctk.CTkFont(size=16),
                                              text_color=["#D8DEE9", "#ECEFF4"])
        self.segmentation_label.pack(expand=True)
        
        # Enhanced Clustering Results tab
        self.notebook.add("Clustering Analysis")
        clustering_tab = self.notebook.tab("Clustering Analysis")
        
        # Create scrollable frame for clustering results
        self.clustering_scrollable = ctk.CTkScrollableFrame(clustering_tab, 
                                                           fg_color=["#3B4252", "#2E3440"])
        self.clustering_scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Individual Clustering Results tab
        self.notebook.add("Individual Results")
        individual_tab = self.notebook.tab("Individual Results")
        
        # Create scrollable frame for individual results
        self.individual_scrollable = ctk.CTkScrollableFrame(individual_tab,
                                                           fg_color=["#434C5E", "#3B4252"])
        self.individual_scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Material Results tab (Consensus Results from Averaging Multiple Clustering Methods)
        self.notebook.add("Consensus Results")
        results_tab = self.notebook.tab("Consensus Results")
        
        # Results container
        results_container = ctk.CTkFrame(results_tab, fg_color=["#4C566A", "#434C5E"])
        results_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Results table frame
        table_frame = ctk.CTkFrame(results_container, fg_color=["#5E81AC", "#4C566A"])
        table_frame.pack(fill="x", pady=(0, 10))
        
        table_label = ctk.CTkLabel(table_frame, text="Consensus Material Identification (Averaged from Multiple Clustering Methods)", 
                                  font=ctk.CTkFont(size=18, weight="bold"),
                                  text_color="white")
        table_label.pack(pady=10)
        
        # Create results table using tkinter Treeview (wrapped in CTkFrame)
        import tkinter.ttk as ttk
        tree_frame = ctk.CTkFrame(table_frame, fg_color="transparent")
        tree_frame.pack(fill="x", padx=10, pady=10)
        
        columns = ('Method/Material', 'Similarity %', 'Extinguishers', 'Status')
        self.results_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=6)
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=200)
        
        # Add scrollbar for tree
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.results_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        
        # Results text areas
        text_frame = ctk.CTkFrame(results_container, fg_color="transparent")
        text_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        # AI Reasoning section
        reasoning_frame = ctk.CTkFrame(text_frame, fg_color=["#88C0D0", "#5E81AC"])
        reasoning_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        reasoning_label = ctk.CTkLabel(reasoning_frame, text="AI Consensus Reasoning (Averaged Analysis)", 
                                     font=ctk.CTkFont(size=16, weight="bold"),
                                     text_color="white")
        reasoning_label.pack(pady=(10, 5))
        
        self.reasoning_text = ctk.CTkTextbox(reasoning_frame, wrap="word", height=120,
                                            fg_color=["#D8DEE9", "#4C566A"],
                                            text_color=["#2E3440", "#ECEFF4"])
        self.reasoning_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Clustering Summary section
        summary_frame = ctk.CTkFrame(text_frame, fg_color=["#A3BE8C", "#8FB572"])
        summary_frame.pack(fill="both", expand=True, padx=10)
        
        summary_label = ctk.CTkLabel(summary_frame, text="Multi-Clustering Method Summary", 
                                   font=ctk.CTkFont(size=16, weight="bold"),
                                   text_color="white")
        summary_label.pack(pady=(10, 5))
        
        self.clustering_summary_text = ctk.CTkTextbox(summary_frame, wrap="word", height=100,
                                                     fg_color=["#D8DEE9", "#4C566A"],
                                                     text_color=["#2E3440", "#ECEFF4"])
        self.clustering_summary_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Server Information tab
        self.notebook.add("Server & Ngrok Info")
        server_tab = self.notebook.tab("Server & Ngrok Info")
        
        # Server info container
        server_info_container = ctk.CTkScrollableFrame(server_tab, fg_color=["#B48EAD", "#8B6F9B"])
        server_info_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Server status section
        server_status_frame = ctk.CTkFrame(server_info_container, fg_color=["#EBCB8B", "#D4AC6A"])
        server_status_frame.pack(fill="x", pady=(0, 20))
        
        server_status_label = ctk.CTkLabel(server_status_frame, text="Server & Ngrok Status", 
                                         font=ctk.CTkFont(size=18, weight="bold"),
                                         text_color=["#2E3440", "#3B4252"])
        server_status_label.pack(pady=(10, 15))
        
        self.server_status_text = ctk.CTkTextbox(server_status_frame, height=150,
                                                fg_color=["#ECEFF4", "#4C566A"],
                                                text_color=["#2E3440", "#D8DEE9"])
        self.server_status_text.pack(fill="x", padx=10, pady=(0, 10))
        
        # Server details section
        server_details_frame = ctk.CTkFrame(server_info_container, fg_color=["#D08770", "#B85A50"])
        server_details_frame.pack(fill="x", pady=(0, 20))
        
        server_details_label = ctk.CTkLabel(server_details_frame, text="Technical Details", 
                                          font=ctk.CTkFont(size=18, weight="bold"),
                                          text_color="white")
        server_details_label.pack(pady=(10, 15))
        
        self.server_details_text = ctk.CTkTextbox(server_details_frame, height=200,
                                                 fg_color=["#ECEFF4", "#4C566A"],
                                                 text_color=["#2E3440", "#D8DEE9"])
        self.server_details_text.pack(fill="x", padx=10, pady=(0, 10))
        
        # Initialize server info texts
        self.update_server_info()
        
    def update_server_info(self):
        """Update server information in the server tab"""
        # Server status
        self.server_status_text.delete("1.0", "end")
        status_info = ""
        
        if self.server_running:
            status_info += f"Local Server: RUNNING\n"
            status_info += f"Local URL: {self.flask_server.server_url}\n"
            status_info += f"Port: {self.flask_server.server_port}\n"
        else:
            status_info += "Local Server: STOPPED\n"
        
        if self.ngrok_running:
            status_info += f"Ngrok Tunnel: RUNNING\n"
            status_info += f"Public URL: {self.ngrok_manager.public_url}\n"
        else:
            status_info += "Ngrok Tunnel: STOPPED\n"
        
        status_info += f"Local Mode: {'DISABLED' if not self.local_mode else 'ENABLED'}\n"
        
        self.server_status_text.insert("1.0", status_info)
        
        # Server details
        self.server_details_text.delete("1.0", "end")
        details_info = f"""Enhanced Flame Analyzer Configuration:

Server Framework: Flask (Development Server)
Host: 0.0.0.0 (All interfaces)
Max Upload Size: 16MB
Supported Formats: JPG, PNG, BMP, TIFF
Clustering Methods: VB-GMM, GMM, K-Means++, DBSCAN, Mean-Shift, Agglomerative

Individual Analysis Features:
• Separate LLaMA queries for each clustering method
• Method-specific confidence scoring
• Sortable results by confidence/similarity/method
• Card-style UI with detailed material identification

Ngrok Configuration:
Token: {'Authenticated' if self.ngrok_manager.authenticated else 'Not authenticated'}
Status: {'Running' if self.ngrok_running else 'Stopped'}
Public Access: {'Available' if self.ngrok_running else 'Not available'}

API Endpoints:
• GET /: Enhanced web interface with individual clustering results
• POST /upload: Image analysis with individual + consensus results
• GET /status: Server health check
• POST /api/analyze: API endpoint for image analysis (JSON response)
• GET /api/demo: Simple test endpoint for API verification

Security Notes:
• Development server (not for production)
• Ngrok provides HTTPS public access
• Temporary files auto-cleaned
• QR code generation for easy mobile access"""
        
        self.server_details_text.insert("1.0", details_info)
            
    def start_server(self):
        """Start the Flask web server"""
        try:
            if not self.analyzer.model_loaded:
                messagebox.showwarning("Warning", "Models not loaded. Please ensure required files are present.")
                return
            
            success, result = self.flask_server.start_server()
            
            if success:
                self.server_running = True
                self.local_mode = False
                
                # Update UI
                self.start_server_btn.configure(state="disabled")
                self.stop_server_btn.configure(state="normal")
                self.open_browser_btn.configure(state="normal")
                self.start_ngrok_btn.configure(state="normal")
                
                # Disable local analysis buttons
                self.load_btn.configure(state="disabled")
                self.analyze_btn.configure(state="disabled")
                
                # Update server info
                self.update_server_info()
                
                # Switch to server info tab
                self.notebook.set("Server & Ngrok Info")
                
                messagebox.showinfo("Server Started", 
                                  f"Web server started successfully!\n\nLocal URL: {result}\n\nLocal analysis is now disabled.\nYou can now start Ngrok for online access.")
            else:
                messagebox.showerror("Server Error", f"Failed to start server: {result}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error starting server: {str(e)}")
    
    def stop_server(self):
        """Stop the Flask web server"""
        try:
            # Stop ngrok first if running
            if self.ngrok_running:
                self.stop_ngrok()
            
            success = self.flask_server.stop_server()
            
            self.server_running = False
            self.local_mode = True
            
            # Update UI
            self.start_server_btn.configure(state="normal")
            self.stop_server_btn.configure(state="disabled")
            self.open_browser_btn.configure(state="disabled")
            self.start_ngrok_btn.configure(state="disabled")
            
            # Enable local analysis buttons
            self.load_btn.configure(state="normal")
            self.analyze_btn.configure(state="normal")
            
            # Update server info
            self.update_server_info()
            
            messagebox.showinfo("Server Stopped", "Web server stopped. Local analysis is now enabled.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error stopping server: {str(e)}")
    
    def start_ngrok(self):
        """Start Ngrok tunnel"""
        try:
            if not self.server_running:
                messagebox.showwarning("Warning", "Please start the web server first.")
                return
            
            success, result = self.ngrok_manager.start_tunnel(self.flask_server.server_port)
            
            if success:
                self.ngrok_running = True
                
                # Update UI
                self.start_ngrok_btn.configure(state="disabled")
                self.stop_ngrok_btn.configure(state="normal")
                self.qr_btn.configure(state="normal")
                
                # Update server info
                self.update_server_info()
                
                messagebox.showinfo("Ngrok Started", 
                                  f"Ngrok tunnel started successfully!\n\nPublic URL: {result}\n\nYour app is now accessible worldwide!\nClick 'Show QR Code' for easy mobile access.")
            else:
                messagebox.showerror("Ngrok Error", f"Failed to start Ngrok: {result}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error starting Ngrok: {str(e)}")
    
    def stop_ngrok(self):
        """Stop Ngrok tunnel"""
        try:
            success = self.ngrok_manager.stop_tunnel()
            
            self.ngrok_running = False
            
            # Update UI
            self.start_ngrok_btn.configure(state="normal" if self.server_running else "disabled")
            self.stop_ngrok_btn.configure(state="disabled")
            self.qr_btn.configure(state="disabled")
            
            # Update server info
            self.update_server_info()
            
            if success:
                messagebox.showinfo("Ngrok Stopped", "Ngrok tunnel stopped. App is no longer publicly accessible.")
            else:
                messagebox.showwarning("Warning", "There may have been issues stopping Ngrok.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error stopping Ngrok: {str(e)}")
    
    def show_qr_code(self):
        """Generate and show QR code for the public URL"""
        try:
            if not self.ngrok_running or not self.ngrok_manager.public_url:
                messagebox.showwarning("Warning", "Ngrok is not running or no public URL available.")
                return
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(self.ngrok_manager.public_url)
            qr.make(fit=True)
            
            # Create QR code image
            qr_image = qr.make_image(fill_color="black", back_color="white")
            
            # Create new window for QR code
            qr_window = ctk.CTkToplevel(self.root)
            qr_window.title("QR Code - Flame Analyzer")
            qr_window.geometry("400x500")
            qr_window.transient(self.root)
            qr_window.grab_set()
            
            # Convert PIL image to PhotoImage
            qr_photo = ImageTk.PhotoImage(qr_image)
            
            # Create label with QR code
            qr_label = ctk.CTkLabel(qr_window, image=qr_photo, text="")
            qr_label.pack(pady=20)
            
            # URL text
            url_label = ctk.CTkLabel(qr_window, text=self.ngrok_manager.public_url,
                                   font=ctk.CTkFont(size=12))
            url_label.pack(pady=10)
            
            # Instructions
            instructions = ctk.CTkLabel(qr_window, 
                                      text="Scan this QR code with your mobile device\nto access the Flame Analyzer remotely!",
                                      font=ctk.CTkFont(size=14, weight="bold"))
            instructions.pack(pady=20)
            
            # Close button
            close_btn = ctk.CTkButton(qr_window, text="Close", command=qr_window.destroy)
            close_btn.pack(pady=20)
            
            # Keep reference to prevent garbage collection
            qr_window.qr_photo = qr_photo
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating QR code: {str(e)}")
    
    def open_browser(self):
        """Open the web interface in default browser"""
        if self.server_running and self.flask_server.server_url:
            try:
                webbrowser.open(self.flask_server.server_url)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open browser: {str(e)}")
        else:
            messagebox.showwarning("Warning", "Server is not running")
            
    def load_image(self):
        """Load image from file (local mode only)"""
        if not self.local_mode:
            messagebox.showwarning("Warning", "Local analysis disabled while server is running")
            return
            
        file_path = filedialog.askopenfilename(
            title="Select Flame Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path, self.image_label)
            self.analyzer.current_process = "Image loaded - Ready for analysis"
            
    def display_image(self, image_path_or_array, label):
        """Display image in GUI"""
        try:
            if isinstance(image_path_or_array, str):
                image = Image.open(image_path_or_array)
            else:
                if isinstance(image_path_or_array, np.ndarray):
                    if len(image_path_or_array.shape) == 2:
                        colored_mask = np.zeros((image_path_or_array.shape[0], image_path_or_array.shape[1], 3), dtype=np.uint8)
                        colored_mask[image_path_or_array > 0] = [255, 0, 0]
                        image = Image.fromarray(colored_mask)
                    else:
                        image = Image.fromarray(image_path_or_array.astype(np.uint8))
                else:
                    return
                
            image.thumbnail((600, 400), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo, text="")
            label.image = photo
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            label.configure(text=f"Error displaying image: {str(e)}")
            
    def run_analysis(self):
        """Run the complete enhanced flame analysis (local mode only)"""
        if not self.local_mode:
            messagebox.showwarning("Warning", "Local analysis disabled while server is running")
            return
            
        if not self.image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return
            
        if not self.analyzer.model_loaded:
            messagebox.showwarning("Warning", "Models not loaded. Please ensure OBJ_best.pt is in the current directory.")
            return
            
        def analyze_in_thread():
            try:
                results = self.analyzer.run_full_analysis(self.image_path)
                
                if results:
                    self.root.after(0, lambda: self.display_results(results))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Analysis failed. Check console for details."))
            except Exception as e:
                print(f"Analysis thread error: {e}")
                
        threading.Thread(target=analyze_in_thread, daemon=True).start()
        
    def display_results(self, results):
        """Display all enhanced results in GUI"""
        try:
            # Original image
            if results.get('original_image') is not None:
                self.display_image(results['original_image'], self.image_label)
            
            # Detection results
            if results.get('detection_result') is not None:
                self.display_image(results['detection_result'], self.detection_label)
                
            # Segmentation results - FIXED
            if results.get('segmentation_mask') is not None and results.get('original_image') is not None:
                original = results['original_image'].copy()
                mask = results['segmentation_mask']
                
                # Create blue overlay only where mask is positive
                segmented_image = original.copy()
                blue_overlay = np.zeros_like(original)
                blue_overlay[:, :] = [0, 0, 255]  # Blue color
                
                # Apply mask: blend original with blue where mask > 0
                mask_3d = np.stack([mask, mask, mask], axis=2) > 0
                segmented_image = np.where(mask_3d,
                                         cv2.addWeighted(original, 0.5, blue_overlay, 0.5, 0),
                                         original)
                
                self.display_image(segmented_image, self.segmentation_label)
                
            # Enhanced clustering analysis
            if results.get('color_features'):
                self.display_clustering_analysis(results['color_features'])
                
            # Individual clustering results
            if results.get('llama_results'):
                self.display_individual_results(results['llama_results'])
                
            # Enhanced LLaMA results (consensus)
            if results.get('llama_results'):
                self.display_enhanced_results(results['llama_results'])
                
        except Exception as e:
            print(f"Error displaying results: {e}")
    
    def display_individual_results(self, llama_results):
        """Display individual clustering method results in card-style UI"""
        try:
            # Clear previous results
            for widget in self.individual_scrollable.winfo_children():
                widget.destroy()
            
            individual_results = llama_results.get('individual_results', {})
            
            if not individual_results:
                no_results_label = ctk.CTkLabel(self.individual_scrollable, 
                                              text="No individual clustering results available",
                                              font=ctk.CTkFont(size=16),
                                              text_color=["#D8DEE9", "#ECEFF4"])
                no_results_label.pack(pady=20)
                return
            
            # Title
            title_label = ctk.CTkLabel(self.individual_scrollable, 
                                     text="Individual Clustering Method Analysis",
                                     font=ctk.CTkFont(size=20, weight="bold"),
                                     text_color=["#88C0D0", "#A3BE8C"])
            title_label.pack(pady=20)
            
            # Sort results by confidence (highest first)
            sorted_results = sorted(individual_results.items(), 
                                  key=lambda x: x[1].get('confidence', 0), reverse=True)
            
            # Create cards for each method
            cards_frame = ctk.CTkFrame(self.individual_scrollable, fg_color="transparent")
            cards_frame.pack(fill="x", padx=20, pady=10)
            
            for i, (method_name, result) in enumerate(sorted_results):
                # Method card with colorful design
                method_card = ctk.CTkFrame(cards_frame, fg_color=["#5E81AC", "#4C566A"])
                method_card.pack(fill="x", pady=10, padx=10)
                
                # Card header with method name and confidence
                header_frame = ctk.CTkFrame(method_card, fg_color="transparent")
                header_frame.pack(fill="x", padx=15, pady=(15, 10))
                
                method_label = ctk.CTkLabel(header_frame, 
                                          text=f"{method_name} Analysis",
                                          font=ctk.CTkFont(size=18, weight="bold"),
                                          text_color=["#ECEFF4", "#D8DEE9"])
                method_label.pack(side="left")
                
                confidence = result.get('confidence', 0)
                confidence_color = ["#A3BE8C", "#8FB572"] if confidence > 70 else ["#EBCB8B", "#D4AC6A"] if confidence > 40 else ["#BF616A", "#A54A5A"]
                confidence_label = ctk.CTkLabel(header_frame,
                                              text=f"Confidence: {confidence:.1f}%",
                                              font=ctk.CTkFont(size=14, weight="bold"),
                                              text_color=confidence_color)
                confidence_label.pack(side="right")
                
                # Top materials section
                materials_frame = ctk.CTkFrame(method_card, fg_color=["#88C0D0", "#5E81AC"])
                materials_frame.pack(fill="x", padx=15, pady=(0, 10))
                
                materials_label = ctk.CTkLabel(materials_frame,
                                             text="Top Material Matches:",
                                             font=ctk.CTkFont(size=14, weight="bold"),
                                             text_color="white")
                materials_label.pack(anchor="w", padx=10, pady=(10, 5))
                
                top_materials = result.get('top_materials', [])
                if top_materials:
                    for j, material in enumerate(top_materials[:3]):  # Show top 3
                        material_frame = ctk.CTkFrame(materials_frame, fg_color=["#B48EAD", "#8B6F9B"])
                        material_frame.pack(fill="x", padx=10, pady=2)
                        
                        # Material info
                        material_info = ctk.CTkLabel(material_frame,
                                                   text=f"#{j+1} {material.get('name', 'Unknown')} - {material.get('similarity', 0):.1f}%",
                                                   font=ctk.CTkFont(size=12, weight="bold"),
                                                   text_color="white")
                        material_info.pack(side="left", padx=10, pady=5)
                        
                        # Extinguishers
                        extinguishers = material.get('extinguishers', ['Unknown'])
                        ext_label = ctk.CTkLabel(material_frame,
                                               text=f"Extinguishers: {', '.join(extinguishers)}",
                                               font=ctk.CTkFont(size=10),
                                               text_color=["#ECEFF4", "#D8DEE9"])
                        ext_label.pack(side="right", padx=10, pady=5)
                else:
                    no_materials_label = ctk.CTkLabel(materials_frame,
                                                    text="No materials identified",
                                                    font=ctk.CTkFont(size=12),
                                                    text_color=["#D8DEE9", "#ECEFF4"])
                    no_materials_label.pack(anchor="w", padx=10, pady=5)
                
                # Reasoning section
                reasoning_frame = ctk.CTkFrame(method_card, fg_color=["#D08770", "#B85A50"])
                reasoning_frame.pack(fill="x", padx=15, pady=(0, 15))
                
                reasoning_label = ctk.CTkLabel(reasoning_frame,
                                             text="AI Reasoning:",
                                             font=ctk.CTkFont(size=14, weight="bold"),
                                             text_color="white")
                reasoning_label.pack(anchor="w", padx=10, pady=(10, 5))
                
                reasoning_text = ctk.CTkTextbox(reasoning_frame, height=80, wrap="word",
                                               fg_color=["#ECEFF4", "#4C566A"],
                                               text_color=["#2E3440", "#D8DEE9"])
                reasoning_text.pack(fill="x", padx=10, pady=(0, 10))
                
                reasoning_content = result.get('reasoning', 'No reasoning provided')
                reasoning_text.insert("1.0", reasoning_content)
                reasoning_text.configure(state="disabled")
                
        except Exception as e:
            print(f"Error displaying individual results: {e}")
            error_label = ctk.CTkLabel(self.individual_scrollable,
                                     text=f"Error displaying individual results: {str(e)}",
                                     font=ctk.CTkFont(size=14),
                                     text_color=["#BF616A", "#A54A5A"])
            error_label.pack(pady=20)
            
    def display_clustering_analysis(self, color_features):
        """Display comprehensive clustering analysis"""
        try:
            # Clear previous results
            for widget in self.clustering_scrollable.winfo_children():
                widget.destroy()
            
            if 'final_results' not in color_features:
                no_results_label = ctk.CTkLabel(self.clustering_scrollable, 
                                              text="No clustering results available",
                                              font=ctk.CTkFont(size=16),
                                              text_color=["#D8DEE9", "#ECEFF4"])
                no_results_label.pack(pady=20)
                return
            
            final_results = color_features['final_results']
            
            # Title
            title_label = ctk.CTkLabel(self.clustering_scrollable, 
                                     text="Multi-Clustering Analysis Results", 
                                     font=ctk.CTkFont(size=20, weight="bold"),
                                     text_color=["#88C0D0", "#A3BE8C"])
            title_label.pack(pady=20)
            
            # Create results table frame
            table_frame = ctk.CTkFrame(self.clustering_scrollable, fg_color=["#5E81AC", "#4C566A"])
            table_frame.pack(fill="x", padx=20, pady=10)
            
            # Table header
            header_frame = ctk.CTkFrame(table_frame, fg_color="transparent")
            header_frame.pack(fill="x", padx=10, pady=10)
            
            headers = ['Method', 'L*', 'a*', 'b*', 'R', 'G', 'B']
            for i, header in enumerate(headers):
                header_label = ctk.CTkLabel(header_frame, text=header, 
                                          font=ctk.CTkFont(size=14, weight="bold"),
                                          text_color="white",
                                          width=80)
                header_label.grid(row=0, column=i, padx=5, pady=5)
            
            # Populate with clustering results
            row = 1
            for method, result in final_results.items():
                method_frame = ctk.CTkFrame(table_frame, fg_color=["#88C0D0", "#5E81AC"])
                method_frame.pack(fill="x", padx=10, pady=2)
                
                lab = result['lab']
                rgb = result['rgb']
                
                values = [
                    method,
                    f"{lab[0]:.1f}",
                    f"{lab[1]:.1f}",
                    f"{lab[2]:.1f}",
                    f"{int(rgb[0])}",
                    f"{int(rgb[1])}",
                    f"{int(rgb[2])}"
                ]
                
                for i, value in enumerate(values):
                    value_label = ctk.CTkLabel(method_frame, text=str(value),
                                             font=ctk.CTkFont(size=12),
                                             text_color="white",
                                             width=80)
                    value_label.grid(row=0, column=i, padx=5, pady=5)
                
                row += 1
            
            # Color visualization section
            viz_frame = ctk.CTkFrame(self.clustering_scrollable, fg_color=["#B48EAD", "#8B6F9B"])
            viz_frame.pack(fill="x", padx=20, pady=20)
            
            viz_label = ctk.CTkLabel(viz_frame, text="Color Samples", 
                                   font=ctk.CTkFont(size=18, weight="bold"),
                                   text_color="white")
            viz_label.pack(pady=(10, 20))
            
            # Color samples grid
            colors_grid = ctk.CTkFrame(viz_frame, fg_color="transparent")
            colors_grid.pack(pady=(0, 20))
            
            cols = 3
            row = 0
            col = 0
            
            for method, result in final_results.items():
                color_frame = ctk.CTkFrame(colors_grid, fg_color=["#D08770", "#B85A50"])
                color_frame.grid(row=row, column=col, padx=10, pady=10)
                
                # Create color sample using CTkLabel with background color
                rgb = result['rgb']
                color_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                
                # Color display
                color_display = ctk.CTkLabel(color_frame, text="", width=80, height=60)
                color_display.configure(fg_color=color_hex)
                color_display.pack(pady=(10, 5))
                
                # Method name
                method_label = ctk.CTkLabel(color_frame, text=method,
                                          font=ctk.CTkFont(size=12, weight="bold"),
                                          text_color="white")
                method_label.pack(pady=(0, 10))
                
                col += 1
                if col >= cols:
                    col = 0
                    row += 1
                
        except Exception as e:
            print(f"Error displaying clustering analysis: {e}")
            error_label = ctk.CTkLabel(self.clustering_scrollable, 
                                     text=f"Error: {str(e)}",
                                     font=ctk.CTkFont(size=14),
                                     text_color=["#BF616A", "#A54A5A"])
            error_label.pack(pady=20)
            
    def display_enhanced_results(self, llama_results):
        """Display enhanced LLaMA analysis results (consensus) with proper text handling"""
        try:
            average_result = llama_results.get('average_result', {})
            
            # Handle case where average_result might be a string
            if isinstance(average_result, str):
                average_result = {
                    "reasoning": average_result,
                    "top_materials": [],
                    "clustering_summary": "Results provided as string"
                }
            
            if not isinstance(average_result, dict):
                average_result = {
                    "reasoning": str(average_result),
                    "top_materials": [],
                    "clustering_summary": "Unknown format"
                }
            
            # Clear existing results from tree
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
                
            # Display top materials
            top_materials = average_result.get('top_materials', [])
            if top_materials:
                for material in top_materials:
                    if isinstance(material, dict):
                        self.results_tree.insert('', 'end', values=(
                            material.get('name', 'Unknown'),
                            f"{material.get('similarity', 0):.1f}%",
                            ', '.join(material.get('extinguishers', ['Unknown'])),
                            f"{material.get('clustering_consensus', 0):.1f}%"
                        ))
            else:
                self.results_tree.insert('', 'end', values=(
                    'No materials identified', '0.0%', 'Unknown', '0.0%'
                ))
            
            # Clear and display reasoning
            self.reasoning_text.delete("1.0", "end")
            reasoning = average_result.get('reasoning', 'No reasoning available')
            if reasoning and str(reasoning).strip():
                self.reasoning_text.insert("1.0", str(reasoning))
            else:
                self.reasoning_text.insert("1.0", "No AI consensus reasoning provided.")
            
            # Clear and display clustering summary
            self.clustering_summary_text.delete("1.0", "end")
            clustering_summary = average_result.get('clustering_summary', 'No clustering summary available')
            if clustering_summary and str(clustering_summary).strip():
                self.clustering_summary_text.insert("1.0", str(clustering_summary))
            else:
                self.clustering_summary_text.insert("1.0", "No clustering summary provided.")
            
        except Exception as e:
            print(f"Error displaying enhanced results: {e}")
            # Clear and show error in text fields
            try:
                self.reasoning_text.delete("1.0", "end")
                self.reasoning_text.insert("1.0", f"Error displaying results: {str(e)}")
                
                self.clustering_summary_text.delete("1.0", "end")
                self.clustering_summary_text.insert("1.0", "Error displaying clustering summary")
            except:
                pass
            
            # Clear and show error in tree
            try:
                for item in self.results_tree.get_children():
                    self.results_tree.delete(item)
                self.results_tree.insert('', 'end', values=('Error', '0.0%', 'N/A', '0.0%'))
            except:
                pass
            
    def run(self):
        """Start the enhanced GUI application"""
        self.root.mainloop()

# Main execution
if __name__ == "__main__":
    print("Starting Enhanced Flame Material Identification Pipeline with Individual Clustering Analysis...")
    print("Looking for required files: OBJ_best.pt, SEG_best.pt, flame_dataset.json")
    print("Using core clustering methods: VB-GMM, GMM, K-Means++, DBSCAN, Mean-Shift, Agglomerative")
    print("Individual clustering analysis: Each method analyzed separately with LLaMA")
    print("Web server functionality: Flask-based REST API with enhanced web interface")
    print("Ngrok integration: Online hosting with QR code generation")
    
    # Check if required packages are available
    try:
        import qrcode
        import subprocess
        print("QR code and Ngrok functionality: Available")
    except ImportError as e:
        print(f"Warning: Some packages not found - {e}")
        print("Install with: pip install qrcode[pil]")
    
    app = FlameAnalyzerGUI()
    app.run()