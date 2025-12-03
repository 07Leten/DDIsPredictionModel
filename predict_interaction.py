import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import json
import os
import sys
import re

# Configuration
MODEL_PATH = 'model_epoch_30.pt' # Default to epoch 30
SCALER_PATH = 'scaler_SMOTE.pkl'
DRUG_LIST_PATH = 'drug_list_all.pkl'
GENE_LIST_PATH = 'use_gene_list_all.pkl'
WALK_RESULTS_PATH = 'save_walk_results_drug_general'
MESH_INFO_PATH = 'mesh_info.csv'

# Model Definition (Must match training)
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.bn1 = nn.BatchNorm1d(hidden_size_1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.bn2 = nn.BatchNorm1d(hidden_size_2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.bn3 = nn.BatchNorm1d(hidden_size_3)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(hidden_size_3, hidden_size_4)
        self.bn4 = nn.BatchNorm1d(hidden_size_4)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(hidden_size_4, num_classes)

    def forward(self, x):
        x = self.dropout1(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.dropout4(F.leaky_relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x

class DDI_Predictor:
    def __init__(self, model_path=MODEL_PATH, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.load_resources()
        self.load_model(model_path)

    def load_resources(self):
        print("Loading metadata...")
        with open(DRUG_LIST_PATH, 'rb') as f:
            self.drug_list = pickle.load(f)
        with open(GENE_LIST_PATH, 'rb') as f:
            self.gene_list = pickle.load(f)
            
        print("Loading scaler...")
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)

        print("Loading drug embeddings (this may take a moment)...")
        self.drug_embeddings = {}
        # Initialize matrix
        matrix = np.zeros((len(self.drug_list), len(self.gene_list)), dtype=np.float32)
        
        if os.path.isdir(WALK_RESULTS_PATH):
            print(f"Loading walk results from directory: {WALK_RESULTS_PATH}")
            files = [f for f in os.listdir(WALK_RESULTS_PATH) if f.endswith('.pkl')]
            
            def get_batch_num(filename):
                match = re.search(r'batch_(\d+)', filename)
                return int(match.group(1)) if match else -1
            
            files.sort(key=get_batch_num)
            
            current_idx = 0
            for filename in files:
                file_path = os.path.join(WALK_RESULTS_PATH, filename)
                with open(file_path, 'rb') as f:
                    batch_results = pickle.load(f)
                    for df in batch_results:
                        if current_idx < len(matrix):
                            vals = df['value'].fillna(0).values
                            matrix[current_idx] = vals
                            current_idx += 1
            print(f"Loaded {current_idx} drug embeddings.")
            
        elif os.path.exists(WALK_RESULTS_PATH):
            with open(WALK_RESULTS_PATH, 'rb') as f:
                walk_results = pickle.load(f)
                for i, df in enumerate(walk_results):
                    vals = df['value'].fillna(0).values
                    matrix[i] = vals
            del walk_results
        else:
            raise FileNotFoundError(f"Could not find {WALK_RESULTS_PATH}")
            
        # Create a quick lookup dict
        self.drug_emb_matrix = matrix
        self.drug_to_idx = {drug: i for i, drug in enumerate(self.drug_list)}

        print("Loading disease embeddings...")
        mesh = pd.read_csv(MESH_INFO_PATH)
        mesh['mesh_embedding'] = mesh['mesh_embedding'].apply(json.loads)
        self.disease_embeddings = {row['mesh']: np.array(row['mesh_embedding'], dtype=np.float32) for _, row in mesh.iterrows()}
        
        self.drug_dim = matrix.shape[1]
        self.disease_dim = len(mesh['mesh_embedding'][0])
        self.input_dim = self.drug_dim + self.disease_dim
        print(f"Input dimension: {self.input_dim}")

    def load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        # Hyperparameters from training script
        hidden_size_1 = 4096
        hidden_size_2 = 1024
        hidden_size_3 = 256
        hidden_size_4 = 64
        num_classes = 1
        
        self.model = SimpleNN(self.input_dim, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, num_classes)
        
        # Load state dict
        # Handle case where model was saved on GPU but loading on CPU
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, drug_name, disease_name):
        if drug_name not in self.drug_to_idx:
            return f"Error: Drug '{drug_name}' not found in database."
        if disease_name not in self.disease_embeddings:
            return f"Error: Disease '{disease_name}' not found in database."

        # Prepare input
        drug_idx = self.drug_to_idx[drug_name]
        drug_emb = self.drug_emb_matrix[drug_idx]
        disease_emb = self.disease_embeddings[disease_name]
        
        # Concatenate
        feature_vector = np.concatenate([drug_emb, disease_emb])
        
        # Scale
        # Scaler expects 2D array
        feature_vector = feature_vector.reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Convert to tensor
        input_tensor = torch.tensor(feature_vector_scaled, dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).item()
            
        return probability

if __name__ == "__main__":
    # Example usage
    predictor = DDI_Predictor()
    
    print("\n--- Prediction System Ready ---")
    print("Enter a drug name and a disease name (MESH ID) to predict interaction probability.")
    print("Type 'exit' to quit.")
    
    while True:
        try:
            drug = input("\nDrug Name (e.g., '10-nitro-oleic acid'): ").strip()
            if drug.lower() == 'exit':
                break
            
            disease = input("Disease Name (e.g., 'Piper'): ").strip()
            if disease.lower() == 'exit':
                break
                
            result = predictor.predict(drug, disease)
            
            if isinstance(result, float):
                print(f"Interaction Probability: {result:.4f}")
                if result > 0.5:
                    print("Prediction: Positive Interaction")
                else:
                    print("Prediction: No Interaction")
            else:
                print(result)
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")
