import os
import numpy as np
from pulse_extractor import extract_multi_roi_signals, calculate_coherence_features
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

def train_and_evaluate_classifier(X, y, groups):
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    gkf = GroupKFold(n_splits=5)
    accuracies = []
    
    clf = SVC(kernel='linear', probability=True)
    
    # Needs at least 5 groups for n_splits=5 to work
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        
        acc = accuracy_score(y_test, predictions)
        accuracies.append(acc)
        
    print(f"Average Model Accuracy: {np.mean(accuracies) * 100:.2f}%")
    return clf

def main():
    dataset_dir = "dataset"
    categories = {"real_videos": 1, "fake_videos": 0}
    
    X_data, y_labels, group_ids = [], [], []
    subject_id_counter = 1
    
    for category, label in categories.items():
        folder_path = os.path.join(dataset_dir, category)
        if not os.path.exists(folder_path):
            print(f"Error: You must create the folder {folder_path} and put videos in it.")
            return
            
        for filename in os.listdir(folder_path):
            if filename.endswith(".mp4"):
                video_path = os.path.join(folder_path, filename)
                print(f"Processing: {video_path}")
                
                try:
                    signals, fps = extract_multi_roi_signals(video_path)
                    features = calculate_coherence_features(signals, fps)
                    
                    X_data.append(features)
                    y_labels.append(label)
                    # Note: If multiple videos feature the EXACT SAME person, 
                    # you must hardcode them to share the same group ID here.
                    group_ids.append(f"subject_{subject_id_counter}")
                    subject_id_counter += 1
                except Exception as e:
                    print(f"Failed to process {filename}. Error: {e}")

    # The GroupKFold needs at least 5 subjects to split
    if len(X_data) < 5:
        print("Error: You need at least 5 properly processed videos to train and evaluate.")
        return
        
    print("\nTraining Model...")
    train_and_evaluate_classifier(X_data, y_labels, group_ids)

if __name__ == "__main__":
    main()