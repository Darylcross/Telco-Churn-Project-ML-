from sklearn.model_selection import cross_validate
import numpy as np

def evaluate_model_performance(model, X, y, cv=10):
    """
    Modelin performansını değerlendiren fonksiyon.
    
    Parameters:
    - model: Değerlendirilecek model (örneğin RandomForestClassifier)
    - X: Özellikler (features)
    - y: Hedef değişken (target)
    - cv: Çapraz doğrulama katman sayısı (default=10)
    
    Returns:
    - Ortalama accuracy, F1 skoru ve ROC AUC değerlerini döner.
    """
    cv_results = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    
    avg_accuracy = np.mean(cv_results["test_accuracy"])
    avg_f1 = np.mean(cv_results["test_f1"])
    avg_roc_auc = np.mean(cv_results["test_roc_auc"])
    
    print(f"Ortalama Accuracy: {avg_accuracy:.4f}")
    print(f"Ortalama F1 Skoru: {avg_f1:.4f}")
    print(f"Ortalama ROC AUC: {avg_roc_auc:.4f}")
    
    return avg_accuracy, avg_f1, avg_roc_auc
