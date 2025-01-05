import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, auc


# Εισαγωγή Δεδομένων
file_path = 'data.csv'
data = pd.read_csv(file_path)

# Επισκόπιση των Δεδομένων , Εμφάνιση τίτλων
data.head(), data.info()

# Διαίρεση σε χαρακτηριστικά και κατηγορίες
X = data.iloc[:, :-1].values #[:, :-1] όλες οι στήλες εκτός της τελευταίας
y = data.iloc[:, -1].values # Τελευταία στήλη

# Μορφή δεδομένων
un_class = np.unique(y)
split_data = {}

# Χωρισμός σε εκπαίδευση, επικύρωση και δοκιμή 
for cls in un_class:
    cls_indices = np.where(y == cls)[0]
    cls_X, cls_y = X[cls_indices], y[cls_indices]
    X_train, X_temp, y_train, y_temp = train_test_split(cls_X, cls_y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    split_data[cls] = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }

# Συνένωση των δεδομένων 
X_train = np.vstack([split_data[cls]["train"][0] for cls in un_class])
y_train = np.hstack([split_data[cls]["train"][1] for cls in un_class])

X_val = np.vstack([split_data[cls]["val"][0] for cls in un_class])
y_val = np.hstack([split_data[cls]["val"][1] for cls in un_class])

X_test = np.vstack([split_data[cls]["test"][0] for cls in un_class])
y_test = np.hstack([split_data[cls]["test"][1] for cls in un_class])

# Κανονικοποίηση 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=2) # 2 components
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

# Parzen 
def parzen_gauss_class(X_train, y_train, X_test, bandwidth):
    models_kde = {}
    for cls in un_class:
        cls_indices = np.where(y_train == cls)[0] 
        kde = gaussian_kde(X_train[cls_indices].T, bw_method=bandwidth)
        models_kde[cls] = kde
        
    predictions = []
    for x in X_test:
        scores = {cls: models_kde[cls](x)[0] for cls in un_class}
        predictions.append(max(scores, key=scores.get))
    return np.array(predictions)

bandwidths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
results = {}

for bw in bandwidths:
    y_pred = parzen_gauss_class(X_train_pca, y_train, X_test_pca, bandwidth=bw)
    accuracy = np.mean(y_pred == y_test)
    results[bw] = accuracy

# Εμφάνιση 
print("Sorting results for different window sizes")
for bw, acc in results.items():
    print(f"Bandwidth {bw}: Accuracy = {acc:.2f}")

# Ερώτημα 2: Εφαρμογή k-NN για διαφορετικές τιμές του k
k_values = [1, 3, 5, 7, 9]
knn_results = {}

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca, y_train)
    y_pred_knn = knn.predict(X_test_pca)
    accuracy_knn = np.mean(y_pred_knn == y_test)
    knn_results[k] = accuracy_knn

print("\nSorting results for k-NN")
for k, acc in knn_results.items():
    print(f"k = {k}: Accuracy = {acc:.2f}")

# Ερώτημα 3: Γραμμικό και μη γραμμικό SVM
svm_results = {}

# Γραμμικό SVM
linear_svm = SVC(kernel='linear', C=1.0)
linear_svm.fit(X_train_pca, y_train)
y_pred_linear_svm = linear_svm.predict(X_test_pca)
accuracy_linear_svm = np.mean(y_pred_linear_svm == y_test)
svm_results['linear'] = accuracy_linear_svm

# Μη γραμμικό SVM με RBF kernel
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
rbf_svm.fit(X_train_pca, y_train)
y_pred_rbf_svm = rbf_svm.predict(X_test_pca)
accuracy_rbf_svm = np.mean(y_pred_rbf_svm == y_test)
svm_results['rbf'] = accuracy_rbf_svm

print("\nSorting results for SVM")
print(f"Linear SVM: Accuracy = {accuracy_linear_svm:.2f}")
print(f"RBF SVM: Accuracy = {accuracy_rbf_svm:.2f}")




# Εξαγωγή των θετικών δειγμάτων
positive_class = 1 
positive_samples = X_train_pca[y_train == positive_class]

# Παράμετροι
components = [1, 2, 3] 
thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

# Αποτελέσματα
roc_data = {}

for n_components in components:
    # Εκπαίδευση GMM
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(positive_samples)

    # Υπολογισμός πιθανοτήτων 
    scores = gmm.score_samples(X_test_pca)
    
    # Υπολογισμός ROC για δ thresholds
    fpr, tpr, _ = roc_curve(y_test == positive_class, scores)
    roc_auc = auc(fpr, tpr)
    roc_data[n_components] = (fpr, tpr, roc_auc)

    print(f"GMM with {n_components} components: AUC = {roc_auc:.2f}")

# Καμπύλες ROC
plt.figure(figsize=(10, 6))
for n_components, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{n_components} components (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.title('ROC Curve for GMM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()
