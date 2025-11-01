import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.genetic_selector import GeneticFeatureSelector

def evaluate_estimator_on_features(estimator, X, y, feature_idx):
    if len(feature_idx) == 0:
        return {'accuracy': 0.0, 'f1': 0.0, 'n_features': 0}
    X_sel = X[:, feature_idx]
    scores = cross_val_score(estimator, X_sel, y, cv=5, scoring='accuracy', n_jobs=-1)
    f1_scores = cross_val_score(estimator, X_sel, y, cv=5, scoring='f1_macro', n_jobs=-1)
    return {'accuracy': float(scores.mean()), 'f1': float(f1_scores.mean()), 'n_features': int(len(feature_idx))}

def run_all(X, y, k_features=None, random_state=42, verbose=False):
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    n_features = X.shape[1]
    if k_features is None:
        k_features = max(1, n_features // 4)

    results = {}

    # GA
    ga = GeneticFeatureSelector(estimator=clf, population_size=30, n_gen=25, cv=3, random_state=random_state)
    ga.fit(X, y, verbose=verbose)
    ga_idx = ga.selected_features_.tolist()
    results['GA'] = evaluate_estimator_on_features(clf, X, y, ga_idx)

    # Lasso
    try:
        las = LassoCV(cv=5, random_state=random_state, n_jobs=-1).fit(X, y)
        lasso_idx = [i for i, coef in enumerate(las.coef_) if abs(coef) > 1e-5]
    except Exception:
        lasso_idx = []
    results['Lasso'] = evaluate_estimator_on_features(clf, X, y, lasso_idx)

    # Chi2
    try:
        skb_chi = SelectKBest(score_func=chi2, k=min(k_features, n_features)).fit(X, y)
        chi_idx = list(np.where(skb_chi.get_support())[0])
    except Exception:
        chi_idx = []
    results['Chi2'] = evaluate_estimator_on_features(clf, X, y, chi_idx)

    # ANOVA (f_classif)
    try:
        skb_f = SelectKBest(score_func=f_classif, k=min(k_features, n_features)).fit(X, y)
        f_idx = list(np.where(skb_f.get_support())[0])
    except Exception:
        f_idx = []
    results['ANOVA'] = evaluate_estimator_on_features(clf, X, y, f_idx)

    # PCA
    try:
        pca = PCA(n_components=min(k_features, n_features), random_state=random_state)
        X_pca = pca.fit_transform(X)
        results['PCA'] = evaluate_estimator_on_features(clf, X_pca, y, list(range(X_pca.shape[1])))
    except Exception:
        results['PCA'] = {'accuracy': 0.0, 'f1': 0.0, 'n_features': 0}

    return results

if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data
    y = data.target
    res = run_all(X, y, k_features=10, verbose=True)
    import json
    print(json.dumps(res, indent=2))
