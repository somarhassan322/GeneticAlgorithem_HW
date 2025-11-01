import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import random

class GeneticFeatureSelector:
    """Implementation of a simple Genetic Algorithm for feature selection.
    - Uses RandomForest (or a provided estimator) and cross_val_score as fitness.
    - Chromosome: binary mask indicating selected features.
    """
    def __init__(self, estimator=None, population_size=30, n_gen=40,
                 crossover_rate=0.8, mutation_rate=0.02, cv=3, random_state=42):
        self.estimator = estimator if estimator is not None else RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.pop_size = population_size
        self.n_gen = n_gen
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.cv = cv
        self.rs = random_state
        random.seed(self.rs)
        np.random.seed(self.rs)

    def _init_population(self, n_features):
        pop = []
        for _ in range(self.pop_size):
            chrom = np.random.choice([0,1], size=n_features, p=[0.7,0.3]).astype(int)
            if chrom.sum() == 0:
                chrom[np.random.randint(0,n_features)] = 1
            pop.append(chrom)
        return pop

    def _fitness(self, chrom, X, y):
        sel_idx = np.where(chrom==1)[0]
        if len(sel_idx) == 0:
            return 0.0
        X_sel = X[:, sel_idx]
        clf = clone(self.estimator)
        scores = cross_val_score(clf, X_sel, y, cv=self.cv, scoring='accuracy', n_jobs=1)
        return scores.mean()

    def _tournament_selection(self, population, fitnesses, k=3):
        selected = []
        for _ in range(len(population)):
            aspirants = random.sample(range(len(population)), k)
            best = max(aspirants, key=lambda i: fitnesses[i])
            selected.append(population[best].copy())
        return selected

    def _crossover(self, p1, p2):
        if random.random() > self.cx_rate or len(p1) < 2:
            return p1.copy(), p2.copy()
        point = random.randint(1, len(p1)-1)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2

    def _mutate(self, chrom):
        for i in range(len(chrom)):
            if random.random() < self.mut_rate:
                chrom[i] = 1 - chrom[i]
        if chrom.sum() == 0:
            chrom[np.random.randint(0,len(chrom))] = 1
        return chrom

    def fit(self, X, y, verbose=False):
        n_features = X.shape[1]
        pop = self._init_population(n_features)
        best_chrom = None
        best_score = -1.0

        for gen in range(self.n_gen):
            fitnesses = [self._fitness(ind, X, y) for ind in pop]
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_score = fitnesses[gen_best_idx]
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_chrom = pop[gen_best_idx].copy()

            if verbose:
                print(f"Gen {gen+1}/{self.n_gen} - best acc: {gen_best_score:.4f} - overall best: {best_score:.4f}")

            selected = self._tournament_selection(pop, fitnesses, k=3)

            children = []
            for i in range(0, len(selected), 2):
                p1 = selected[i]
                p2 = selected[(i+1) % len(selected)]
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                children.append(c1)
                children.append(c2)
            pop = children[:self.pop_size]

        self.best_chromosome_ = best_chrom
        self.best_score_ = best_score
        self.selected_features_ = np.where(best_chrom==1)[0]
        return self

    def transform(self, X):
        return X[:, self.selected_features_]

    def fit_transform(self, X, y, verbose=False):
        self.fit(X, y, verbose=verbose)
        return self.transform(X)
