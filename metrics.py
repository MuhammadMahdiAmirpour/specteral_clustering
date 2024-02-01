from sklearn.metrics import adjusted_rand_score

def clustering_score(true_tables, predicted_tables):
    return adjusted_rand_score(true_tables, predicted_tables)

