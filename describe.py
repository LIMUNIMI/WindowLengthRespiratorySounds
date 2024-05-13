import pickle
# script per la lettura degli ensemble generati

with open('Models_Ensembles/half_hour_0.6', 'rb') as m:
    ensemble = pickle.load(m)

models = ensemble.get_models_with_weights()
for i, (w, model) in enumerate(models):
            print(f"Model number {i}")
            print(f"Weight: {w}")
            print(model.config)
