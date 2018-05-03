#!/bin/bash
python neural_sort.py
python neural_pruning.py >> log-retrain
python neural_retrain.py >>log-retrain

find -name 'model_init_params.py' | xargs perl -pi -e 's|prune_rate=0.1|prune_rate=0.15|g'
python neural_sort.py
python neural_pruning.py >> log-retrain
python neural_retrain.py >>log-retrain

find -name 'model_init_params.py' | xargs perl -pi -e 's|prune_rate=0.1|prune_rate=0.2|g'
python neural_sort.py
python neural_pruning.py >> log-retrain
python neural_retrain.py >>log-retrain

find -name 'model_init_params.py' | xargs perl -pi -e 's|prune_rate=0.1|prune_rate=0.25|g'
python neural_sort.py
python neural_pruning.py >> log-retrain
python neural_retrain.py >>log-retrain

find -name 'model_init_params.py' | xargs perl -pi -e 's|prune_rate=0.1|prune_rate=0.3|g'
python neural_sort.py
python neural_pruning.py >> log-retrain
python neural_retrain.py >>log-retrain

find -name 'model_init_params.py' | xargs perl -pi -e 's|prune_rate=0.1|prune_rate=0.35|g'
python neural_sort.py
python neural_pruning.py >> log-retrain
python neural_retrain.py >>log-retrain

find -name 'model_init_params.py' | xargs perl -pi -e 's|prune_rate=0.1|prune_rate=0.4|g'
python neural_sort.py
python neural_pruning.py >> log-retrain
python neural_retrain.py >>log-retrain
