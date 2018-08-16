# Train, tune hyper-parameters, publish model, and predict.

## Default values
```
JOB_NAME = ml_job$(date +%Y%m%d%H%M%S)
JOB_FOLDER = MLEngine/${JOB_NAME}
BUCKET_NAME = saltdetection
MODEL_PATH = $(gsutil ls gs://${BUCKET_NAME}/${JOB_FOLDER}/export/estimator/ | tail -1)
MODEL_NAME = prediction_model
MODEL_VERSION = version_1
TEST_DATA = data/csv/DataTest.csv
PREDICTIONS_FOLDER = ${JOB_FOLDER}/test_predictions
```

## Train model
```
gcloud ml-engine jobs submit training $JOB_NAME \
        --job-dir=gs://${BUCKET_NAME}/${JOB_FOLDER} \
        --runtime-version=1.8 \
        --region=us-central1 \
        --scale-tier=PREMIUM_1 \
        --module-name=trainer.task \
        --package-path=trainer
```

## Hyper-parameter tuning
```
gcloud ml-engine jobs submit training ${JOB_NAME} \
        --job-dir=gs://${BUCKET_NAME}/${JOB_FOLDER} \
        --runtime-version=1.8 \
        --region=us-central1 \
        --module-name=trainer.task \
        --package-path=trainer \
        --config=config.yaml
```

## Create model
```
gcloud ml-engine models create ${MODEL_NAME} --regions=us-central1
```

## Create model version
```
gcloud ml-engine versions create ${MODEL_VERSION} \
        --model=${MODEL_NAME} \
        --origin=${MODEL_PATH} \
        --runtime-version=1.8
```

## Predict on test data
```
gcloud ml-engine jobs submit prediction ${JOB_NAME} \
    --model=${MODEL_NAME} \
    --input-paths=gs://${BUCKET_NAME}/${TEST_DATA} \
    --output-path=gs://${BUCKET_NAME}/${PREDICTIONS_FOLDER} \
    --region=us-central1 \
    --data-format=TEXT \
    --version=${MODEL_VERSION}
```