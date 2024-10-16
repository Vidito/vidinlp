import joblib

# Load the pipeline to check its type
pipeline = joblib.load('vidinlp/best_sentiment_pipeline_calibrated.joblib')
print(type(pipeline))  # Should be something like 'sklearn.pipeline.Pipeline'
