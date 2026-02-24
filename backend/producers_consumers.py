import joblib
from confluent_kafka import Producer, Consumer
import json
import time

class DataProducer:
    def __init__(self, bootstrap_servers="kafka:9092"):
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'batch.num.messages': '10000',
            'linger.ms': '100'
        })
    
    def send_data(self, topic, data):
        if 'user_id' not in data:
            data['user_id'] = f"user_{int(time.time())}"
        self.producer.produce(topic, value=json.dumps(data))
        return True
    
    def flush(self):
        self.producer.flush()

class RiskClassifierConsumer:
    def __init__(self, bootstrap_servers="kafka:9092", model_path="/app/model/model_Logistic_Regression.joblib"):
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': 'risk_group',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True
        })
        self.model = joblib.load(model_path)
    
    def process_message(self, message):
        if not message:
            return None
        
        try:
            data = json.loads(message)
            user_id = data.pop('user_id', f"unknown_{int(time.time())}")
            
            features = [float(data.get(feature, 0.0)) for feature in self.model['feature_names']]
            
            prediction = self.model['model'].predict([features])[0]
            probability = self.model['model'].predict_proba([features])[0][prediction]
            
            return {
                'user_id': user_id,
                'risk_level': 'High' if prediction == 1 else 'Low',
                'confidence': float(probability),
                'prediction': int(prediction),
                'raw_data': data
            }
        except:
            return None
