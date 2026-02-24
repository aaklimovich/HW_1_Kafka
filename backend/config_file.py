bootstrap_servers = "kafka:9092"

producer_config = {
    'bootstrap.servers': bootstrap_servers
}

consumer_config = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': 'risk_group',
    'auto.offset.reset': 'earliest'
}

kafka_topic = "gaming-risk"