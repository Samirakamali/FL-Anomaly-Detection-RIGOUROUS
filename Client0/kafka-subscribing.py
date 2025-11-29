from kafka import KafkaConsumer
import logging

kafka_address = "kafka_address"
kafka_topic = "test_kafka"
security_protocol = 'SASL_SSL'
ssl_cafile = "/home/ubuntu/API_integration_device/Client0/ca-cert.pem"
sasl_mechanism = 'PLAIN'
sasl_plain_username = 'username'
sasl_plain_password = 'password'

consumer = KafkaConsumer(
    kafka_topic,
    bootstrap_servers=kafka_address,
    security_protocol=security_protocol,
    ssl_cafile=ssl_cafile,
    sasl_mechanism=sasl_mechanism,
    sasl_plain_username=sasl_plain_username,
    sasl_plain_password=sasl_plain_password,
    auto_offset_reset='earliest',
    group_id='RIGOUROUS',
    enable_auto_commit=True
)

print(f" Listening to topic: {kafka_topic}")
for message in consumer:
    print("\n🟩 Received message:")
    print(message.value.decode('utf-8'))
