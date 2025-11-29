from kafka import KafkaProducer
import json
import logging

def send_stix_files():
    kafka_address = "kafka_address"
    kafka_topic = "test_kafka"
    security_protocol = 'SASL_SSL'
    ssl_cafile = "/home/ubuntu/API_integration_device/Client0/ca-cert.pem" 
    sasl_mechanism = 'PLAIN'
    sasl_plain_username = 'username'
    sasl_plain_password = 'password'

    
    producer = KafkaProducer(
        bootstrap_servers=kafka_address,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda v: json.dumps(v).encode('utf-8'),
        ssl_cafile=ssl_cafile,
        ssl_check_hostname=False,
        security_protocol=security_protocol,
        sasl_mechanism=sasl_mechanism,
        sasl_plain_username=sasl_plain_username,
        sasl_plain_password=sasl_plain_password
    )

    
    for filename in ["combined_output_bundle.stix"]:
        with open(filename, "r") as f:
            stix_data = f.read() 
            producer.send(kafka_topic, key="stix", value=stix_data)
            print(f"[✓] Sent: {filename}")

    producer.flush()
    producer.close()
