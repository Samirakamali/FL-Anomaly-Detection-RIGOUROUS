#------------------ Request Handler For EaaS API -------------------------
#-------------------------------------------------------------------------


import requests
from phe.paillier import PaillierPublicKey, EncryptedNumber
import time

url = 'http://0.0.0.0:0'
token = 'token'

MAX_RETRIES = 5
RETRY_DELAY = 2  
TIMEOUT = 20  


def retry_request(func, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            result = func()
            if result != 'ERROR':
                return result
        except requests.Timeout:
            print(f"Request timed out. Attempt {attempt+1}/{max_retries}. Retrying...")
        time.sleep(RETRY_DELAY)
    print("Max retries reached. Exiting...")
    return 'ERROR'


def encrypt(request_id, algorithm, text):
    global url
    complete_url = url + '/encrypt/'

    def encrypt_single_value(t):
        jobject = {'id': str(request_id), 'text': str(t), 'algo': str(algorithm)}
        try:
            response = requests.post(complete_url, json=jobject, headers={'Authorization': f'Bearer {token}'}, timeout=TIMEOUT)
            response_data = response.json()
            if algorithm == 'paillier' and 'cipher' in response_data:
                key = PaillierPublicKey(response_data['cipher']['key']['n'])
                cipher = EncryptedNumber(key, response_data['cipher']['cypher'], response_data['cipher']['exponent'])
            else:
                cipher = str(response_data.get('cipher', 'ERROR'))
        except requests.Timeout:
            print(f"Request timed out for value {t}")
            cipher = 'ERROR'
        except Exception as e:
            print(f"Error during encryption for value {t}: {e}")
            cipher = 'ERROR'
        return cipher

    
    if isinstance(text, list):
        encrypted_list = []
        previous_encrypted_value = None  

        for idx, t in enumerate(text):  
            encrypted_value = retry_request(lambda: encrypt_single_value(t))
            if encrypted_value == 'ERROR':
                if previous_encrypted_value is not None:
                    print(f"Encryption failed at index {idx} for value {t}, replacing with previous successful encrypted value.")
                    encrypted_value = previous_encrypted_value
                else:
                    print(f"Encryption failed at index {idx} for value {t}, no previous value available.")
                    encrypted_value = retry_request(lambda: encrypt_single_value(t))  
            else:
                previous_encrypted_value = encrypted_value  

            encrypted_list.append(encrypted_value)
        return encrypted_list
    
   
    return retry_request(lambda: encrypt_single_value(text))



def decrypt(request_id, algorithm, ciphertext):
    global url
    complete_url = url + '/decrypt/'
    default_value = 0.0 

    def decrypt_single_value(cipher):
        if algorithm == 'paillier':
            
            if isinstance(cipher, EncryptedNumber):
                exponent = str(cipher.exponent)
                cipher_value = str(cipher.ciphertext())
                cipher_combined = exponent + ' ' + cipher_value
            else:
                print(f"Invalid cipher format: {cipher}")
                return default_value  
        else:
            cipher_combined = str(cipher)

        jobject = {'id': str(request_id), 'cipher': cipher_combined, 'algo': str(algorithm)}
        try:
            response = requests.post(complete_url, json=jobject, headers={'Authorization': f'Bearer {token}'}, timeout=TIMEOUT)
            plain = response.json().get('cleartext', default_value)  
            plain = float(plain) if plain != 'ERROR' else default_value  
            print(f"Decrypted value: {plain}")
        except requests.Timeout:
            print(f"Decryption request timed out for cipher {cipher}")
            plain = default_value  
        except Exception as e:
            print(f"Error during decryption for cipher {cipher}: {e}")
            plain = default_value  
        return plain

   
    if isinstance(ciphertext, list):
        decrypted_list = []
        for cipher in ciphertext:
            decrypted_value = retry_request(lambda: decrypt_single_value(cipher))
            decrypted_list.append(decrypted_value)
        return decrypted_list

   
    return retry_request(lambda: decrypt_single_value(ciphertext))



