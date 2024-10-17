import json
import logging
import yaml
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def get_secret(config: dict):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        aws_access_key_id=config['aws_access_key_id'],
        aws_secret_access_key=config['aws_secret_access_key'],
        region_name=config['aws_region']
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=config['aws_secret_name']
        )

        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
    
        else:
            secret = get_secret_value_response['SecretBinary']
            
        if secret:
            secret = json.loads(secret)
            return secret
        
    except ClientError as err:
        logger.error(f"Error while retrieving values from Secrets Manager: {err}")

if __name__ == '__main__':
    with open(Path('config/config.yaml').resolve()) as f:
        config = yaml.safe_load(f)

    secret = get_secret()
    logger.info(f"Secret Keys: {secret}")