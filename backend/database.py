import os
import json
import boto3
from peewee import *

# 1. Fetch DB Credentials from AWS Secrets Manager if available
def get_db_credentials():
    secret_name = "mpd/backend/db-credentials-v2"
    region_name = "ap-south-1"
    
    # Check for local DATABASE_URL first
    if "DATABASE_URL" in os.environ:
        return os.environ["DATABASE_URL"]

    try:
        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=region_name)
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(get_secret_value_response['SecretString'])
        
        # We need the endpoint of the RDS instance, we can pass it via ENV or hardcode for now
        host = os.environ.get("DB_HOST", "localhost")
        return f"postgresql://{secret['username']}:{secret['password']}@{host}:5432/postgres"
    except Exception as e:
        print(f"Warning: Could not fetch secrets from AWS. Using local SQLite fallback. Error: {e}")
        db_path = "/tmp/local_dev.db" if os.environ.get("VERCEL") else "local_dev.db"
        return f"sqlite:///{db_path}"

db_url = get_db_credentials()

if db_url.startswith("postgres"):
    from playhouse.db_url import connect
    db = connect(db_url)
else:
    db_path = "/tmp/local_dev.db" if os.environ.get("VERCEL") else "local_dev.db"
    db = SqliteDatabase(db_path)

class BaseModel(Model):
    class Meta:
        database = db

class User(BaseModel):
    username = CharField(unique=True)
    email = CharField(unique=True)

class Holding(BaseModel):
    user = ForeignKeyField(User, backref='holdings')
    ticker = CharField()
    quantity = FloatField()
    buyPrice = FloatField()

def init_db():
    db.connect()
    db.create_tables([User, Holding], safe=True)
    print("✅ Database initialized successfully.")
