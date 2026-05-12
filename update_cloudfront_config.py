import json

# Try to read with utf-16, fallback to utf-8
try:
    with open('dist-config.json', 'r', encoding='utf-16') as f:
        config_data = json.load(f)
except:
    with open('dist-config.json', 'r', encoding='utf-8') as f:
        config_data = json.load(f)

dist_config = config_data['DistributionConfig']

# 1. Add ALB Origin
alb_origin = {
    "Id": "ALB-mpd-backend",
    "DomainName": "mpd-backend-alb-1607763580.ap-south-1.elb.amazonaws.com",
    "OriginPath": "",
    "CustomHeaders": {"Quantity": 0},
    "CustomOriginConfig": {
        "HTTPPort": 80,
        "HTTPSPort": 443,
        "OriginProtocolPolicy": "http-only",
        "OriginSslProtocols": {
            "Quantity": 3,
            "Items": ["TLSv1", "TLSv1.1", "TLSv1.2"]
        },
        "OriginReadTimeout": 30,
        "OriginKeepaliveTimeout": 5
    },
    "ConnectionAttempts": 3,
    "ConnectionTimeout": 10,
    "OriginShield": {"Enabled": False},
    "OriginAccessControlId": ""
}

dist_config['Origins']['Items'].append(alb_origin)
dist_config['Origins']['Quantity'] = len(dist_config['Origins']['Items'])

# 2. Add Cache Behavior for /api/*
api_behavior = {
    "PathPattern": "api/*",
    "TargetOriginId": "ALB-mpd-backend",
    "TrustedSigners": {"Enabled": False, "Quantity": 0},
    "TrustedKeyGroups": {"Enabled": False, "Quantity": 0},
    "ViewerProtocolPolicy": "https-only",
    "AllowedMethods": {
        "Quantity": 7,
        "Items": ["GET", "HEAD", "POST", "PUT", "PATCH", "OPTIONS", "DELETE"],
        "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]}
    },
    "SmoothStreaming": False,
    "Compress": True,
    "LambdaFunctionAssociations": {"Quantity": 0},
    "FunctionAssociations": {"Quantity": 0},
    "FieldLevelEncryptionId": "",
    "ForwardedValues": {
        "QueryString": True,
        "Cookies": {"Forward": "all"},
        "Headers": {
            "Quantity": 3,
            "Items": ["Origin", "Access-Control-Request-Method", "Access-Control-Request-Headers"]
        },
        "QueryStringCacheKeys": {"Quantity": 0}
    },
    "MinTTL": 0,
    "DefaultTTL": 0,
    "MaxTTL": 0
}

# Fix if CacheBehaviors is missing Items
if 'Items' not in dist_config['CacheBehaviors']:
    dist_config['CacheBehaviors']['Items'] = []

dist_config['CacheBehaviors']['Items'].append(api_behavior)
dist_config['CacheBehaviors']['Quantity'] = len(dist_config['CacheBehaviors']['Items'])

# Save modified config (only the DistributionConfig part)
with open('modified-config.json', 'w', encoding='utf-8') as f:
    json.dump(dist_config, f, indent=4)

print(f"Modified config saved. ETag: {config_data['ETag']}")
