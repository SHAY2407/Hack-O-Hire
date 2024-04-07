import json
import boto3

def main():
    AWS_ACCESS_KEY = ""  # Replace with your access key
    AWS_SECRET_KEY = ""  # Replace with your secret key
    AWS_REGION_NAME = ""

    # Read data from fin_trans.json
    with open('fin_trans.json', 'r') as json_file:
        data = json.load(json_file)

    # Send each record to Kinesis Firehose
    client = boto3.client('firehose',
                          aws_access_key_id=AWS_ACCESS_KEY,
                          aws_secret_access_key=AWS_SECRET_KEY,
                          region_name=AWS_REGION_NAME)

    for record in data:
        response = client.put_record(
            DeliveryStreamName='anomaly_detection_stream',
            Record={
                'Data': json.dumps(record)
            }
        )
        print(response)

if __name__ == '__main__':
    main()
