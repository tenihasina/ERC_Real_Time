import time
import boto3
from botocore.config import Config


def transcribe_file(job_name, file_uri, transcribe_client):
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': file_uri},
        MediaFormat='wav',
        LanguageCode='fr-FR'
    )

    max_tries = 60
    while max_tries > 0:
        max_tries -= 1
        job = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job_status = job['TranscriptionJob']['TranscriptionJobStatus']
        if job_status in ['COMPLETED', 'FAILED']:
            print(f"Job {job_name} is {job_status}.")
            if job_status == 'COMPLETED':
                print(
                    f"Download the transcript from\n"
                    f"\t{job['TranscriptionJob']['Transcript']['TranscriptFileUri']}.")
            break
        else:
            print(f"Waiting for {job_name}. Current status is {job_status}.")
        time.sleep(10)


ACCESS_KEY = "AKIAT7BLGMLMS5RLKUE2"
SECRET_KEY = "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"
my_config = Config(
    region_name='eu-west-3',
    signature_version='v4',
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)


def main():
    transcribe_client = boto3.client('transcribe', aws_access_key_id=ACCESS_KEY,
                                     aws_secret_access_key=SECRET_KEY,
                                     config=my_config
                                     )
    list_uri = []
    for i in range(2, 39):
        list_uri.append(f"s3://rasensanspeechbucket/Recording_{i}.wav")
    for file_uri in list_uri:
        id_audio = file_uri.split("/")[-1].split(".")[0]
        transcribe_file(f"transcript_{id_audio}", file_uri, transcribe_client)


if __name__ == '__main__':
    main()
