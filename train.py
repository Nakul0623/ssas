import boto3
import face_recognition
from sklearn import neighbors
import joblib
import os
from io import BytesIO

# Initialize AWS S3 client
s3 = boto3.client('s3', aws_access_key_id='AKIATROPCW2PVLZXKFGW', aws_secret_access_key='f899hKgOG+04qROasgD+QDc6ttITD5Bpq8ERRrY7')

# Replace with your S3 bucket name
bucket_name = 'studentattendencesystem'

# Directory where images are stored in the S3 bucket
s3_images_dir = 'images/'

def load_and_extract_embeddings(s3_bucket, folder):
    images = []
    labels = []

    # List objects in the S3 bucket
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=folder)
    for obj in response.get('Contents', []):
        if obj['Key'].endswith('.png'):
            image_data = s3.get_object(Bucket=s3_bucket, Key=obj['Key'])['Body'].read()
            image = face_recognition.load_image_file(BytesIO(image_data))
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                embedding = face_encodings[0]
                images.append(embedding)
                labels.append(os.path.basename(obj['Key']).split('_')[0])
            else:
                print(f"No face detected in the image: {obj['Key']}")

    return images, labels

images, labels = load_and_extract_embeddings(bucket_name, s3_images_dir)

knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3)  # Use a higher n_neighbors for better generalization
knn_clf.fit(images, labels)

model_filename = "face_recognition_model.clf"
joblib.dump(knn_clf, model_filename)

# Upload the trained model file to S3
with open(model_filename, 'rb') as model_file:
    s3.upload_fileobj(model_file, bucket_name, f"models/{model_filename}")

print("Model trained and saved successfully.")
