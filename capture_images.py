import cv2
import boto3
import os
from io import BytesIO

# Initialize AWS S3 client
s3 = boto3.client('s3', aws_access_key_id='aws_key', aws_secret_access_key='secret_key')

# Replace with your S3 bucket name
bucket_name = 'studentattendencesystem'

# Rest of your code
cam_port = 0
cam = cv2.VideoCapture(cam_port)

student_name = input("Enter student's name: ")
student_roll = input("Enter student's roll number: ")

image_count = 0

while image_count < 20:
    ret, frame = cam.read()

    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    image_filename = f"images/{student_roll}/{student_roll}_{image_count}.png"

    # Encode the image data
    _, temp_image_data = cv2.imencode('.png', frame)
    
    # Create a BytesIO object to wrap the image data
    image_stream = BytesIO(temp_image_data)

    # Upload the image to S3 using the BytesIO stream
    s3.upload_fileobj(image_stream, bucket_name, image_filename)

    print(f"Image {image_count + 1} captured and uploaded to S3.")

    image_count += 1

    if image_count == 50:
        break

cam.release()
cv2.destroyAllWindows()
