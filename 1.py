import base64
import boto3
import json
import os
import io

# Параметри S3
s3_client = boto3.client('s3')
bucket_name = 'test-bedrock-base'
file_key = 'summary_result.json'

# Завантаження файлу summary_result.json із S3
try:
    resource_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    resource_data = json.load(io.BytesIO(resource_obj['Body'].read()))
    print("Файл успішно завантажений із S3.")
except ClientError as e:
    print(f"Помилка завантаження файлу з S3: {e}")
    exit(1)

# Витягуємо необхідні параметри з файлу
prompt = resource_data.get("summary", "").splitlines()[0]  # Беремо першу тезу для генерації
cfg_scale = resource_data.get("cfg_scale", 10)
steps = resource_data.get("steps", 50)
seed = resource_data.get("seed", 0)
width = resource_data.get("width", 1024)
height = resource_data.get("height", 1024)
samples = resource_data.get("samples", 1)

# Перевірка, чи існує prompt
if not prompt:
    print("Помилка: 'prompt' не знайдено у файлі.")
    exit(1)

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Set the model ID, e.g., Stability AI's Stable Diffusion.
model_id = "stability.stable-diffusion-xl-v1"

# Формування запиту для моделі на основі параметрів із файлу
native_request = {
    "text_prompts": [{"text": prompt, "weight": 1}],
    "cfg_scale": cfg_scale,
    "steps": steps,
    "seed": seed,
    "width": width,
    "height": height,
    "samples": samples
}

# Конвертація запиту в JSON
try:
    request = json.dumps(native_request)
    
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)

    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract the image data.
    base64_image_data = model_response["artifacts"][0]["base64"]

    # Save the generated image to a local folder.
    i, output_dir = 1, "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    while os.path.exists(os.path.join(output_dir, f"image_{i}.png")):
        i += 1

    image_data = base64.b64decode(base64_image_data)

    image_path = os.path.join(output_dir, f"image_{i}.png")
    with open(image_path, "wb") as file:
        file.write(image_data)

    print(f"The generated image has been saved to {image_path}.")
except Exception as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")

