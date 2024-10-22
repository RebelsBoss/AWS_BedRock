import boto3
import json
import io
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup

# Параметри S3
s3_client = boto3.client('s3')
bucket_name = 'test-bedrock-base'  # Вкажіть ім'я вашого S3 бакета
file_key = 'resource.json'  # Файл із текстом у S3

# Завантаження файлу resource.json із S3
try:
    resource_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    resource_data = json.load(io.BytesIO(resource_obj['Body'].read()))
    print("Файл успішно завантажений із S3.")
except ClientError as e:
    print(f"Помилка завантаження файлу з S3: {e}")
    exit(1)

# Витягування контенту з поля body
html_content = resource_data.get("body", "")
if not html_content:
    print("Поле 'body' відсутнє або порожнє у файлі.")
    exit(1)

# Використовуємо BeautifulSoup для вилучення тексту з HTML у полі body
soup = BeautifulSoup(html_content, "html.parser")
extracted_text = soup.get_text(separator=" ")

# Створення клієнта Bedrock Runtime для AWS
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Налаштування моделі (Titan Text G1)
model_id = "amazon.titan-text-premier-v1:0"

# Промпт для генерації 5 тез (1 речення з ~7 слів кожна)
user_message = f"""
Below is some text. Please extract 5 key takeaways from the text. 
Each key takeaway should be a single sentence with around 7 words.

Text: {extracted_text}
"""

# Формуємо розмову з промптом
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Відправка повідомлення до моделі Titan Text G1
    response = client.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "stopSequences": [], "temperature": 0.7, "topP": 0.9},
        additionalModelRequestFields={}
    )

    # Отримуємо та виводимо відповідь від моделі
    response_text = response["output"]["message"]["content"][0]["text"]

    # Розділяємо відповідь на окремі речення та додаємо нумерацію
    response_lines = response_text.split('\n')
    numbered_theses = "\n".join([f"{i+1}. {line.strip()}" for i, line in enumerate(response_lines) if line.strip()])




    print(f"Згенеровані тези:\n{response_text}")

except (ClientError, Exception) as e:
    print(f"Помилка під час виклику моделі '{model_id}': {e}")
    exit(1)

# Збереження результату назад у S3
output_file_key = 'summary_result.json'  # Файл для збереження результату
output_data = {"summary": response_text}

try:
    # Збереження результату в S3 у форматі JSON
    s3_client.put_object(
        Bucket=bucket_name,
        Key=output_file_key,
        Body=json.dumps(output_data),
        ContentType='application/json'
    )
    print(f"Результат успішно збережено в S3: {output_file_key}")
except ClientError as e:
    print(f"Помилка під час збереження результату в S3: {e}")

