# 1️⃣ Base image
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements first (for caching)
COPY requirements.txt .

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy entire project
COPY . .

# 6️⃣ Expose Flask port
EXPOSE 5000

# 7️⃣ Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 8️⃣ Run Flask app
CMD ["python", "app.py"]
