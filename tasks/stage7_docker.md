# Stage 7 — Docker & Deployment

**Status:** Not started  
**Files:** `docker/Dockerfile.pipeline`, `docker/Dockerfile.dashboard`, `docker/docker-compose.yml`  
**Input:** Everything built in Stages 1–6  
**Output:** A single `docker compose up` command that runs the full pipeline and serves the dashboard

---

## What

Containerise the pipeline and the dashboard so the entire project can be reproduced on any machine with one command. The pipeline runs as a container that executes all stages in order and writes the JSON files. The dashboard runs as an nginx container that serves the HTML/JS/JSON files.

---

## Why This Approach

### Why Docker for a portfolio project?
"It runs on my machine" is not a deliverable. A hiring manager who clones your repo wants to run it. Without Docker, they need to install Python, the right version, all your dependencies, Java for PySpark, configure paths — and hope nothing conflicts with their existing setup.

With Docker: `docker compose up` and it works. This is the minimum bar for anything that calls itself a production system.

### Why two containers instead of one?
Separation of concerns. The pipeline is a batch job — it runs, finishes, and exits. The dashboard is a long-running server. Combining them in one container would mean the server never starts until the pipeline finishes, and you can't restart the dashboard without rerunning the pipeline. Two containers is the correct architecture.

### Why docker-compose and not just two Dockerfiles?
`docker-compose` defines multi-container applications as a single unit. One command starts both containers, handles networking between them, and manages startup order. It's also what most teams use for local development of multi-service applications.

---

## New Concepts to Learn Before Building

### Dockerfile basics
A Dockerfile is a recipe for building a container image. Each line is an instruction:

```dockerfile
FROM python:3.11-slim          # start from this base image
WORKDIR /app                   # all subsequent commands run here
COPY requirements.txt .        # copy one file from host to container
RUN pip install -r requirements.txt  # run a command during build
COPY . .                       # copy everything else
CMD ["python", "pipeline/run_pipeline.py"]  # run this when the container starts
```

### Docker Compose basics
```yaml
services:
  pipeline:
    build:
      context: ..
      dockerfile: docker/Dockerfile.pipeline
    volumes:
      - ../data:/app/data          # mount host data/ into container
      - ../dashboard/data:/app/dashboard/data
      - ../mlruns:/app/mlruns

  dashboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dashboard
    ports:
      - "8080:80"                  # host port 8080 → container port 80
    depends_on:
      - pipeline
```

`volumes` mount directories from your host machine into the container. This means the pipeline can write Parquet and JSON files that persist after the container exits, and that the dashboard container can serve.

### Why PySpark needs Java
Spark is written in Scala/Java. PySpark is a Python wrapper. You need a JVM installed in the container:

```dockerfile
RUN apt-get update && apt-get install -y default-jdk-headless
ENV JAVA_HOME=/usr/lib/jvm/default-java
```

---

## How to Build It (Step by Step)

### Step 1 — Create `docker/Dockerfile.pipeline`

```dockerfile
FROM python:3.11-slim

# Install Java for PySpark
RUN apt-get update && apt-get install -y default-jdk-headless procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PYSPARK_PYTHON=python3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./pipeline/
COPY config/ ./config/
COPY analysis/ ./analysis/

CMD ["python", "pipeline/run_pipeline.py"]
```

### Step 2 — Create `docker/Dockerfile.dashboard`

```dockerfile
FROM nginx:alpine

# Remove default nginx page
RUN rm -rf /usr/share/nginx/html/*

# Copy dashboard files
COPY dashboard/ /usr/share/nginx/html/

# Custom nginx config to serve from /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
```

### Step 3 — Create `docker/nginx.conf`

```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /data/ {
        add_header 'Access-Control-Allow-Origin' '*';
    }
}
```

The `Access-Control-Allow-Origin` header allows the browser to fetch the JSON data files without CORS errors.

### Step 4 — Create `docker/docker-compose.yml`

```yaml
version: "3.9"

services:
  pipeline:
    build:
      context: ..
      dockerfile: docker/Dockerfile.pipeline
    volumes:
      - ../data:/app/data
      - ../dashboard/data:/app/dashboard/data
      - ../mlruns:/app/mlruns
      - ../config:/app/config
    environment:
      - ODDS_API_KEY=${ODDS_API_KEY}

  dashboard:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dashboard
    ports:
      - "8080:80"
    volumes:
      - ../dashboard/data:/usr/share/nginx/html/data
    depends_on:
      - pipeline
```

The `${ODDS_API_KEY}` uses an environment variable from the host so you never hardcode a secret in a file that gets committed to git.

### Step 5 — Create `pipeline/run_pipeline.py`

This script runs all stages in order:

```python
import subprocess
import sys

stages = [
    "pipeline/stage1_ingest.py",
    "pipeline/stage2_features.py",
    "pipeline/stage3_train.py",
    "pipeline/stage4_odds_gen.py",
    "pipeline/stage5_compare.py",
    "pipeline/export_dashboard_data.py",
]

for stage in stages:
    print(f"\n{'='*50}\nRunning {stage}\n{'='*50}")
    result = subprocess.run([sys.executable, stage], check=True)
    print(f"Completed: {stage}")

print("\nPipeline complete.")
```

### Step 6 — Add a `.env` file (never commit this)
```
ODDS_API_KEY=your_actual_key_here
```

Add `.env` to `.gitignore`.

### Step 7 — Build and test

```bash
# From the project root
docker compose -f docker/docker-compose.yml up --build

# Once the pipeline finishes and the dashboard is running:
# Open http://localhost:8080
```

### Step 8 — Add a one-command run instruction to the README

The README should have a section that says exactly:
```bash
git clone https://github.com/iraklisp98/sports_modelling.git
cd sports_modelling
echo "ODDS_API_KEY=your_key" > .env
docker compose -f docker/docker-compose.yml up --build
# Open http://localhost:8080
```

That's it. Five lines. Anyone can run this project.

---

## Acceptance Criteria

- [ ] `docker compose -f docker/docker-compose.yml up --build` runs without errors
- [ ] Pipeline container runs all stages in order and exits with code 0
- [ ] Dashboard container serves `index.html` at `http://localhost:8080`
- [ ] All four dashboard tabs load with data (JSON files present in the volume)
- [ ] Stopping and restarting the dashboard container does not require rerunning the pipeline
- [ ] API key is loaded from an environment variable, not hardcoded in any file
- [ ] `.env` is in `.gitignore`

---

## Interview Q&A

**Q: Why did you containerise this project?**  
A: "Reproducibility. Without Docker, running this project requires installing Python, the right version, all dependencies, Java for PySpark, and configuring paths — and hoping nothing conflicts with the user's existing environment. Docker packages everything. Anyone with Docker can clone the repo and run the full pipeline with one command. That's the minimum bar for production software."

**Q: Walk me through your Docker architecture.**  
A: "Two containers. The pipeline container runs Python 3.11 with all dependencies including PySpark and Java. It executes all pipeline stages in order and writes output files to a shared volume. The dashboard container runs nginx and serves the static HTML, CSS, JS, and JSON files from that same volume. Docker Compose wires them together and ensures the pipeline runs before the dashboard starts."

**Q: How do you handle secrets in Docker?**  
A: "The API key is never hardcoded in any file. It's read from an environment variable `ODDS_API_KEY` which Docker Compose picks up from a `.env` file on the host. The `.env` file is in `.gitignore` so it never gets committed. In a real production environment this would come from a secrets manager like AWS Secrets Manager or HashiCorp Vault."

**Q: What's the difference between `COPY` and a volume mount in Docker?**  
A: "`COPY` bakes files into the image at build time — they're static. A volume mount links a directory on the host to a directory in the container at runtime — changes on one side are reflected on the other. I use volumes for the data directories so the pipeline can write output files that persist after the container exits and that the dashboard container can serve, without rebuilding either image."
