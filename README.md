# Llama DNA Model Converter

Convert dnotitia/llama-dna-1.0-8b-instruct model using UV environment.

## Usage

```bash
# Install dependencies
uv sync

# Run conversion
uv run convert-model

# With 8bit quantization
uv run convert-model --use-8bit

# Check system only
uv run convert-model --check-only
```

# 변환된 모델을 Ollama에서 서비스하기

## 1. 전제 조건

### 모델 변환 완료 확인
먼저 제공하신 스크립트로 모델 변환이 완료되었는지 확인하세요:

```bash
# 변환된 모델 디렉토리 확인
ls -la ./converted_model/
```

다음 파일들이 있어야 합니다:
- `config.json`
- `tokenizer.json` 또는 `tokenizer_config.json`
- `pytorch_model.bin` 또는 `.safetensors` 파일들
- `conversion_info.json`

### Ollama 설치
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# 또는 직접 다운로드
# https://ollama.ai/download
```

## 2. Ollama용 Modelfile 생성

변환된 모델을 Ollama에서 사용하려면 먼저 GGUF 형식으로 변환해야 합니다.

### 2.1 llama.cpp 설치 및 변환

```bash
# llama.cpp 클론
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 컴파일
make

# Python 환경 설정
pip install -r requirements.txt

# HuggingFace 모델을 GGUF로 변환
python convert.py ./converted_model --outdir ./gguf_model --outtype f16
```

### 2.2 양자화 (선택사항)

```bash
# 4비트 양자화 (권장)
./quantize ./gguf_model/model.gguf ./gguf_model/model_q4_0.gguf q4_0

# 8비트 양자화
./quantize ./gguf_model/model.gguf ./gguf_model/model_q8_0.gguf q8_0
```

## 3. Ollama Modelfile 생성

```bash
# Modelfile 생성
cat > Modelfile << 'EOF'
FROM ./gguf_model/model_q4_0.gguf

# 템플릿 설정 (모델에 맞게 조정)
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

# 파라미터 설정
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

# 시스템 메시지
SYSTEM """You are a helpful AI assistant specialized in DNA and biological research."""
EOF
```

## 4. Ollama에 모델 등록

```bash
# 모델 생성
ollama create llama-dna-1.0-8b -f Modelfile

# 모델 목록 확인
ollama list
```

## 5. 모델 서비스 시작

### 5.1 로컬에서 테스트

```bash
# 대화형 테스트
ollama run llama-dna-1.0-8b

# 단일 질문 테스트
ollama run llama-dna-1.0-8b "What is DNA?"
```

### 5.2 API 서버 모드

```bash
# Ollama 서버 시작 (기본 포트: 11434)
ollama serve

# 백그라운드에서 실행
nohup ollama serve > ollama.log 2>&1 &
```

### 5.3 API 사용 예제

```bash
# REST API 호출
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "llama-dna-1.0-8b",
  "prompt": "Explain the structure of DNA",
  "stream": false
}'

# 스트림 모드
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "llama-dna-1.0-8b",
  "prompt": "What are the four bases of DNA?",
  "stream": true
}'
```

## 6. Python에서 사용하기

```python
import requests
import json

def query_ollama(prompt, model="llama-dna-1.0-8b"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}"

# 사용 예제
result = query_ollama("Explain the process of DNA replication")
print(result)
```

## 7. 고급 설정

### 7.1 GPU 가속 활성화

```bash
# NVIDIA GPU 사용
OLLAMA_GPU=nvidia ollama serve

# AMD GPU 사용
OLLAMA_GPU=rocm ollama serve
```

### 7.2 메모리 설정

```bash
# 컨텍스트 크기 조정
ollama run llama-dna-1.0-8b --ctx-size 8192

# GPU 메모리 할당량 설정
OLLAMA_GPU_MEMORY=8GB ollama serve
```

### 7.3 멀티 GPU 설정

```bash
# 여러 GPU 사용
OLLAMA_GPU_COUNT=2 ollama serve
```

## 8. 문제 해결

### 8.1 일반적인 오류

1. **메모리 부족**
   ```bash
   # 더 작은 양자화 모델 사용
   ./quantize model.gguf model_q2_k.gguf q2_k
   ```

2. **모델 로딩 실패**
   ```bash
   # 모델 파일 권한 확인
   chmod 644 ./gguf_model/*.gguf
   ```

3. **API 연결 실패**
   ```bash
   # 포트 확인
   netstat -tlnp | grep 11434
   
   # 방화벽 설정
   sudo ufw allow 11434
   ```

### 8.2 성능 최적화

1. **양자화 레벨 조정**
   - `q2_k`: 가장 작지만 정확도 낮음
   - `q4_0`: 균형잡힌 선택 (권장)
   - `q8_0`: 높은 정확도, 큰 크기

2. **배치 크기 조정**
   ```bash
   # Modelfile에서 설정
   PARAMETER num_batch 512
   ```

## 9. 서비스 배포

### 9.1 Docker로 배포

```dockerfile
FROM ollama/ollama:latest

COPY gguf_model/ /models/
COPY Modelfile /tmp/

RUN ollama create llama-dna-1.0-8b -f /tmp/Modelfile

EXPOSE 11434

CMD ["ollama", "serve"]
```

### 9.2 시스템 서비스 등록

```bash
# systemd 서비스 파일 생성
sudo tee /etc/systemd/system/ollama.service > /dev/null <<EOF
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=ollama
Group=ollama
ExecStart=/usr/local/bin/ollama serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화
sudo systemctl enable ollama
sudo systemctl start ollama
```

## 10. 모니터링 및 로깅

```bash
# 서비스 상태 확인
ollama ps

# 로그 확인
tail -f ollama.log

# 시스템 리소스 모니터링
htop
nvidia-smi  # GPU 사용량 확인
```

이제 변환된 모델이 Ollama에서 서비스되어 API를 통해 접근할 수 있습니다.
