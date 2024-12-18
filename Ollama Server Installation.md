# **Manual de Instalação e Configuração do Ollama Server**

## **1. Pré-requisitos**

Antes de começar, certifique-se de ter:

- **Ubuntu 24.04**.
- **NVIDIA Driver** instalado.
- **CUDA Toolkit** (versão >= 12.4).
- **Docker** instalado e configurado para usar a GPU.
- **NVIDIA Container Toolkit** instalado.

---

## **2. Verificações Iniciais**

### **2.1 Verificar o Driver NVIDIA e CUDA**

Execute o comando:

```bash
nvidia-smi
```

- Confirme que os drivers NVIDIA estão instalados e a versão CUDA é compatível (>= 12.4).

---

### **2.2 Verificar Docker Instalado**

Verifique se o Docker está instalado executando:

```bash
docker --version
```

Se o Docker não estiver instalado, siga estas etapas:

#### Instalar o Docker

1. Atualize o sistema e instale pacotes necessários:
   
   ```bash
   sudo apt update
   sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
   ```

2. Adicione o repositório oficial do Docker:
   
   ```bash
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
   echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

3. Instale o Docker:
   
   ```bash
   sudo apt update
   sudo apt install -y docker-ce docker-ce-cli containerd.io
   ```

4. Adicione seu usuário ao grupo Docker:
   
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

5. Teste o Docker:
   
   ```bash
   docker run hello-world
   ```

---

### **2.3 Configurar NVIDIA Container Toolkit**

Para permitir que o Docker utilize a GPU, instale o **NVIDIA Container Toolkit**:

1. Instale o toolkit:
   
   ```bash
   sudo apt install -y nvidia-container-toolkit
   ```

2. Configure o Docker:
   
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

3. Teste a configuração:
   
   ```bash
   docker run --gpus all nvidia/cuda:12.4.1-base nvidia-smi
   ```
   
   - Verifique se o comando mostra a GPU corretamente.

---

## **3. Instalar e Configurar o Ollama Server**

1. **Baixe e instale o Ollama usando Docker**:
   
   ```bash
   docker pull ollama/ollama
   ```

2. **Inicie o Ollama Server**:
   
   ```bash
   docker run --gpus all -d --name ollama-server -p 11434:11434 ollama/ollama
   ```

3. **Verifique se o servidor está rodando**:
   
   ```bash
   curl http://localhost:11434/status
   ```
   
   - A resposta deve conter informações básicas do servidor.

---

## **4. Configurar o Ollama como um Serviço `systemd`**

Para iniciar o Ollama Server automaticamente na inicialização do sistema, criaremos um serviço `systemd`.

### **4.1 Criar o Arquivo de Serviço**

Crie o arquivo `ollama.service` em `/etc/systemd/system/`:

```bash
sudo nano /etc/systemd/system/ollama.service
```

Adicione o conteúdo abaixo:

```ini
[Unit]
Description=Ollama Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/docker run --gpus all --rm -p 11434:11434 ollama/ollama
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

### **4.2 Ativar e Iniciar o Serviço**

1. **Recarregue o `systemd`**:
   
   ```bash
   sudo systemctl daemon-reload
   ```

2. **Ative o serviço para iniciar automaticamente**:
   
   ```bash
   sudo systemctl enable ollama
   ```

3. **Inicie o serviço Ollama**:
   
   ```bash
   sudo systemctl start ollama
   ```

4. **Verifique o status do serviço**:
   
   ```bash
   sudo systemctl status ollama
   ```

---

## **5. Testar o Ollama Server**

### **5.1 Testar o Modelo Básico**

Baixe e rode um modelo Llama para teste. Por exemplo, o **Llama 2**:

1. Puxe o modelo:
   
   ```bash
   ollama pull llama2
   ```

2. Execute o modelo:
   
   ```bash
   ollama run llama2
   ```

3. Envie uma mensagem de teste:
   
   ```bash
   curl -X POST http://localhost:11434/api/generate -d '{
      "model": "llama2",
      "prompt": "Olá, como você está?",
      "stream": false
   }' | jq
   ```
   
   - A resposta deve conter o texto gerado pelo modelo.

---

## **6. Script para Testes Automatizados**

Crie um script Python simples para verificar o servidor Ollama:

```python
import requests

url = "http://localhost:11434/api/generate"
data = {
    "model": "llama2",
    "prompt": "Teste do servidor Ollama.",
    "stream": False
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Servidor Ollama funcionando corretamente!")
    print("Resposta do modelo:", response.json()["response"])
else:
    print("Erro ao acessar o servidor Ollama:", response.status_code)
```

### **Executar o Script**

1. Salve o script como `test_ollama.py`.

2. Execute:
   
   ```bash
   python3 test_ollama.py
   ```

---

## **Resumo**

Este manual cobre:

1. Verificação dos pré-requisitos (GPU, Docker, CUDA).
2. Instalação e configuração do Ollama usando Docker.
3. Configuração do Ollama como serviço `systemd`.
4. Testes para verificar o servidor Ollama.
