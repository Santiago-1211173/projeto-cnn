import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import random
import time
import sys

# --- 1. DEFINIÇÃO DA ARQUITETURA (Igual ao treino) ---
def get_robust_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model

def run_continuous_inference():
    # Setup inicial
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INIT] A carregar modelo para {device}...")
    
    # 1. Carregar Modelo (Só acontece uma vez!)
    model = get_robust_model().to(device)
    try:
        checkpoint = torch.load("outputs/checkpoints/mnist_final.pth", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
    except FileNotFoundError:
        print("ERRO: Não encontrei o ficheiro 'mnist_final.pth'. Treina o modelo primeiro!")
        return

    model.eval() # Modo de inferência
    
    # 2. Carregar Dataset de Teste
    print("[INIT] A carregar dados de teste...")
    stats = ((0.1307,), (0.3081,))
    tfms = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    # download=False porque já deves ter os dados
    test_ds = datasets.MNIST(root='./data', train=False, download=False, transform=tfms)
    
    print("\n--- A INICIAR STREAM DE PREVISÕES (Pressiona Ctrl+C para parar) ---")
    print(f"{'IMG_ID':<8} | {'REAL':<6} | {'PREVISTO':<8} | {'CONFIANÇA':<10} | {'STATUS'}")
    print("-" * 60)

    # 3. O Loop Infinito (Simulação de Agente)
    try:
        count = 0
        while True:
            # Escolher imagem aleatória
            idx = random.randint(0, len(test_ds)-1)
            img_tensor, label_real = test_ds[idx]
            
            # Preparar Batch (Adicionar dimensão extra [1, 1, 32, 32])
            img_input = img_tensor.unsqueeze(0).to(device)

            # --- A INFERÊNCIA (Isto é o que demora milissegundos) ---
            t0 = time.time()
            with torch.no_grad():
                output = model(img_input)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            t1 = time.time()
            
            # Interpretar
            confianca, previsto_tensor = torch.max(probabilities, 0)
            previsto = previsto_tensor.item()
            conf_pct = confianca.item() * 100
            
            # Lógica de Visualização
            status = "✅ OK" if previsto == label_real else "❌ ERRO"
            cor = "\033[92m" if previsto == label_real else "\033[91m" # Verde ou Vermelho (ANSI codes)
            reset = "\033[0m"
            
            # Print formatado
            print(f"{idx:<8} | {label_real:<6} | {cor}{previsto:<8}{reset} | {conf_pct:.2f}%     | {status}")
            
            # Pequena pausa para conseguires ler (remove se quiseres velocidade máxima)
            time.sleep(0.5) 
            count += 1

    except KeyboardInterrupt:
        print(f"\n\n[FIM] Stream interrompido pelo utilizador. Total analisado: {count} imagens.")

if __name__ == "__main__":
    run_continuous_inference()