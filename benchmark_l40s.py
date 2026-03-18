"""
Script de Benchmark para comparação de throughput entre CPU e NVIDIA L40S.
Mede o tempo real de execução de multiplicações matriciais massivas,
forçando a sincronização da GPU para evitar medições assíncronas falsas.
"""

import time
import logging
import tensorflow as tf

# Configuração de logging para output limpo no terminal
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_benchmark(device_name: str, matrix_size: int = 8192, iterations: int = 10) -> float:
    """
    Executa multiplicações matriciais no dispositivo especificado.
    
    Args:
        device_name: Identificador do dispositivo (ex: '/CPU:0' ou '/GPU:0').
        matrix_size: Dimensão N da matriz N x N (default: 8192x8192, exige ~700MB de RAM por matriz).
        iterations: Número de vezes a repetir a operação para obter a média.
        
    Returns:
        float: Tempo médio por iteração em segundos.
    """
    logger.info(f"\n--- Iniciando teste em {device_name} ---")
    
    try:
        with tf.device(device_name):
            # 1. Alocação de Tensores (Float32 puro)
            # Uma matriz 8192x8192 em float32 ocupa cerca de 268 MB.
            shape = [matrix_size, matrix_size]
            matrix_a = tf.random.normal(shape)
            matrix_b = tf.random.normal(shape)
            
            # 2. Warmup (Crucial para a GPU)
            # Forçamos a compilação JIT/PTX e a alocação de memória.
            logger.info("A executar Warmup (Compilação JIT/PTX)...")
            warmup_result = tf.matmul(matrix_a, matrix_b)
            # Em eager mode, o TensorFlow envia o comando para a GPU assincronamente.
            # Chamar .numpy() bloqueia o CPU até a GPU terminar fisicamente a operação.
            _ = warmup_result.numpy() 
            
            # 3. Benchmark Real
            logger.info(f"A executar {iterations} iterações de multiplicação {matrix_size}x{matrix_size}...")
            start_time = time.time()
            
            for _ in range(iterations):
                result = tf.matmul(matrix_a, matrix_b)
                
            # Bloqueio final para garantir que o tempo não para antes da GPU acabar
            _ = result.numpy()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / iterations
            logger.info(f"-> Tempo médio por iteração: {avg_time:.4f} segundos")
            
            return avg_time

    except Exception as e:
        logger.error(f"Falha ao executar benchmark em {device_name}: {e}")
        raise


def main() -> None:
    """Ponto de entrada do script."""
    # Verificar se as GPUs estão disponíveis e ativar memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    logger.info(f"Versão TensorFlow: {tf.__version__}")
    logger.info(f"Hardware detetado: {tf.config.list_physical_devices()}")

    # Executar na CPU
    time_cpu = run_benchmark('/CPU:0', matrix_size=8192, iterations=5)
    
    # Executar na L40S (GPU 0)
    time_gpu = run_benchmark('/GPU:0', matrix_size=8192, iterations=20)
    
    # Cálculo do Speedup
    if time_gpu > 0:
        speedup = time_cpu / time_gpu
        logger.info("\n==========================================")
        logger.info(f"RESULTADO FINAL: A NVIDIA L40S foi {speedup:.2f}x mais rápida que o CPU!")
        logger.info("==========================================")


if __name__ == "__main__":
    main()