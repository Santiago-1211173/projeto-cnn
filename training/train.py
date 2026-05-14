"""
Script principal de treino (Single-GPU).
Treina a CNN dissecada ('Headless') forçando a execução na GPU 0.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import time
import datetime
import logging
from typing import Tuple
import tensorflow as tf

from src.data.loader import create_dataset
from src.models.custom_cnn import RawModel
from src.scratch.losses import categorical_crossentropy
from src.scratch.optimizers import SGD

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Hiperparâmetros
BATCH_SIZE = 256  # Numa só L40S podemos voltar a subir o batch size confortavelmente
EPOCHS = 10
LEARNING_RATE = 0.01  # Descida de gradiente segura e estável
DATA_DIR = os.path.join("data", "MNIST", "raw")


def calculate_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """Calcula a precisão comparando a classe de maior probabilidade com a real."""
    predicted_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct_predictions = tf.equal(predicted_classes, y_true)
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def main() -> None:
    # 1. Configurar Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    logger.info("\n=========================================")
    logger.info("A iniciar treino focado numa única NVIDIA L40S (GPU 0)")
    logger.info("=========================================\n")

    # 2. Carregar Dados
    train_dataset = create_dataset(DATA_DIR, batch_size=BATCH_SIZE)

    # 3. Forçar a alocação na GPU Primária
    with tf.device('/GPU:0'):
        # Instanciar Modelo e Otimizador
        model = RawModel()
        optimizer = SGD(learning_rate=LEARNING_RATE)

        # Configurar o TensorBoard Writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("outputs", "logs", current_time)
        summary_writer = tf.summary.create_file_writer(log_dir)

        # 4. O Passo de Treino (Compilado via XLA para máxima velocidade)
        @tf.function
        def train_step(x_batch: tf.Tensor, y_batch: tf.Tensor) -> Tuple[tf.Tensor, float]:
            with tf.GradientTape() as tape:
                # AQUI: O modelo agora devolve o dicionário da autópsia. 
                # Extraímos as probabilidades para calcular o erro.
                outputs = model(x_batch)
                predictions = outputs["probabilities"]
                
                # A Loss usa o BATCH_SIZE escalar para manter o rigor matemático
                loss = categorical_crossentropy(y_batch, predictions, BATCH_SIZE)

            # Backpropagation
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(model.trainable_variables, gradients)
            
            # Cálculo de métricas
            accuracy = calculate_accuracy(y_batch, predictions)
            return loss, accuracy

        # 5. O Loop de Treino
        logger.info("A iniciar varrimento do dataset...")
        for epoch in range(EPOCHS):
            start_time = time.time()
            total_loss = 0.0
            total_acc = 0.0
            steps = 0

            for step, (x_batch, y_batch) in enumerate(train_dataset):
                loss, acc = train_step(x_batch, y_batch)
                
                total_loss += float(loss)
                total_acc += float(acc)
                steps += 1

                if step % 50 == 0 and step > 0:
                    logger.info(f"  [Época {epoch+1} | Batch {step}] Loss: {float(loss):.4f} | Acc: {float(acc):.4f}")

            # Métricas finais da época
            avg_loss = total_loss / steps
            avg_acc = total_acc / steps
            epoch_time = time.time() - start_time

            logger.info(f"-> FIM DA ÉPOCA {epoch+1}: Tempo: {epoch_time:.2f}s | Loss Média: {avg_loss:.4f} | Acc Média: {avg_acc:.4f}\n")

            # Escrever para o TensorBoard
            with summary_writer.as_default():
                tf.summary.scalar('Loss/Treino', avg_loss, step=epoch)
                tf.summary.scalar('Accuracy/Treino', avg_acc, step=epoch)

            # AQUI: Guardar os pesos no disco no final do treino
            logger.info("A guardar checkpoint do modelo...")
            ckpt = tf.train.Checkpoint(model=model)
            ckpt.save(os.path.join("outputs", "checkpoints", "modelo_dissecado"))
            logger.info("Treino e gravação concluídos com sucesso!")

if __name__ == "__main__":
    main()