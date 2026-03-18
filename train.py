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

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Hiperparâmetros
# Hiperparâmetros
BATCH_SIZE_PER_REPLICA = 64
EPOCHS = 10
# A nossa Learning Rate base (0.05) multiplicada pelas 2 L40S
LEARNING_RATE = 0.05 * 2
DATA_DIR = os.path.join("data", "MNIST", "raw")


def calculate_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    predicted_classes = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    correct_predictions = tf.equal(predicted_classes, y_true)
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def main() -> None:
    # 1. Configurar Hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 2. Instanciar a Estratégia Multi-GPU
    # 2. Instanciar a Estratégia Multi-GPU contornando o NCCL no Windows
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    logger.info(f"\n=========================================")
    logger.info(f"MirroredStrategy: {strategy.num_replicas_in_sync} L40S em sincronia")
    logger.info(f"=========================================\n")

    # O batch global é o batch por gráfica multiplicado pelo número de gráficas (256 * 2 = 512)
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    # 3. Carregar e Distribuir Dados
    train_dataset = create_dataset(DATA_DIR, batch_size=GLOBAL_BATCH_SIZE)
    # Partir o pipeline de dados ao meio para alimentar as duas GPUs em simultâneo
    dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    # 4. Instanciar Modelo e Otimizador DENTRO do Scope da Estratégia
    # Isto garante que o TensorFlow aloca as variáveis nas duas GPUs
    with strategy.scope():
        model = RawModel()
        optimizer = SGD(learning_rate=LEARNING_RATE)

        # O TensorBoard Writer também vive bem aqui
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("outputs", "logs", current_time)
        summary_writer = tf.summary.create_file_writer(log_dir)

    # 5. O Passo de Treino (Executado individualmente em CADA GPU)
    def train_step(x_batch: tf.Tensor, y_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            predictions = model(x_batch)
            loss = categorical_crossentropy(y_batch, predictions, GLOBAL_BATCH_SIZE)

        # 1. Cada GPU calcula os seus gradientes locais
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # 2. MAGIA DISTRIBUÍDA: Forçar as duas L40S a somarem os gradientes (All-Reduce)
        replica_context = tf.distribute.get_replica_context()
        gradients = replica_context.all_reduce(tf.distribute.ReduceOp.SUM, gradients)

        # 3. Agora sim, com os gradientes iguais e somados, ambas atualizam os pesos em perfeita sincronia
        optimizer.apply_gradients(model.trainable_variables, gradients)
        
        accuracy = calculate_accuracy(y_batch, predictions)
        return loss, accuracy

    # 6. O Passo Distribuído (Orquestra a chamada às duas GPUs e junta os resultados)
    @tf.function
    def distributed_train_step(dataset_inputs: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, float]:
        x_batch, y_batch = dataset_inputs
        
        # O strategy.run envia a função train_step para as duas L40S
        per_replica_losses, per_replica_accs = strategy.run(train_step, args=(x_batch, y_batch))
        
        # Juntar (Reduzir) os resultados: Somar as losses e fazer a média das accuracies
        global_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        global_acc = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accs, axis=None)
        
        return global_loss, global_acc

    # 7. O Loop de Treino
    logger.info("A iniciar varrimento do dataset...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        # Iterar sobre o Dataset Distribuído
        for step, dist_inputs in enumerate(dist_dataset):
            loss, acc = distributed_train_step(dist_inputs)
            
            total_loss += float(loss)
            total_acc += float(acc)
            steps += 1

            if step % 50 == 0 and step > 0:
                logger.info(f"  [Época {epoch+1} | Batch Global {step}] Loss: {float(loss):.4f} | Acc: {float(acc):.4f}")

        avg_loss = total_loss / steps
        avg_acc = total_acc / steps
        epoch_time = time.time() - start_time

        logger.info(f"-> FIM DA ÉPOCA {epoch+1}: Tempo: {epoch_time:.2f}s | Loss Média: {avg_loss:.4f} | Acc Média: {avg_acc:.4f}\n")

        with summary_writer.as_default():
            tf.summary.scalar('Loss/Treino', avg_loss, step=epoch)
            tf.summary.scalar('Accuracy/Treino', avg_acc, step=epoch)

if __name__ == "__main__":
    main()