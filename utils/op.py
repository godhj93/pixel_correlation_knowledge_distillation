from logging.config import valid_ident
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.data import data_loader
from tensorflow.keras.optimizers import Adam
import copy
from datetime import datetime
from utils.loss import pixel_correlation_loss
class Trainer:
    '''
    Train a Neural Network
    Author: H.J Shin
    Date: 2022.05.02
    '''
    def __init__(self, model, kd, dataset, epochs, batch_size, size, name='MODEL', DEBUG=False):
        '''
        model: model for training.
        dataset: cifar10 or cifar100.
        epochs: positive int
        batch_size: positive int
        '''
        super(Trainer, self).__init__()

        self.kd = kd
        if self.kd:
            from utils.networks.Baseline_DenseNet import DenseNet
            self.teacher = DenseNet(kd=self.kd).model(input_shape=(size,size,3))
            self.teacher.load_weights('./ckpt/2022-07-28/MODEL_22-44-11/') # Acc: 8665
            print(self.teacher.summary())
            print("Teacher has been loaded!")

        self.name = name
        self.batch_size = batch_size
        self._model = copy.deepcopy(model)
        self._epochs = epochs
        self.data_loader = data_loader(dataset=dataset, batch_size=batch_size, size=size, DEBUG=DEBUG)
        self.train_ds, self.test_ds = self.data_loader.load()
        self._optimizer = Adam(learning_rate = self.LR_Scheduler())
        self.CrossEntropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
        self.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_path = './models/' + self.time[:10] + '/' + self.name + self.time[10:] + '/'
        self.ckpt_path = './ckpt/' + self.time[:10] + '/' + self.name + self.time[10:] + '/'
        #Tensorboard
        train_log_dir = 'logs/' + self.time[:10] + '/' + self.name + self.time[10:] + '/train'
        test_log_dir = 'logs/' + self.time[:10] + '/' + self.name + self.time[10:] + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def LR_Scheduler(self):
        
        return LearningRateScheduler(initial_learning_rate=0.0002, steps=np.ceil(50000/ self.batch_size))
        
    def progress_bar(self, dataset):
        if dataset == 'train':
            return tqdm(self.train_ds, ncols=0)
        elif dataset == 'test':
            return tqdm(self.test_ds, ncols=0)
        else:
            raise ValueError("dataset must be 'train' or 'test'")

    

    
    def train(self):
        print(f"Initializing...")
        
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')

        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

        best_acc = 0
        for e in range(self._epochs):
            print(f"\nEPOCH: {e+1}/{self._epochs}")
            
            train_bar = self.progress_bar('train')
            for x,y in train_bar:
                self.train_step(x,y)
                train_bar.set_description(f"Loss: {self.train_loss.result().numpy():.4f}, Acc: {self.train_accuracy.result().numpy():.4f}, Learning Rate: {self._optimizer._decayed_lr('float32').numpy()}")
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=e)
                tf.summary.scalar('accuracy', self.train_accuracy.result(), step=e)

            test_bar = self.progress_bar('test')
            for x,y in test_bar:
                self.test_step(x,y)
                test_bar.set_description(f"Loss: {self.test_loss.result().numpy():.4f}, Acc: {self.test_accuracy.result().numpy():.4f}")
            with self.test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=e)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=e)

            if best_acc < self.test_accuracy.result().numpy():
                self._model.save_weights(self.ckpt_path)
                print(f"The best accuracy has been updated {self.test_accuracy.result().numpy():.4f}... Save checkpoint...")
                best_acc = self.test_accuracy.result().numpy()
            self.reset_metric()
        

        
        print(f"Training is completed.")
        self.save_model()
    
    def reset_metric(self):

        self.train_loss.reset_states()
        self.test_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_accuracy.reset_states()

    @tf.function
    def train_step(self, x,y):
              
        with tf.GradientTape() as tape:
                
            y_hat = self._model(x, training=True)
            loss = self.CrossEntropy(y,y_hat[-1])

            if self.kd:
                loss_pc = 0
                yt_hat = self.teacher(x,training=False)
                
                for y_t, y_s in zip(yt_hat[:-1], y_hat[:-1]):
                    loss_pc += pixel_correlation_loss(y_t, y_s)
                
                lamb = 10
                loss_total = loss + lamb*loss_pc

        if self.kd:
            grads = tape.gradient(loss_total, self._model.trainable_variables)    
        
        else:
            grads = tape.gradient(loss, self._model.trainable_variables)

        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
        
        self.train_accuracy.update_state(y, y_hat[-1])
        self.train_loss.update_state(loss)
       
    @tf.function
    def test_step(self, x,y):
              
        y_hat = self._model(x, training=False)
        loss = self.CrossEntropy(y,y_hat[-1])
        self.test_accuracy.update_state(y, y_hat[-1])
        self.test_loss.update_state(loss)

    def save_model(self):

        self._model.save(self.save_path)
        print(f'the model has been saved in {self.save_path}')


class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

        def __init__(self, initial_learning_rate=0.0002,steps=3000):
            print(f"Total Steps: {steps}")
            self.steps = steps

            self.initial_learning_rate = initial_learning_rate
            
            
            self.cosine_annealing = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=0.002,
                first_decay_steps= self.steps,
                t_mul=1.0,
                m_mul=1.0,
                alpha=2e-4,
                name=None
    )

        def __call__(self, step):
            return tf.cond(step<=self.steps, lambda: self.linear_increasing(step) ,lambda: self.cosine_annealing(step) )
            # return self.cosine_annealing(step)
        
        def linear_increasing(self, step):
            return (0.002-0.0002)/(self.steps)*step + self.initial_learning_rate
