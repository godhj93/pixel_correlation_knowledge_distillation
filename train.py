import tensorflow as tf
from utils.op import Trainer
import argparse
from utils.networks.BinaryDenseNet import BinaryDenseNet
from utils.networks.Baseline_DenseNet import DenseNet
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

parser = argparse.ArgumentParser("Pixel-Correlation Knowledge distillation")
parser.add_argument("--ep", default=100, type=int,help="Epochs")
parser.add_argument("--bs", default=32, type=int,help="Batch Size")
parser.add_argument("--data", default='cifar10')
parser.add_argument("--size", default=32, type=int, help="data size")
parser.add_argument("--name", default='MODEL')
parser.add_argument("--kd", help='Knowledge Distillation -> bool')


args = parser.parse_args()

def main():
    if args.data == 'cifar10':
        classes = 10
    elif args.data == 'cifar100':
        classes =100
    else:
        raise ValueError("Data must be cifar10 or cifar100")

    model = BinaryDenseNet(kd=args.kd, arch='bdn-28',classes=classes).model(input_shape=(args.size,args.size,3))
    #model = MobileNetv1(classes=classes).model(input_shape=(args.size, args.size, 3))
    print(model.summary())
    trainer = Trainer(model, kd=args.kd, dataset=args.data, epochs=args.ep, batch_size=args.bs, size=args.size, name=args.name ,DEBUG=False)
    trainer.train()
    
    
if __name__ == '__main__':

    main()
    print("Done.")