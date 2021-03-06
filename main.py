import argparse
from solver import Solver
from torch.backends import cudnn
from utils import make_folder

channel_list = [512, 512, 512, 512, 256, 128, 64, 32]


def main(config):
    # find optimal set of algorithm on our configuration setting
    cudnn.benchmark = True
    log_root, model_root, sample_root = make_folder(config.exp, config.log_dir, config.model_dir,
                                                    config.sample_dir)
    config.log_root = log_root
    config.model_root = model_root
    config.sample_root = sample_root

    solver = Solver(config, channel_list)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Config - Model
    parser.add_argument('--z_dim', type=int, default=512, help='dimension of random vector')

    # Config - Training
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--g_lr', type=float, default=0.001, help='gradient of generator')
    parser.add_argument('--d_lr', type=float, default=0.001, help='gradient of discriminator')
    parser.add_argument('--decay_iter', type=int, default=10000, help='learning rate decay iteration')
    parser.add_argument('--decay_ratio', type=float, default=0.1, help='learning rate decay ratio')
    parser.add_argument('--beta1', type=float, default=0.0, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.99, help='beta2')
    parser.add_argument('--lambda_value', type=float, default=10.0,
                        help='weight of gradient penalty or r1 regularization')
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=50000)
    parser.add_argument('--loss_name', type=str, default='wgan-gp', help='input stylegan or wgan-gp')

    # Config - Path
    parser.add_argument('--data_root', type=str, default="/home/nas1_userC/yonggyu/dataset/FFHQ")
    parser.add_argument('--exp', type=str, default='StyleGAN_512')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--model_dir', type=str, default='model')
    parser.add_argument('--sample_dir', type=str, default='sample')
    parser.add_argument('--result_dir', type=str, default='result')

    # Config - Miscellanceous
    parser.add_argument('--print_loss_iter', type=int, default=5000)
    parser.add_argument('--save_image_iter', type=int, default=5000)
    parser.add_argument('--save_parameter_iter', type=int, default=5000)
    parser.add_argument('--save_log_iter', type=int, default=5000)
    config = parser.parse_args()

    print(config)
    main(config)
