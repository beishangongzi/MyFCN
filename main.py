# create by andy at 2022/5/9
# reference:
from optparse import OptionParser

from train import train


def main():
    parser = OptionParser()
    parser.add_option("--prefix", dest="prefix", default="a", type=str, help="the prefix of model name")
    parser.add_option("-m", "--model", dest="model", default="FCN32", type="str",
                      help="model for train (default: FCN32)")


    (options, args) = parser.parse_args()
    save_name = "-".join(options.__dict__.values())
    train(options.model, save_name)

if __name__ == '__main__':
    main()