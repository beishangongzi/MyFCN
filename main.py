# create by andy at 2022/5/9
# reference:
from optparse import OptionParser

from train import train


def main():
    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model", default="FCN32", type="str",
                      help="model for train (default: FCN32)")

    (options, args) = parser.parse_args()
    train(options.model)

if __name__ == '__main__':
    main()