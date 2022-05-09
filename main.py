# create by andy at 2022/5/9
# reference:
from optparse import OptionParser

from train import run


def main():
    parser = OptionParser()
    parser.add_option("--prefix", dest="prefix", default="a", type=str, help="the prefix of model name")
    parser.add_option("-m", "--model", dest="model", default="FCN32", type="str",
                      help="model for train (default: FCN32)")
    parser.add_option("-l", "--load_model", dest="load_model", default="", type="str",
                      help="model for train")
    parser.add_option("--mode", dest="mode", default="train", type="str",
                      help="model for train (default: FCN32)")
    parser.add_option("--dataset", dest="dataset", default="data/obt/image", type="str",
                      help="dataset path)")
    (options, args) = parser.parse_args()
    save_name = "-".join(options.__dict__.values())
    run(options.model, save_name, dataset=options.dataset, mode=options.mode, load_name=options.load_model)


if __name__ == '__main__':
    main()
