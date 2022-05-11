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
    parser.add_option("--backbone", dest="backbone", default="resnet50", type="str",
                      help="dataset path)")
    parser.add_option("--save_freq", dest="save_freq", default=20, type=int,
                      help="how many epoch to save model")
    (options, args) = parser.parse_args()
    save_name = "-".join(options.__dict__.values()).replace("/", "_")
    run(options.model, save_name,
        backbone=options.backbone,
        dataset=options.dataset,
        mode=options.mode,
        load_name=options.load_model,
        save_freq=options.save_freq)


if __name__ == '__main__':
    main()
