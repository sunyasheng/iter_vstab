import argparse


class Options:
    def __init__(self, verbose=False):
        self.parser = argparse.ArgumentParser()
        self.verbose = verbose
        # argument list...
        self.parser.add_argument('--input_video', type=str, default='../test_in/examples_5')
        self.parser.add_argument('--output_video', type=str, default='../test_out/examples_5_out')
        # self.parser.add_argument('--style', type=str, default='wired_frame', help='font type, eg: heiti')
        # self.parser.add_argument('--resize_w', type=int, default=-1)
        # self.parser.add_argument('--custom_bold', type=str2bool, default='F')
        # # res_dir for crop
        # self.parser.add_argument('--res_dir', type=str, default='../inter_data/result')
        #
        self.opt = self.parser.parse_args()

    def parse(self):
        opts = vars(self.opt)
        if self.verbose:
            print('--------- load options ---------')
            for name, value in sorted(opts.items()):
                print('%s : %s' % (str(name), str(value)))
            print('--------- done loaded ----------')
        return self.opt


def str2bool(stri: str):
    return 't' in stri.lower() or '1' in stri.lower() or 'y' in stri.lower()
