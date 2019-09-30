import argparse
import os

class Options:
    def __init__(self, verbose=False):
        self.parser = argparse.ArgumentParser()
        self.verbose = verbose
        # argument list...
        self.parser.add_argument('--input_video', type=str, default='/Users/yashengsun/Downloads/Regular/18_dir/')
        self.parser.add_argument('--output_video', type=str, default='/Users/yashengsun/Pictures/480p/Regular/18stb_dir')


        self.opt = self.parser.parse_args()
        assert os.path.exists(self.opt.output_video), print('{} does not exists!'.format(self.opt.output_video))
        self.opt.stable_logfile = os.path.join(self.opt.output_video, 'stable.log')

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
