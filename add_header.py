import argparse
import fnmatch
import os

CC_HEADER = """// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

"""

PY_HEADER = """# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

"""

SH_HEADER = """# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

"""

TEX_HEADER = """% Copyright (c) Meta Platforms, Inc. and affiliates.
%
% This software may be used and distributed according to the terms of the
% GNU General Public License version 2.

"""

FILE_TO_HEADER = {
    "cc": CC_HEADER,
    "py": PY_HEADER,
    "sh": SH_HEADER,
    "tex": TEX_HEADER,
}

parser = argparse.ArgumentParser()
parser.add_argument("dir")
args = parser.parse_args()


def add_header(file, file_type):
    print(f"(Add header to {file}")
    with open(file, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(FILE_TO_HEADER[file_type] + content)


def main():
    path = args.dir
    print(f"path = {path}")

    for root, d_names, f_names in os.walk(path):
        for f in f_names:
            file = os.path.join(root, f)
            if (fnmatch.fnmatch(f, "*.c") or fnmatch.fnmatch(f, "*.cc")
                    or fnmatch.fnmatch(f, "*.cpp")
                    or fnmatch.fnmatch(f, "*.h")
                    or fnmatch.fnmatch(f, "*.hpp")):
                add_header(file, "cc")
            elif fnmatch.fnmatch(f, "*.py"):
                add_header(file, "py")
            elif fnmatch.fnmatch(f, "*.sh"):
                add_header(file, "sh")
            elif fnmatch.fnmatch(f, "*.tex"):
                add_header(file, "tex")


if __name__ == "__main__":
    main()
