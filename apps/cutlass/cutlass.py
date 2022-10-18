import tvm

OBJ_FILE = "/tmp/main.o"


def load_cutlass():
    return tvm.runtime.load_module(OBJ_FILE)


def main():
    mod_cutlass = load_cutlass()
    print(mod_cutlass["HGEMM"])


if __name__ == "__main__":
    main()
