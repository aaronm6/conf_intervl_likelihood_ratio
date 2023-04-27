import os as _os
import sys as _sys
fname_base = 'profile_likelihood'
so_name = f'{fname_base}.so'

if so_name not in _os.listdir():
    print(f"\n{so_name} does not exist, and must be compiled.  Make sure the following")
    print("paths are added to the C_INCLUDE_PATH environment variable: \n")
    from distutils.sysconfig import get_python_inc
    from numpy import get_include as npy_get_include
    print(get_python_inc())
    print(npy_get_include())
    print("\nCompile the .c file to .so via this command:")
    print(f"* macOS:\n     clang -shared -undefined dynamic_lookup -o {so_name} {fname_base}.c")
    print(f"* Linux:\n     gcc -shared -o {so_name} -fPIC {fname_base}.c")
    print(f"* Windows:\n     Use a different computer. Or google how to compile C code for use in Python")
    print("     Windows doesn't use .so files, so this file will have to change.")
    _sys.exit()

def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, importlib.util
    __file__ = pkg_resources.resource_filename(__name__, so_name)
    __loader__ = None; del __bootstrap__, __loader__
    spec = importlib.util.spec_from_file_location(__name__,__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
__bootstrap__()




