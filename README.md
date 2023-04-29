# Simple likelihood-ratio confidence intervals example.

First things first: this is a python project, but it involves a C extension module (`profile_likelihood.c`), which needs to be compiled.  This is not a project that really needs to be "installed", so I feel it is best to just compile this file by hand at the command line (the more "proper" way to do it is to use `distutils` to do this automatically with a `pip install` command, were this a deployable package).  Compiling this will produce a shared object file; on Mac and Linux, the compiled file will carry a `.so` extension.  However, on Windows, I think it would need to be a `.dyld` file; I don't have access to a Windows machine, so I haven't figured out how to do this.  These compile instructions pertain to Mac and Linux.  On both OSs, the C libraries need to be findable by the compiler, which will look into the `C_INCLUDE_PATH` bash environment variable (which acts similar to the `PATH` environment variable).  You can see what's already in that variable by typing at the command line:

```
echo $C_INCLUDE_PATH
```

There are two include paths that the compiler will need to take from: the general Python installation, and from Numpy.  In order to find the appropriate paths for these two libraries, open a python3 instance and type the following:

General Python import path:
```
>>> from distutils.sysconfig import get_python_inc
>>> print(get_python_inc())
```

Numpy import path:
```
>>> from numpy import get_include as npy_get_include
>>> print(npy_get_include())
```

If either of these [full] paths are not already in the `C_INCLUDE_PATH` variable, they needed to be added in the same way that one adds paths to the `PATH` variable.  For example, imagine the Python path (returned by the first python command above) is:

```
/home/my_username/.local/lib/python3.10/site-packages/numpy/core/include/:/usr/include/python3.10/
```

then in your `.bashrc` file, the following line needs to be added:

```
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/home/my_username/.local/lib/python3.10/site-packages/numpy/core/include/:/usr/include/python3.10/
```

and the same for the Numpy path.  Once this is done, the following lines will perform the compilation:

Mac:
```
clang -shared -undefined dynamic_lookup -o profile_likelihood.so profile_likelihood.c
```

Linux:
```
gcc -shared -o profile_likelihood.so -fPIC profile_likelihood.c
```

If the compilation has produced a `profile_likelihood.so` file, one can check that it's working correctly by opening a Python terminal in the same directory and typing:

```
>>> import profile_likelihood
```

If the above command completed without complaint, then everything should be working fine.
