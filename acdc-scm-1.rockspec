package = "acdc"
version = "scm-1"

source = {
   url = "https://github.com/mdenil/acdc-torch",
}

description = {
   summary = "ACDC",
   detailed = [[
   	    Implements the layer described in:

        ACDC: A Structured Efficient Linear Layer

        http://arxiv.org/abs/1511.05946
   ]],
   homepage = "https://github.com/mdenil/acdc-torch"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
