import re

###############################
#
#   Name of the folder with the contribution sources
#
###############################
package_name = 'pyronn_layers'

###############################
#
#   Build all .cc, .cu, .h files from the contribution folder (package_name) into one .so file
#
###############################
def build():
    file_path = '../../tensorflow/tools/pip_package/BUILD'
    tf_build = open(file_path,'r')
    build = tf_build.read()
    tf_build.close()
    pattern = re.compile('sh_binary.*?COMMON_PIP_DEPS.*?\],\n', re.DOTALL)

    for m in pattern.finditer(build):
        s = m.start()
        e = m.end()
        g = m.group()

        start_pos = e - 4
        build = build[:start_pos] + '"//'+package_name+':'+package_name+'.so",\n' + build[start_pos:]
        tf_build_w = open(file_path, 'w')
        tf_build_w.write(build)
        tf_build_w.flush()
        tf_build_w.close()

###################################
#
#   Recursivly copy the compiled files into the respective subfolder in preparation to build the pip_package
#
###################################

def build_pip_package():
    file_path = '../../tensorflow/tools/pip_package/build_pip_package.sh'
    tf_build = open(file_path,'r')
    build = tf_build.read()
    tf_build.close()
    pattern = re.compile('function prepare_src\(\).*?\}\n\nfunction build_wheel\(\)', re.DOTALL)

    for m in pattern.finditer(build):
        pointer = m.start()
        offset = m.group().rfind('}')


        insert_position = pointer+offset-1

        build = build[:insert_position] + '\n  cp -r '+package_name+' ${TMPDIR}\n  cp -r bazel-bin/'+package_name+'/*.so ${TMPDIR}/'+package_name + build[insert_position:]

        tf_build_w = open(file_path, 'w')
        tf_build_w.write(build)
        tf_build_w.flush()
        tf_build_w.close()

###################################
#
#   Append our .so file to the build process of the pip package
#
###################################
def setup():
    file_path = '../../tensorflow/tools/pip_package/setup.py'
    tf_build = open(file_path, 'r')
    build = tf_build.read()
    tf_build.close()
    pattern = re.compile('headers.*?setup\(\n', re.DOTALL)

    for m in pattern.finditer(build):
        pointer = m.start()
        offset = m.group().rfind(')')

        insert_position = pointer + offset +1

        build = build[:insert_position] + '\n\npackages_to_install = find_packages()\npackages_to_install.append(\''+package_name+'\')' + build[insert_position:]
        tf_build_w = open(file_path, 'w')
        tf_build_w.write(build)
        tf_build_w.flush()
        tf_build_w.close()


###################################
#
#   Tensorflow 1.9 containts a Bug for user locations than us_us, which leads to a wrong parsing of the us decimal sign '.' to ','
#    ,which later in the run process leads to wrong interpretations as next argument instead of decimal sign. This method applies
#   a valid bugfix from the tensorflow github issue:
#       https://github.com/tensorflow/tensorflow/pull/22044/commits/ce9e5b035b32ef02cd7d10f6ffdd27cc2a75664d
#   Original issue:
#       https://github.com/tensorflow/tensorflow/issues/21164
#
###################################
def location_bugfix():
    file_path = "../tensorflow/python/framework/python_op_gen_internal.cc"
    tf_build = open(file_path, 'r')
    build = tf_build.read()
    tf_build.close()
    build_updated = build.replace('return strings::StrCat(value.f());',
                                  'std::ostringstream s;\n s.imbue(std::locale::classic());\n s << value.f();\n return s.str();')
    tf_build_w = open(file_path, 'w')
    tf_build_w.write(build_updated)
    tf_build_w.flush()
    tf_build_w.close()

print('Prepare BUILD in tensorflow/tools/pip_package/')
build()
print('Prepare build_pip_package.sh in tensorflow/tools/pip_package/')
build_pip_package()
print('Prepare setup.py in tensorflow/tools/pip_package/')
setup()
print('Apply bugfix for Tensorflow regarding String parsing w.r.t to location')
location_bugfix()
print('Tensorflow is patched')
